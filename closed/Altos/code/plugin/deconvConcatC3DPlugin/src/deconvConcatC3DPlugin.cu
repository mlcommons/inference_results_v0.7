/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdexcept>

#include "deconvConcatC3DPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::DeconvConcatC3DPlugin;
using nvinfer1::plugin::DeconvConcatC3DPluginCreator;

const char* deconv_xmma_lib_file = "deconv.so";

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call) do { \
    cudaError_t status_ = call; \
    if( status_ != cudaSuccess ) { \
      fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(status_)); \
      exit(1); \
    } \
  } while(0)
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  
  #define CHECK_CUDNN(call) do { \
    cudnnStatus_t status_ = call; \
    if( status_ != CUDNN_STATUS_SUCCESS ) { \
      fprintf(stderr, "CUDNN Error at line %d: %s\n", __LINE__, cudnnGetErrorString(status_)); \
      exit(1); \
    } \
  } while(0)
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////
/* #define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

#define CHECK_CUDNN(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        cudnnStatus_t status = call;                                                                                   \
        if (status != CUDNN_STATUS_SUCCESS)                                                                            \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0) */

inline int get_deconv_output_size(int P, int paddingTop, int paddingBottom, int strideH, int dilationH, int R)
{
    return (P - 1) * strideH - paddingTop - paddingBottom + (R - 1) * dilationH + 1; 
}

const IDimensionExpr * get_deconv_output_dim_expr( const IDimensionExpr * P, 
    int paddingTop, int paddingBottom, int strideH, int dilationH, int R,
    nvinfer1::IExprBuilder& exprBuilder)
{
    auto res = exprBuilder.operation(DimensionOperation::kSUB, *P, *exprBuilder.constant(1));
    res = exprBuilder.operation(DimensionOperation::kPROD, *res, *exprBuilder.constant(strideH));
    res = exprBuilder.operation(DimensionOperation::kSUM, *res, *exprBuilder.constant((R - 1) * dilationH + 1 - paddingTop - paddingBottom));
    return res;
}

// This is derived from: https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
inline float half_to_float_fast(unsigned short value)
{
    union F32
    {
        unsigned int u;
        float f;
    };
    static const F32 magic = {(254 - 15) << 23};
    static const F32 was_infnan = {(127 + 16) << 23};
    F32 result;
    result.u = (value & 0x7fff) << 13; // exponent/mantissa bits
    result.f *= magic.f;               // exponent adjust
    if (result.f >= was_infnan.f)
    { // make sure Inf/NaN survive
        result.u |= 255 << 23;
    }
    result.u |= (value & 0x8000) << 16; // sign bit
    return result.f;
}

cudnnStatus_t convert_trt2cudnn_dtype(nvinfer1::DataType trt_dtype, cudnnDataType_t* cudnn_dtype)
{
    switch (trt_dtype)
    {
    case nvinfer1::DataType::kFLOAT: *cudnn_dtype = CUDNN_DATA_FLOAT; break;
    case nvinfer1::DataType::kHALF: *cudnn_dtype = CUDNN_DATA_HALF; break;
    default: return CUDNN_STATUS_BAD_PARAM;
    }
    return CUDNN_STATUS_SUCCESS;
}

namespace {
    const char* DECONVCONCATC3D_PLUGIN_VERSION{"1"};
    const char* DECONVCONCATC3D_PLUGIN_NAME{"DECONVCONCATC3D_TRT"};
}

REGISTER_TENSORRT_PLUGIN(DeconvConcatC3DPluginCreator);

PluginFieldCollection DeconvConcatC3DPluginCreator::mFC{};
std::vector<PluginField> DeconvConcatC3DPluginCreator::mPluginAttributes;

void DeconvConcatC3DPlugin::initDeconvXmma( int num_output_maps_with_stride ) {
#ifdef USE_XMMA_DECONV_LIB
    ASSERT(NDHWC_CUDNN == 0);
    deconvXmmaCreate(&_deconv_xmma_handle, _num_groups, _num_output_maps, 
        num_output_maps_with_stride, _num_input_maps, 3, 
        &_kernel_size_nd[0], &_stride_nd[0],  &_padding_nd[0],
        &_dilation_nd[0], true);
#endif
}

DeconvConcatC3DPlugin::DeconvConcatC3DPlugin(
    int num_output_maps,
    int num_groups,
    int kernel_id,
    const std::vector<float>& kernel,
    //const std::vector<float>& bias,
    const std::vector<int>& kernel_size_nd,
    const std::vector<int>& stride_nd,
    const std::vector<int>& padding_nd,
    const std::vector<int>& dilation_nd)
    : _num_output_maps(num_output_maps)
    , _num_groups(num_groups)
    , _kernel_id(kernel_id)
    , _nbInputs(1)
    , _kernel_h(kernel)
    , _bias_h(0)
    , _kernel_size_nd(kernel_size_nd)
    , _stride_nd(stride_nd)
    , _padding_nd(padding_nd)
    , _dilation_nd(dilation_nd)
{
    ASSERT(_bias_h.size() == 0);
}

DeconvConcatC3DPlugin::DeconvConcatC3DPlugin(
    int num_output_maps,
    int num_groups,
    int kernel_id,
    nvinfer1::Weights const& kernel,
    //nvinfer1::Weights const& bias,
    const nvinfer1::Dims& kernel_size_nd,
    const nvinfer1::Dims& stride_nd,
    const nvinfer1::Dims& padding_nd,
    const nvinfer1::Dims& dilation_nd)
    : _num_output_maps(num_output_maps)
    , _num_groups(num_groups)
    , _kernel_id(kernel_id)
    , _nbInputs(1)
{
    //ASSERT((bias.count == 0) || (bias.values == nullptr));

    if (kernel.type == nvinfer1::DataType::kFLOAT)
    {
        _kernel_h.assign((float*) kernel.values, (float*) kernel.values + kernel.count);
    }
    else if (kernel.type == nvinfer1::DataType::kHALF)
    {
        _kernel_h.reserve(kernel.count);
        for (int64_t c = 0; c < kernel.count; ++c)
        {
            unsigned short value = ((unsigned short*) kernel.values)[c];
            _kernel_h.push_back(half_to_float_fast(value));
        }
    }
    else
    {
        throw std::runtime_error("Unsupported kernel dtype");
    }
#if 0
    if (bias.type == nvinfer1::DataType::kFLOAT)
    {
        //_bias_h.assign((float*) bias.values, (float*) bias.values + bias.count);
    }
    else if (bias.type == nvinfer1::DataType::kHALF)
    {
/*         _bias_h.reserve(bias.count);
        for (int c = 0; c < bias.count; ++c)
        {
            unsigned short value = ((unsigned short*) bias.values)[c];
            _bias_h.push_back(half_to_float_fast(value));
        } */
    }
    else
    {
        throw std::runtime_error("Unsupported bias dtype");
    }
#endif
    _kernel_size_nd.resize(kernel_size_nd.nbDims);
    for (int i = 0; i < kernel_size_nd.nbDims; i++)
    {
        _kernel_size_nd[i] = kernel_size_nd.d[i];
    }

    _stride_nd.resize(stride_nd.nbDims);
    for (int i = 0; i < stride_nd.nbDims; i++)
    {
        _stride_nd[i] = stride_nd.d[i];
    }

    _padding_nd.resize(padding_nd.nbDims);
    for (int i = 0; i < padding_nd.nbDims; i++)
    {
        _padding_nd[i] = padding_nd.d[i];
    }

    _dilation_nd.resize(dilation_nd.nbDims);
    for (int i = 0; i < dilation_nd.nbDims; i++)
    {
        _dilation_nd[i] = dilation_nd.d[i];
    }
}

DeconvConcatC3DPlugin::DeconvConcatC3DPlugin(void const* serialData, size_t serialLength)
{
    deserialize_value(&serialData, &serialLength, &_num_output_maps);
    deserialize_value(&serialData, &serialLength, &_num_groups);
    deserialize_value(&serialData, &serialLength, &_kernel_id);
    deserialize_value(&serialData, &serialLength, &_kernel_h);
    //deserialize_value(&serialData, &serialLength, &_bias_h);
    deserialize_value(&serialData, &serialLength, &_kernel_size_nd);
    deserialize_value(&serialData, &serialLength, &_stride_nd);
    deserialize_value(&serialData, &serialLength, &_padding_nd);
    deserialize_value(&serialData, &serialLength, &_dilation_nd);
}

DeconvConcatC3DPlugin::~DeconvConcatC3DPlugin()
{
    terminate();
}

// DeconvConcatC3DPlugin returns one output.
int DeconvConcatC3DPlugin::getNbOutputs() const
{
    return 1;
}

DimsExprs DeconvConcatC3DPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    auto one = exprBuilder.constant(1);
    nvinfer1::DimsExprs output(inputs[0]);
    if (nbInputs == 1) {
        output.d[1] = exprBuilder.constant(_num_output_maps);
        //output.d[1] = inputs[0].d[0];
    } else if (nbInputs == 2) { 
        //output.d[1] = exprBuilder.operation(DimensionOperation::kSUM, *exprBuilder.constant(_num_output_maps), *inputs[1].d[1]);
        output.d[1] = exprBuilder.operation(DimensionOperation::kSUM, *inputs[1].d[1], *inputs[1].d[1]);
    } else {
        ASSERT( nbInputs == 1 || nbInputs == 2 );
    }
    output.d[2] = get_deconv_output_dim_expr(inputs[0].d[2], _padding_nd[0], _padding_nd[0], 
                                             _stride_nd[0], _dilation_nd[0], _kernel_size_nd[0], exprBuilder);
    output.d[3] = get_deconv_output_dim_expr(inputs[0].d[3], _padding_nd[1], _padding_nd[1], 
                                             _stride_nd[1], _dilation_nd[1], _kernel_size_nd[1], exprBuilder);
    output.d[4] = get_deconv_output_dim_expr(inputs[0].d[4], _padding_nd[2], _padding_nd[2], 
                                             _stride_nd[2], _dilation_nd[2], _kernel_size_nd[2], exprBuilder);

    //std::copy(inputs[1].d + 2, inputs[1].d + inputs[1].nbDims, output.d + 2);

    _nbInputs = nbInputs;
    return output;
}

void DeconvConcatC3DPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    _filter_dim = {_num_input_maps, _num_output_maps, _kernel_size_nd[0], _kernel_size_nd[1], _kernel_size_nd[2]};

    _input_tensor_format = in[0].desc.format;
    _trt_data_type = in[0].desc.type;
    _cudnn_data_type = (_trt_data_type == nvinfer1::DataType::kHALF)? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
    _cudnn_tensor_format = CUDNN_TENSOR_NCHW;

    std::vector<__half> converted_kernel(_kernel_h.size());
    if (_input_tensor_format == nvinfer1::PluginFormat::kLINEAR && _trt_data_type == nvinfer1::DataType::kHALF){
        for (size_t i = 0; i < _kernel_h.size(); i++) {
            converted_kernel[i] = (__half)_kernel_h[i];
        }
    }
    else if (_input_tensor_format == nvinfer1::PluginFormat::kDHWC8)
    {
        //printf("configurePlugin converting weights to KRSTC...\n");
        // convert to KRSC
        // this is NHWC case https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnSetFilter4dDescriptor for more info
        int kernel_size_mult = _kernel_size_nd[0] * _kernel_size_nd[1] * _kernel_size_nd[2];

        //printf("_num_input_maps = %d, _num_output_maps = %d, kernel_size_mult = %d\n", _num_input_maps, _num_output_maps, kernel_size_mult);
        for (int k = 0; k < _num_input_maps; k++)
        {
            for (int c = 0; c < _num_output_maps; c++)
            {
                for (int dhw = 0; dhw < kernel_size_mult; dhw++)
                {
                    int id_out =   k * _num_output_maps * kernel_size_mult
                                + dhw * _num_output_maps + c;
                    int id_in =    k * _num_output_maps * kernel_size_mult
                                + c *  kernel_size_mult + dhw;
                    converted_kernel[id_out] =  (__half)_kernel_h[id_in];
                    //converted_kernel[id_in] =  (__half)_kernel_h[id_in];
                }
            }
        }

        //_filter_dim = {_num_input_maps, _kernel_size_nd[0], _kernel_size_nd[1], _kernel_size_nd[2], _num_output_maps};
        _cudnn_tensor_format = CUDNN_TENSOR_NHWC;

#ifdef USE_XMMA_DECONV_LIB
        int n_max = 1;
        int d_max = out[0].max.d[2];
        int h_max = out[0].max.d[3];
        int w_max = out[0].max.d[4];
        int stride_c = _num_output_maps;
        if (nbInputs == 2) {
            stride_c = _num_output_maps + in[1].max.d[1];
        }
        bool use_idx_kernels = false;
        deconvXmmaConfigure(_deconv_xmma_handle, n_max, d_max, h_max, w_max, stride_c, use_idx_kernels, &_kernel_id);
#endif
    }

    if (_trt_data_type == nvinfer1::DataType::kHALF) {
        std::vector<__half> converted_bias(_bias_h.size());
        for (size_t i = 0; i < _bias_h.size(); i++) {
            converted_bias[i] = (__half)_bias_h[i];
        }
        CHECK_CUDA(cudaMemcpy(_kernel_d, &converted_kernel[0], converted_kernel.size()*sizeof(uint16_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(_bias_d, &converted_bias[0], converted_bias.size()*sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
    else {
        CHECK_CUDA(cudaMemcpy(_kernel_d, &_kernel_h[0], _kernel_h.size()*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(_bias_d, &_bias_h[0], _bias_h.size()*sizeof(float), cudaMemcpyHostToDevice));
    }

    CHECK_CUDNN(cudnnSetFilterNdDescriptor(_kernel_desc,_cudnn_data_type,_cudnn_tensor_format,5,&_filter_dim[0]));
    _conv_algo = (_cudnn_data_type == CUDNN_DATA_FLOAT)? CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 : CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    _cudnn_math_type = (_cudnn_tensor_format == CUDNN_TENSOR_NHWC)? CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION : CUDNN_DEFAULT_MATH;
    CHECK_CUDNN(cudnnSetConvolutionMathType(_conv_desc,_cudnn_math_type));

    _nbInputs = nbInputs;
}

int DeconvConcatC3DPlugin::initialize()
{
    if (!initialized)
    {
        _num_input_maps = _kernel_h.size() / _num_output_maps / _kernel_size_nd[0] / _kernel_size_nd[1] / _kernel_size_nd[2];

        CHECK_CUDA(cudaStreamCreate(&_concat_stream));
        CHECK_CUDA(cudaEventCreateWithFlags(&_concat_event, cudaEventDisableTiming ));

        // FLOAT data path
        CHECK_CUDNN(cudnnCreate(&_cudnn_handle));

        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&_conv_desc));
        cudnnConvolutionMode_t conv_mode = CUDNN_CROSS_CORRELATION;

        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(_conv_desc,
            3,
            &_padding_nd[0],
            &_stride_nd[0],
            &_dilation_nd[0],
            conv_mode,
            CUDNN_DATA_FLOAT));
        CHECK_CUDNN(cudnnSetConvolutionGroupCount(_conv_desc, _num_groups));

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&_error_desc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&_kernel_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&_image_desc));

        // allocate more than might be needed
        CHECK_CUDA(cudaMalloc(&_kernel_d, _kernel_h.size()*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&_bias_d, _bias_h.size()*sizeof(float)));

        _cudnn_workspace_sz = 0;

        initDeconvXmma(_num_output_maps); // !!! change to 2*_num_output_maps support concat
    }

    initialized = true;

    return 0;
}

void DeconvConcatC3DPlugin::terminate()
{
    if (initialized)
    {
        CHECK_CUDA(cudaStreamDestroy(_concat_stream));
        CHECK_CUDA(cudaEventDestroy(_concat_event));

#ifdef USE_XMMA_DECONV_LIB
        deconvXmmaDestroy(_deconv_xmma_handle);
#endif
        cudnnDestroyTensorDescriptor(_image_desc);
        cudnnDestroyFilterDescriptor(_kernel_desc);
        cudnnDestroyTensorDescriptor(_error_desc);
        cudnnDestroyConvolutionDescriptor(_conv_desc);

        cudnnDestroy(_cudnn_handle);

        cudaFree(_bias_d);
        cudaFree(_kernel_d);
    }
    initialized = false;
    return;
}

size_t DeconvConcatC3DPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const { 

    nvinfer1::Dims     input_dims = inputs[0].dims;

    int n = input_dims.d[0];
    int k = input_dims.d[1];
    int o = input_dims.d[2];
    int p = input_dims.d[3];
    int q = input_dims.d[4];

/*     for (int i = 0; i < 3; i++) {
        printf("_padding_nd[%d] = %d ", i, _padding_nd[i]);
        printf("_stride_nd[%d] = %d ", i, _stride_nd[i]);
        printf("_kernel_size_nd[%d] = %d ", i, _kernel_size_nd[i]);
    }
    printf("\n"); */

    int c = _num_output_maps;
    int d = get_deconv_output_size(o, _padding_nd[0], _padding_nd[0], _stride_nd[0], _dilation_nd[0], _kernel_size_nd[0]);
    int h = get_deconv_output_size(p, _padding_nd[1], _padding_nd[1], _stride_nd[1], _dilation_nd[1], _kernel_size_nd[1]);
    int w = get_deconv_output_size(q, _padding_nd[2], _padding_nd[2], _stride_nd[2], _dilation_nd[2], _kernel_size_nd[2]);

    //printf("n,c,d,h,w = %d, %d, %d, %d, %d\n", n,c,d,h,w);
    //printf("n,k,o,p,q = %d, %d, %d, %d, %d\n", n,k,o,p,q);

    nvinfer1::DataType trt_data_type = inputs[0].type;
    cudnnTensorFormat_t cudnn_tensor_format = (inputs[0].format == nvinfer1::PluginFormat::kDHWC8) ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
    cudnnDataType_t cudnn_data_type = (trt_data_type == nvinfer1::DataType::kHALF)? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;

    cudnnConvolutionBwdDataAlgo_t conv_algo = (_cudnn_data_type == CUDNN_DATA_FLOAT)? CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 : CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

    int image_dim[] = {n, c, d, h, w};
    CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(_image_desc,cudnn_tensor_format,cudnn_data_type,5,&image_dim[0]));

/*     int filter_dim[] = {k, c, _kernel_size_nd[0], _kernel_size_nd[1], _kernel_size_nd[2]};
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(_kernel_desc,cudnn_data_type,cudnn_tensor_format,5,&filter_dim[0])); */

    int error_dim[] = {n, k, o, p, q};
    CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(_error_desc,cudnn_tensor_format,cudnn_data_type,5,&error_dim[0]));

    cudnnMathType_t cudnn_math_type = (cudnn_tensor_format == CUDNN_TENSOR_NHWC)? CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION : CUDNN_DEFAULT_MATH;
    CHECK_CUDNN(cudnnSetConvolutionMathType(_conv_desc,cudnn_math_type));

    size_t cudnn_workspace_sz    = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(_cudnn_handle,
            _kernel_desc,
            _error_desc,
            _conv_desc,
            _image_desc,
            conv_algo,
            &cudnn_workspace_sz));

    if (nbInputs == 2) {
        // may lead to very large tmp requirement
        //cudnn_workspace_sz += n*c*d*h*w*((_cudnn_data_type == CUDNN_DATA_HALF)? sizeof(uint16_t) : sizeof(float));
    }
    if (inputs[0].format == nvinfer1::PluginFormat::kLINEAR)
    {
        return cudnn_workspace_sz;
    }
    else if (inputs[0].format == nvinfer1::PluginFormat::kDHWC8)
    {
#if (NDHWC_CUDNN == 1)
        return cudnn_workspace_sz;
#else
#ifdef USE_XMMA_DECONV_LIB
        size_t workspace_sz = 0;
        deconvXmmaGetWorkspaceSize( _deconv_xmma_handle, n, d, h, w, &workspace_sz );
        return workspace_sz;
#endif
        ASSERT(0);
#endif
    }
    else
    {
        ASSERT(0);
    }
}


int DeconvConcatC3DPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    ASSERT(initialized);

    nvinfer1::Dims     input_dims = inputDesc[0].dims;

    int n = input_dims.d[0];
    int k = input_dims.d[1];
    int o = input_dims.d[2];
    int p = input_dims.d[3];
    int q = input_dims.d[4];

    int c = _num_output_maps;
    int d = get_deconv_output_size(o, _padding_nd[0], _padding_nd[0], _stride_nd[0], _dilation_nd[0], _kernel_size_nd[0]);
    int h = get_deconv_output_size(p, _padding_nd[1], _padding_nd[1], _stride_nd[1], _dilation_nd[1], _kernel_size_nd[1]);
    int w = get_deconv_output_size(q, _padding_nd[2], _padding_nd[2], _stride_nd[2], _dilation_nd[2], _kernel_size_nd[2]);

    void const* error_d = inputs[0];
    void* image_d = outputs[0];

    int n2 = 0, c2 = 0, d2 = 0, h2 = 0, w2 = 0;
    size_t dst_pitch = 0, src_pitch = 0, transfer_width = 0, transfer_height = 0, 
           src_pitch_2nd = 0, transfer_width_2nd = 0, transfer_height_2nd = 0;
    int sizeof_datatype = (inputDesc[0].type == DataType::kHALF)? sizeof(uint16_t) : sizeof(float);
    if (_nbInputs == 2) {
        nvinfer1::Dims     input2nd_dims = inputDesc[1].dims;
        n2 = input2nd_dims.d[0];
        c2 = input2nd_dims.d[1];
        d2 = input2nd_dims.d[2];
        h2 = input2nd_dims.d[3];
        w2 = input2nd_dims.d[4];

        if (inputDesc[0].format == nvinfer1::PluginFormat::kLINEAR) {
            dst_pitch = (c * d * h * w + c2 * d2 * h2 * w2) * sizeof_datatype;
            src_pitch = c * d * h * w * sizeof_datatype;
            transfer_width = c * d * h * w * sizeof_datatype;
            transfer_height = n;

            src_pitch_2nd = c2 * d2 * h2 * w2 * sizeof_datatype;
            transfer_width_2nd = c2 * d2 * h2 * w2 * sizeof_datatype;
            transfer_height_2nd = n2;
        }
        else {
            dst_pitch = (c + c2) * sizeof_datatype;
            src_pitch = c * sizeof_datatype;
            transfer_width = c * sizeof_datatype;
            transfer_height = n * d * h * w;

            src_pitch_2nd = c2 * sizeof_datatype;
            transfer_width_2nd = c2 * sizeof_datatype;
            transfer_height_2nd = n2 * d2 * h2 * w2;
        }
    }

    //std::cout << dst_pitch << "  " << src_pitch_2nd << "  " << transfer_width_2nd << " " << transfer_height_2nd << "\n";


/*     printf("n,c,d,h,w = %d, %d, %d, %d, %d\n", n,c,d,h,w);
    printf("n,k,o,p,q = %d, %d, %d, %d, %d\n", n,k,o,p,q); */
    if (inputDesc[0].format == nvinfer1::PluginFormat::kLINEAR || 
        (NDHWC_CUDNN == 1 && inputDesc[0].format == nvinfer1::PluginFormat::kDHWC8))
    {
        CHECK_CUDNN(cudnnSetStream(_cudnn_handle, stream));

        void* image_tmp_d = image_d;
        if (_nbInputs == 2) {
            // disable tmp workspace, since needs too much memory - TODO implement concat kernel
            //image_tmp_d = ((char*) workspace + _cudnn_workspace_sz);
        }

        float alpha = 1.f, beta = 0.f;

        _image_dim = {n, c, d, h, w};
        CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(_image_desc,_cudnn_tensor_format,_cudnn_data_type,5,&_image_dim[0]));


/*         _filter_dim = {k, c, _kernel_size_nd[0], _kernel_size_nd[1], _kernel_size_nd[2]};
        CHECK_CUDNN(cudnnSetFilterNdDescriptor(_kernel_desc,_cudnn_data_type,_cudnn_tensor_format,5,&_filter_dim[0])); */
        _error_dim = {n, k, o, p, q};
        CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(_error_desc,_cudnn_tensor_format,_cudnn_data_type,5,&_error_dim[0]));

        CHECK_CUDNN(cudnnSetConvolutionMathType(_conv_desc,_cudnn_math_type));
        //_cudnn_workspace_sz = 0;
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(_cudnn_handle,
                _kernel_desc,
                _error_desc,
                _conv_desc,
                _image_desc,
                _conv_algo,
                &_cudnn_workspace_sz));

        CHECK_CUDNN(cudnnConvolutionBackwardData(_cudnn_handle,
            &(alpha),
            _kernel_desc,
            _kernel_d,
            _error_desc,
            error_d,
            _conv_desc,
            _conv_algo,
            workspace,
            _cudnn_workspace_sz,
            &(beta),
            _image_desc,
            image_tmp_d));

        if (_nbInputs == 2) {
            // copy deconv output
/*             CHECK_CUDA(cudaMemcpy2DAsync(image_d, dst_pitch, image_tmp_d, 
            src_pitch, transfer_width, transfer_height, cudaMemcpyDeviceToDevice, stream)); */
            // workaround:
            for (int batch = transfer_height - 1; batch > 0; batch--) {
                CHECK_CUDA(cudaMemcpyAsync( (char *)image_d + batch * dst_pitch, (char *)image_d + batch * src_pitch,  transfer_width, cudaMemcpyDeviceToDevice, stream));
            }

            // TODO implement second concat copy in parallel stream
            void* image2nd_d = ((char *)image_d + transfer_width);
            void const* error2nd_d = inputs[1];
            CHECK_CUDA(cudaMemcpy2DAsync(image2nd_d, dst_pitch, error2nd_d,
            src_pitch_2nd, transfer_width_2nd, transfer_height_2nd, cudaMemcpyDeviceToDevice, stream));
        }
    }
    else if (inputDesc[0].format == nvinfer1::PluginFormat::kDHWC8)
    {
#if (NDHWC_CUDNN == 1)
        return 0;
#endif
#ifdef USE_XMMA_DECONV_LIB
        ASSERT(_kernel_id >= 0);
        deconvXmmaEnqueue(_deconv_xmma_handle, image_d, _kernel_d, error_d, nullptr, workspace, n, d, h, w, stream);

        if (_nbInputs == 2) {
            void* image2nd_d = ((char *)image_d + c * sizeof_datatype);
            void const* error2nd_d = inputs[1];
#ifdef USE_CONCAT_IN_SEPARATE_STREAM
            CHECK_CUDA(cudaMemcpy2DAsync(image2nd_d, dst_pitch, error2nd_d,
                  src_pitch_2nd, transfer_width_2nd, transfer_height_2nd, cudaMemcpyDeviceToDevice, 
                  _concat_stream));
            CHECK_CUDA(cudaEventRecord(_concat_event, _concat_stream));
            CHECK_CUDA(cudaStreamWaitEvent(stream, _concat_event, 0));
#else
            CHECK_CUDA(cudaMemcpy2DAsync(image2nd_d, dst_pitch, error2nd_d,
                src_pitch_2nd, transfer_width_2nd, transfer_height_2nd, cudaMemcpyDeviceToDevice, stream));
#endif
        }

        return 0;
#endif
        ASSERT(false && "Unexpected input format");
    }
    else
    {
        ASSERT(false && "Unexpected input format");
    }

    return 0;
}

size_t DeconvConcatC3DPlugin::getSerializationSize() const
{
    return (serialized_size(_num_output_maps) +
            serialized_size(_num_groups) +
            serialized_size(_kernel_id) +
            serialized_size(_kernel_h) +
            //serialized_size(_bias_h) +
            serialized_size(_kernel_size_nd) +
            serialized_size(_stride_nd) +
            serialized_size(_padding_nd) +
            serialized_size(_dilation_nd));
}

void DeconvConcatC3DPlugin::serialize(void *buffer) const
{
    serialize_value(&buffer, _num_output_maps);
    serialize_value(&buffer, _num_groups);
    serialize_value(&buffer, _kernel_id);
    serialize_value(&buffer, _kernel_h);
    //serialize_value(&buffer, _bias_h);
    serialize_value(&buffer, _kernel_size_nd);
    serialize_value(&buffer, _stride_nd);
    serialize_value(&buffer, _padding_nd);
    serialize_value(&buffer, _dilation_nd);
}

bool DeconvConcatC3DPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));

    bool support_fp32_linear = (inOut[pos].type == nvinfer1::DataType::kFLOAT
            && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
            && inOut[pos].type == inOut[0].type
            && inOut[pos].format == inOut[0].format);

    bool support_fp16_linear = (inOut[pos].type == nvinfer1::DataType::kHALF
            && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
            && inOut[pos].type == inOut[0].type
            && inOut[pos].format == inOut[0].format);

    bool support_fp16_dhwc8 = (inOut[pos].type == nvinfer1::DataType::kHALF
            && inOut[pos].format == nvinfer1::PluginFormat::kDHWC8
            && inOut[pos].type == inOut[0].type
            && inOut[pos].format == inOut[0].format);

    //return support_fp32_linear;
    //return support_fp16_dhwc8;
    //return support_fp16_linear;
    return support_fp32_linear || support_fp16_linear || support_fp16_dhwc8;
}

const char* DeconvConcatC3DPlugin::getPluginType() const
{
    return DECONVCONCATC3D_PLUGIN_NAME;
}

const char* DeconvConcatC3DPlugin::getPluginVersion() const
{
    return DECONVCONCATC3D_PLUGIN_VERSION;
}

void DeconvConcatC3DPlugin::destroy()
{ 
    delete this;
}

IPluginV2DynamicExt* DeconvConcatC3DPlugin::clone() const
{ 
    auto plugin = new DeconvConcatC3DPlugin{_num_output_maps, _num_groups, _kernel_id,
                                            _kernel_h, _kernel_size_nd, 
                                            _stride_nd, _padding_nd, _dilation_nd};
    plugin->setPluginNamespace(mPluginNamespace);
    plugin->initialize();

    return plugin;
}

// Set plugin namespace
void DeconvConcatC3DPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* DeconvConcatC3DPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType DeconvConcatC3DPlugin::getOutputDataType(

    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}

// DeconvConcatC3DPluginCreator methods
DeconvConcatC3DPluginCreator::DeconvConcatC3DPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("num_output_maps", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_groups", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("kernel_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("kernel", nullptr, PluginFieldType::kFLOAT32, 1));
    //mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("kernel_size_nd", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("stride_nd", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("padding_nd", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dilation_nd", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* DeconvConcatC3DPluginCreator::getPluginName() const
{
    return DECONVCONCATC3D_PLUGIN_NAME;
}

const char* DeconvConcatC3DPluginCreator::getPluginVersion() const
{
    return DECONVCONCATC3D_PLUGIN_VERSION;
}

const PluginFieldCollection* DeconvConcatC3DPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* DeconvConcatC3DPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    std::vector<float> kernel_values;
    //std::vector<float> bias_values;
    std::vector<int> kernel_size_nd;
    std::vector<int> stride_nd;
    std::vector<int> padding_nd;
    std::vector<int> dilation_nd;
    int num_output_maps {};
    int num_groups {};
    int kernel_id {};

    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "num_output_maps"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            num_output_maps= *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "num_groups"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            num_groups= *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "kernel_id"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            kernel_id= *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "kernel"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            kernel_values.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                kernel_values.push_back(*w);
                w++;
            }
        }
#if 0
        else if (!strcmp(attrName, "bias"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            bias_values.reserve(size);
            const auto* w = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                bias_values.push_back(*w);
                w++;
            }
        }
#endif
        else if (!strcmp(attrName, "kernel_size_nd"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            kernel_size_nd.reserve(size);
            const auto* w = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                kernel_size_nd.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "stride_nd"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            stride_nd.reserve(size);
            const auto* w = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                stride_nd.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "padding_nd"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            padding_nd.reserve(size);
            const auto* w = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                padding_nd.push_back(*w);
                w++;
            }
        }
        else if (!strcmp(attrName, "dilation_nd"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            dilation_nd.reserve(size);
            const auto* w = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                dilation_nd.push_back(*w);
                w++;
            }
        }
    }


    // TODO test constructor with TRT data structures
/*     Weights kernel_weights{DataType::kFLOAT, kernel_values.data(), (int64_t) kernel_values.size()};
       Weights bias_weights{DataType::kFLOAT, bias_values.data(), (int64_t) bias_values.size()}; 
       ... */

    DeconvConcatC3DPlugin* obj = new DeconvConcatC3DPlugin(num_output_maps, num_groups, kernel_id, kernel_values,
                                                           kernel_size_nd, stride_nd, padding_nd, dilation_nd);
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}

IPluginV2DynamicExt* DeconvConcatC3DPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    DeconvConcatC3DPlugin* obj = new DeconvConcatC3DPlugin{serialData, serialLength}; 
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}
