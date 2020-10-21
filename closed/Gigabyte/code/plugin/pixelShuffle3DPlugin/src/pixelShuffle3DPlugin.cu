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

#include "pixelShuffle3DPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::pixelShuffle3DPlugin;
using nvinfer1::plugin::pixelShuffle3DPluginCreator;

#define CHECK_CUDA(call)                                                                                               \
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
    } while (0)


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

namespace {
    const char* PIXELSHUFFLE3D_PLUGIN_VERSION{"1"};
    const char* PIXELSHUFFLE3D_PLUGIN_NAME{"PIXELSHUFFLE3D_TRT"};
}

REGISTER_TENSORRT_PLUGIN(pixelShuffle3DPluginCreator);

PluginFieldCollection pixelShuffle3DPluginCreator::mFC{};
std::vector<PluginField> pixelShuffle3DPluginCreator::mPluginAttributes;

pixelShuffle3DPlugin::pixelShuffle3DPlugin(
    int r, int s, int t)
    : mR(r)
    , mS(s)
    , mT(t)
    , mInScale(-1.f)
    , mOutScale(-1.f)
{
}

pixelShuffle3DPlugin::pixelShuffle3DPlugin(void const* serialData, size_t serialLength)
{
    deserialize_value(&serialData, &serialLength, &mR);
    deserialize_value(&serialData, &serialLength, &mS);
    deserialize_value(&serialData, &serialLength, &mT);
    deserialize_value(&serialData, &serialLength, &mInScale);
    deserialize_value(&serialData, &serialLength, &mOutScale);
}

pixelShuffle3DPlugin::~pixelShuffle3DPlugin()
{
    terminate();
}

// pixelShuffle3DPlugin returns one output.
int pixelShuffle3DPlugin::getNbOutputs() const
{
    return 1;
}

DimsExprs pixelShuffle3DPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
{
    nvinfer1::DimsExprs output(inputs[0]);

    output.d[0] = inputs[0].d[0];
    output.d[1] = exprBuilder.operation(DimensionOperation::kFLOOR_DIV,  *inputs[0].d[1], *exprBuilder.constant(mR * mS * mT));
    output.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *exprBuilder.constant(mR));
    output.d[3] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[3], *exprBuilder.constant(mS));
    output.d[4] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[4], *exprBuilder.constant(mT));

    return output;
}

int pixelShuffle3DPlugin::initialize()
{
    if (!initialized)
    {
    }
    initialized = true;
    return 0;
}

void pixelShuffle3DPlugin::terminate()
{
    if (initialized)
    {
    }
    initialized = false;
    return;
}

size_t pixelShuffle3DPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const 
{ 
    return 0;
}


int pixelShuffle3DPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    ASSERT(initialized);

    if (inputDesc[0].format == nvinfer1::PluginFormat::kLINEAR || inputDesc[0].format == nvinfer1::PluginFormat::kCDHW32)
    {

        nvinfer1::Dims input_dims = inputDesc[0].dims;
        int n = input_dims.d[0];
        int c = input_dims.d[1];
        int d = input_dims.d[2];
        int h = input_dims.d[3];
        int w = input_dims.d[4];

        _params.o = d * mR;
        _params.p = h * mS;
        _params.q = w * mT;
        _params.k = c/mT/mR/mS;
        _params.n = n;
        _params.r = mR;
        _params.s = mS;
        _params.t = mT;
        _params.scale = mInScale / mOutScale;


        _params.gmem_src = const_cast<void *>(inputs[0]);
        _params.gmem_dst = outputs[0];

        if (inputDesc[0].format == nvinfer1::PluginFormat::kCDHW32)
        {
            assert(mOutScale != 0.f);
            int res = pixel_shuffle_ncdhw32_to_ncdhw32_dispatch(_params, stream);
        }
        else
        {
            int res = pixel_shuffle_ncdhw_to_ncdhw_dispatch(_params, stream);
        }
    }
    else
    {
        ASSERT(false && "Unexpected input format");
    }

    return 0;
}

size_t pixelShuffle3DPlugin::getSerializationSize() const
{
    return (serialized_size(mR) +
            serialized_size(mS) +
            serialized_size(mT) +
            serialized_size(mInScale) +
            serialized_size(mOutScale));
}

void pixelShuffle3DPlugin::serialize(void *buffer) const
{
    serialize_value(&buffer, mR);
    serialize_value(&buffer, mS);
    serialize_value(&buffer, mT);
    serialize_value(&buffer, mInScale);
    serialize_value(&buffer, mOutScale);
}

bool pixelShuffle3DPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
{
    ASSERT(inOut && pos < (nbInputs + nbOutputs));

    bool support_fp32_linear = (inOut[pos].type == nvinfer1::DataType::kFLOAT
        && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR
        && inOut[pos].type == inOut[0].type
        && inOut[pos].format == inOut[0].format);

    bool support_int8_cdhw32 = (inOut[pos].type == nvinfer1::DataType::kINT8
        && inOut[pos].format == nvinfer1::PluginFormat::kCDHW32
        && inOut[pos].type == inOut[0].type
        && inOut[pos].format == inOut[0].format);

    return support_fp32_linear || support_int8_cdhw32;
}

const char* pixelShuffle3DPlugin::getPluginType() const
{
    return PIXELSHUFFLE3D_PLUGIN_NAME;
}

const char* pixelShuffle3DPlugin::getPluginVersion() const
{
    return PIXELSHUFFLE3D_PLUGIN_VERSION;
}

void pixelShuffle3DPlugin::destroy()
{ 
    delete this;
}

IPluginV2DynamicExt* pixelShuffle3DPlugin::clone() const
{ 
    auto plugin = new pixelShuffle3DPlugin{mR, mS, mT};
    plugin->setPluginNamespace(mPluginNamespace);
    plugin->initialize();
    return plugin;
}

// Set plugin namespace
void pixelShuffle3DPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* pixelShuffle3DPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType pixelShuffle3DPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(inputTypes && nbInputs > 0 && index == 0);

    return nvinfer1::DataType::kFLOAT;
}

void pixelShuffle3DPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs)
{
    mInScale = in[0].desc.scale;
    mOutScale = out[0].desc.scale;
}

// pixelShuffle3DPluginCreator methods
pixelShuffle3DPluginCreator::pixelShuffle3DPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("R", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("S", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("T", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* pixelShuffle3DPluginCreator::getPluginName() const
{
    return PIXELSHUFFLE3D_PLUGIN_NAME;
}

const char* pixelShuffle3DPluginCreator::getPluginVersion() const
{
    return PIXELSHUFFLE3D_PLUGIN_VERSION;
}

const PluginFieldCollection* pixelShuffle3DPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2DynamicExt* pixelShuffle3DPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    int r {};
    int s {};
    int t {};
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "R"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            r = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "S"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            s = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "T"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            t = *(static_cast<const int*>(fields[i].data));
        }
    }

    pixelShuffle3DPlugin* obj = new pixelShuffle3DPlugin(r, s, t);
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}

IPluginV2DynamicExt* pixelShuffle3DPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    pixelShuffle3DPlugin* obj = new pixelShuffle3DPlugin{serialData, serialLength}; 
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}
