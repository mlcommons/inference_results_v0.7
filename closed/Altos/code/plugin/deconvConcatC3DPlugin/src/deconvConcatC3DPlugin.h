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
#ifndef TRT_DECONVOLUTION_CONCATENATION_C_3D_PLUGIN_H
#define TRT_DECONVOLUTION_CONCATENATION_C_3D_PLUGIN_H
#include "serialize.hpp"
#include "plugin.h"
#include <cudnn.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <string>

#ifdef USE_XMMA_DECONV_LIB
#define NDHWC_CUDNN 0
#else
#define NDHWC_CUDNN 1
#endif

#ifdef USE_XMMA_DECONV_LIB
#include "deconvXmmaApi.h"
#endif

typedef unsigned short half_type;

namespace nvinfer1
{
namespace plugin
{
class DeconvConcatC3DPlugin final : public nvinfer1::IPluginV2DynamicExt
{

public:
  DeconvConcatC3DPlugin(int num_output_maps, int num_groups, int kernel_id, nvinfer1::Weights const& kernel, 
                        const nvinfer1::Dims& kernel_size_nd, 
                        const nvinfer1::Dims& stride_nd, const nvinfer1::Dims& padding_nd,
                        const nvinfer1::Dims& dilation_nd);
  DeconvConcatC3DPlugin(int num_output_maps, int num_groups, int kernel_id, const std::vector<float>& kernel,
                        const std::vector<int>& kernel_size_nd,
                        const std::vector<int>& stride_nd, const std::vector<int>& padding_nd,
                        const std::vector<int>& dilation_nd);
  DeconvConcatC3DPlugin(void const* serialData, size_t serialLength);

  DeconvConcatC3DPlugin() = delete;

  ~DeconvConcatC3DPlugin() override;

  int getNbOutputs() const override;

  // DynamicExt plugins returns DimsExprs class instead of Dims
  using nvinfer1::IPluginV2::getOutputDimensions;
  DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) override;

  int initialize() override;

  void terminate() override;

  using nvinfer1::IPluginV2::getWorkspaceSize;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

  using nvinfer1::IPluginV2::enqueue;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs,
              void* workspace,
              cudaStream_t stream) override;

  size_t getSerializationSize() const override;

  void serialize(void* buffer) const override;

  // DynamicExt plugin supportsFormat update.
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  void destroy() override;

  nvinfer1::IPluginV2DynamicExt* clone() const override;

  void setPluginNamespace(const char* pluginNamespace) override;

  const char* getPluginNamespace() const override;

  DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

  using nvinfer1::IPluginV2Ext::configurePlugin;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;
private:
    int _num_input_maps;
    int _num_output_maps;
    int _num_groups;
    int _kernel_id;
    int _nbInputs;
    std::vector<float> _kernel_h;
    std::vector<float> _bias_h;
    void* _kernel_d;
    void* _bias_d;
    std::vector<int> _kernel_size_nd;
    std::vector<int> _stride_nd;
    std::vector<int> _padding_nd;
    std::vector<int> _dilation_nd;
    cudnnHandle_t _cudnn_handle;
    cudnnTensorDescriptor_t _error_desc, _image_desc;
    cudnnFilterDescriptor_t _kernel_desc;
    cudnnConvolutionDescriptor_t _conv_desc;
    cudnnConvolutionBwdDataAlgo_t _conv_algo;
    std::vector<int> _image_dim, _filter_dim, _error_dim;
    nvinfer1::TensorFormat _input_tensor_format;
    nvinfer1::DataType _trt_data_type;
    cudnnDataType_t _cudnn_data_type;
    cudnnTensorFormat_t _cudnn_tensor_format;
    cudnnMathType_t _cudnn_math_type;
    mutable size_t _cudnn_workspace_sz;
    const char* mPluginNamespace;
    std::string mNamespace;
    bool initialized{false};

    cudaStream_t _concat_stream;
    cudaEvent_t _concat_event;

#ifdef USE_XMMA_DECONV_LIB
    DeconvXmmaHandle_t _deconv_xmma_handle;
    void* _lib_xmma_handle;
#endif
    void initDeconvXmma(int num_output_maps_with_stride);
};

class DeconvConcatC3DPluginCreator : public BaseCreator
{
public:
  DeconvConcatC3DPluginCreator();

  ~DeconvConcatC3DPluginCreator() override = default;

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  const PluginFieldCollection* getFieldNames() override;

  IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

  IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
  static PluginFieldCollection mFC;
  static std::vector<PluginField> mPluginAttributes;
  std::string mNamespace;
};
} //namespace plugin
} //namespace nvinfer1

#endif // TRT_DECONVOLUTION_CONCATENATION_C_3D_PLUGIN_H
