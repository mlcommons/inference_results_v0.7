/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef NV_INFER_SAFE_RUNTIME_H
#define NV_INFER_SAFE_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include "NvInferRuntimeCommon.h"

//!
//! \namespace nvinfer1
//!
//! \brief The TensorRT API version 1 namespace.
//!
namespace nvinfer1
{
//!
//! \namespace nvinfer1::safe
//!
//! \brief The safety subset of TensorRT's API version 1 namespace.
//!
namespace safe
{
class ICudaEngine; //!< Forward declare safe::ICudaEngine for use in other interfaces.
class IExecutionContext; //!< Forward declare safe::IExecutionContext for use in other interfaces.

//!
//! \class IRuntime
//!
//! \brief Allows a serialized functionally safe engine to be deserialized.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRuntime
{
public:
    //!
    //! \brief Deserialize an engine from a stream.
    //!
    //! \param blob The memory that holds the serialized engine.
    //! \param size The size of the memory.
    //!
    //! \return The engine, or nullptr if it could not be deserialized.
    //!
    virtual ICudaEngine* deserializeCudaEngine(const void* blob, std::size_t size) noexcept = 0;

    //!
    //! \brief Destroy this object.
    //!
    virtual void destroy() noexcept = 0;

    //!
    //! \brief Set the GPU allocator.
    //! \param allocator Set the GPU allocator to be used by the runtime. All GPU memory acquired will use this allocator. If NULL is passed, the default allocator will be used.
    //!
    //! Default: uses cudaMalloc/cudaFree.
    //!
    //! If nullptr is passed, the default allocator will be used.
    //!
    virtual void setGpuAllocator(IGpuAllocator* allocator) noexcept = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder
    //!
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;

protected:
    virtual ~IRuntime() TRTNOEXCEPT {}
};

//!
//! \class ICudaEngine
//!
//! \brief A functionally safe engine for executing inference on a built network.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ICudaEngine
{
public:
    //!
    //! \brief Get the number of binding indices.
    //!
    //! \see getBindingIndex();
    //!
    virtual int32_t getNbBindings() const noexcept = 0;

    //!
    //! \brief Retrieve the binding index for a named tensor.
    //!
    //! safe::IExecutionContext::enqueue() requires an array of buffers.
    //!
    //! Engine bindings map from tensor names to indices in this array.
    //! Binding indices are assigned at engine build time, and take values in the range [0 ... n-1] where n is the total
    //! number of inputs and outputs.
    //!
    //! \param name The tensor name.
    //! \return The binding index for the named tensor, or -1 if the name is not found.
    //!
    //! see getNbBindings() getBindingIndex()
    //!
    virtual int32_t getBindingIndex(const char* name) const noexcept = 0;

    //!
    //! \brief Retrieve the name corresponding to a binding index.
    //!
    //! This is the reverse mapping to that provided by getBindingIndex().
    //!
    //! \param bindingIndex The binding index.
    //! \return The name corresponding to the index, or nullptr if the index is out of range.
    //!
    //! \see getBindingIndex()
    //!
    virtual const char* getBindingName(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Determine whether a binding is an input binding.
    //!
    //! \param bindingIndex The binding index.
    //! \return True if the index corresponds to an input binding and the index is in range.
    //!
    //! \see getBindingIndex()
    //!
    virtual bool bindingIsInput(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Get the dimensions of a binding.
    //!
    //! \param bindingIndex The binding index.
    //! \return The dimensions of the binding if the index is in range, otherwise Dims()
    //!
    //! \see getBindingIndex()
    //!
    virtual Dims getBindingDimensions(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Determine the required data type for a buffer from its binding index.
    //!
    //! \param bindingIndex The binding index.
    //! \return The type of the data in the buffer.
    //!
    //! \see getBindingIndex()
    //!
    virtual DataType getBindingDataType(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Get the maximum batch size which can be used for inference.
    //!
    //! \return The maximum batch size for this engine.
    //!
    virtual int32_t getMaxBatchSize() const noexcept = 0;

    //!
    //! \brief Get the number of layers in the network.
    //!
    //! The number of layers in the network is not necessarily the number in the original network definition, as layers
    //! may be combined or eliminated as the engine is optimized. This value can be useful when building per-layer
    //! tables, such as when aggregating profiling data over a number of executions.
    //!
    //! \return The number of layers in the network.
    //!
    virtual int32_t getNbLayers() const noexcept = 0;

    //!
    //! \brief Create an execution context.
    //!
    //! \see safe::IExecutionContext.
    //!
    virtual IExecutionContext* createExecutionContext() noexcept = 0;

    //!
    //! \brief Destroy this object;
    //!
    virtual void destroy() noexcept = 0;

    //!
    //! \brief Get location of binding
    //!
    //! This lets you know whether the binding should be a pointer to device or host memory.
    //!
    //! \see ITensor::setLocation() ITensor::getLocation()
    //!
    //! \param bindingIndex The binding index.
    //! \return The location of the bound tensor with given index.
    //!
    virtual TensorLocation getLocation(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief create an execution context without any device memory allocated
    //!
    //! The memory for execution of this device context must be supplied by the application.
    //!
    //! \see getDeviceMemorySize() safe::IExecutionContext::setDeviceMemory()
    //!
    virtual IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept = 0;

    //!
    //! \brief Return the amount of device memory required by an execution context.
    //!
    //! \see safe::IExecutionContext::setDeviceMemory()
    //!
    virtual size_t getDeviceMemorySize() const noexcept = 0;

    //!
    //! \brief Return the number of bytes per component of an element.
    //!
    //! The vector component size is returned if getBindingVectorizedDim() != -1.
    //!
    //! \param bindingIndex The binding Index.
    //!
    //! \see safe::ICudaEngine::getBindingVectorizedDim()
    //!
    virtual int32_t getBindingBytesPerComponent(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Return the number of components included in one element.
    //!
    //! The number of elements in the vectors is returned if getBindingVectorizedDim() != -1.
    //!
    //! \param bindingIndex The binding Index.
    //!
    //! \see safe::ICudaEngine::getBindingVectorizedDim()
    //!
    virtual int32_t getBindingComponentsPerElement(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Return the binding format.
    //!
    //! \param bindingIndex The binding Index.
    //!
    virtual TensorFormat getBindingFormat(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Return the human readable description of the tensor format.
    //!
    //! The description includes the order, vectorization, data type, strides,
    //! and etc. Examples are shown as follows:
    //!   Example 1: kCHW + FP32
    //!     "Row major linear FP32 format"
    //!   Example 2: kCHW2 + FP16
    //!     "Two wide channel vectorized row major FP16 format"
    //!   Example 3: kHWC8 + FP16 + Line Stride = 32
    //!     "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"
    //!
    //! \param bindingIndex The binding Index.
    //!
    virtual const char* getBindingFormatDesc(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Return the dimension index that the buffer is vectorized.
    //!
    //! Specifically -1 is returned if scalars per vector is 1.
    //!
    //! \param bindingIndex The binding Index.
    //!
    virtual int32_t getBindingVectorizedDim(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Returns the name of the network associated with the engine.
    //!
    //! The name is set during network creation and is retrieved after
    //! building or deserialization.
    //!
    //! \see INetworkDefinition::setName(), INetworkDefinition::getName()
    //!
    //! \return A zero delimited C-style string representing the name of the network.
    //!
    virtual const char* getName() const noexcept = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder
    //!
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;

protected:
    virtual ~ICudaEngine() noexcept {}
};

//!
//! \class IExecutionContext
//!
//! \brief Functionally safe context for executing inference using an engine.
//!
//! Multiple safe execution contexts may exist for one safe::ICudaEngine instance, allowing the same
//! engine to be used for the execution of multiple batches simultaneously.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IExecutionContext
{
public:
    //!
    //! \brief Asynchronously execute inference on a batch.
    //!
    //! This method requires an array of input and output buffers. The mapping from tensor names to indices can be
    //! queried using safe::ICudaEngine::getBindingIndex() \param batchSize The batch size. This is at most the value
    //! supplied when the engine was built. \param bindings An array of pointers to input and output buffers for the
    //! network. \param stream A cuda stream on which the inference kernels will be enqueued \param inputConsumed An
    //! optional event which will be signaled when the input buffers can be refilled with new data
    //!
    //! \return True if the kernels were enqueued successfully.
    //!
    //! \see safe::ICudaEngine::getBindingIndex() safe::ICudaEngine::getMaxBatchSize()
    //!
    virtual bool enqueue(int32_t batchSize, void** bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) noexcept
        = 0;

    //!
    //! \brief Get the associated engine.
    //!
    //! \see safe::ICudaEngine
    //!
    virtual const ICudaEngine& getEngine() const noexcept = 0;

    //!
    //! \brief Destroy this object.
    //!
    virtual void destroy() noexcept = 0;

    //!
    //! \brief Set the name of the execution context.
    //!
    //! This method copies the name string.
    //!
    //! \see getName()
    //!
    virtual void setName(const char* name) noexcept = 0;

    //!
    //! \brief Return the name of the execution context.
    //!
    //! \see setName()
    //!
    virtual const char* getName() const noexcept = 0;

    //!
    //! \brief set the device memory for use by this execution context.
    //!
    //! The memory must be aligned with cuda memory alignment property (using cudaGetDeviceProperties()), and its size must be at least that
    //! returned by getDeviceMemorySize(). If using enqueue() to run the network, The memory is in
    //! use from the invocation of enqueue() until network execution is complete.
    //! Releasing or otherwise using the memory for other purposes during this time will result in undefined behavior.
    //!
    //! \see safe::ICudaEngine::getDeviceMemorySize() safe::ICudaEngine::createExecutionContextWithoutDeviceMemory()
    //!
    virtual void setDeviceMemory(void* memory) noexcept = 0;

    //!
    //! \brief Return the strides of the buffer for the given binding.
    //!
    //! The strides are in units of elements, not components or bytes.
    //! For example, for TensorFormat::kHWC8, a stride of one spans 8 scalars.
    //!
    //! If the bindingIndex is invalid, returns Dims with Dims::nbDims = -1.
    //!
    //! \param bindingIndex The binding index.
    //!
    //! \see getBindingComponentsPerElement
    //!
    virtual Dims getStrides(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Get the dimensions of a binding.
    //!
    //! \param bindingIndex The binding index.
    //! \return The dimensions of the binding if the index is in range, otherwise Dims()
    //!
    //! \see getBindingIndex()
    //!
    virtual Dims getBindingDimensions(int32_t bindingIndex) const noexcept = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder
    //!
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;

    //!
    //! \brief Asynchronously execute inference on a batch.
    //!
    //! This method requires an array of input and output buffers. The mapping from tensor names to indices can be
    //! queried using safe::ICudaEngine::getBindingIndex().
    //! This method only works for an execution context built from a network without an implicit batch dimension.
    //! \param bindings An array of pointers to input and output buffers for the network.
    //! \param stream A cuda stream on which the inference kernels will be enqueued
    //! \param inputConsumed An optional event which will be signaled when the input buffers can be refilled with new
    //! data
    //!
    //! \return True if the kernels were enqueued successfully.
    //!
    //! \see safe::ICudaEngine::getBindingIndex() safe::ICudaEngine::getMaxBatchSize()
    //!
    virtual bool enqueueV2(void** bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) noexcept = 0;

protected:
    virtual ~IExecutionContext() noexcept
    {
    }
};

namespace // unnamed namespace avoids linkage surprises when linking objects built with different versions of this header.
{

//!
//! \brief Create an instance of an safe::IRuntime class.
//!
//! This is the logging class for the runtime.
//!
inline IRuntime* createInferRuntime(ILogger& logger)
{
    return static_cast<IRuntime*>(createSafeInferRuntime_INTERNAL(&logger, NV_TENSORRT_VERSION));
}

} // unnamed namespace
} // namespace safe

} // namespace nvinfer1

#endif // NV_INFER_SAFE_RUNTIME_H
