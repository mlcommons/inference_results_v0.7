// Copyright 2020 NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "trtis_frontend.hpp"

// TRTIS
#include "src/core/constants.h"
#include "src/core/model_config.pb.h"
#include "src/servers/common.h"

// Protobuf
#include <google/protobuf/util/json_util.h>

// Google Logging
#include <glog/logging.h>

// General C++
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <vector>

// LoadGen
#include "loadgen.h"

// DLRM QSL
#include "dlrm_qsl.hpp"

/* Use TRTIS namespace */
namespace ni = nvidia::inferenceserver;

/* Define macro for enabling tracing */
#ifndef TRTIS_FRONTEND_TRACE
#define TRTIS_FRONTEND_TRACE 0
#endif

namespace {
// Callback function for released request
void RequestRelease(TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
    auto request_block = reinterpret_cast<trtis_frontend::RequestPool::Block*>(userp);
    trtis_frontend::RequestPool::Release(request_block);
}
// Callback function for completed response
void ResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
    /* Pass the response (and various items from the userp field) to the member Completion function
     */
    auto response_metadata = reinterpret_cast<trtis_frontend::ResponseMetaData*>(userp);
    trtis_frontend::Server_SUT* sut = response_metadata->m_ServerPtr;
    sut->Completion(response, response_metadata);
#if TRTIS_FRONTEND_TRACE
    if(response_metadata->m_TracePtr != nullptr)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t trace_id = 0;
        TRITONSERVER_InferenceTraceId(response_metadata->m_TracePtr, &trace_id);
        sut->m_TraceManager->CaptureTimestamp(trace_id, TRITONSERVER_TRACE_LEVEL_MIN,
                                              "request responsed", TIMESPEC_TO_NANOS(ts));
    }
#endif // TRTIS_FRONTEND_TRACE
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(response), "deleting resonse");
}

// Callback function for inference requests during warmup phase
void WarmupResponseComplete(TRITONSERVER_InferenceResponse* response, const uint32_t flag,
                            void* userp)
{
    /* Make sure we have a valid response */
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(response), "response");

    /* Increment the number of warmup responses received via member function call */
    auto sut = reinterpret_cast<trtis_frontend::Server_SUT*>(userp);
    sut->IncrementWarmupResponses();
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(response), "deleting resonse");
}

std::string MemoryTypeString(TRITONSERVER_MemoryType memory_type)
{
    return (memory_type == TRITONSERVER_MEMORY_CPU) ? "CPU memory" : "GPU memory";
}

TRITONSERVER_DataType DataTypeToTriton(const inference::DataType dtype)
{
    switch(dtype)
    {
        case inference::DataType::TYPE_BOOL:
            return TRITONSERVER_TYPE_BOOL;
        case inference::DataType::TYPE_UINT8:
            return TRITONSERVER_TYPE_UINT8;
        case inference::DataType::TYPE_UINT16:
            return TRITONSERVER_TYPE_UINT16;
        case inference::DataType::TYPE_UINT32:
            return TRITONSERVER_TYPE_UINT32;
        case inference::DataType::TYPE_UINT64:
            return TRITONSERVER_TYPE_UINT64;
        case inference::DataType::TYPE_INT8:
            return TRITONSERVER_TYPE_INT8;
        case inference::DataType::TYPE_INT16:
            return TRITONSERVER_TYPE_INT16;
        case inference::DataType::TYPE_INT32:
            return TRITONSERVER_TYPE_INT32;
        case inference::DataType::TYPE_INT64:
            return TRITONSERVER_TYPE_INT64;
        case inference::DataType::TYPE_FP16:
            return TRITONSERVER_TYPE_FP16;
        case inference::DataType::TYPE_FP32:
            return TRITONSERVER_TYPE_FP32;
        case inference::DataType::TYPE_FP64:
            return TRITONSERVER_TYPE_FP64;
        case inference::DataType::TYPE_STRING:
            return TRITONSERVER_TYPE_BYTES;
        default:
            break;
    }

    return TRITONSERVER_TYPE_INVALID;
}

size_t GetDataTypeByteSize(const inference::DataType dtype)
{
    switch(dtype)
    {
        case inference::TYPE_BOOL:
            return 1;
        case inference::TYPE_UINT8:
            return 1;
        case inference::TYPE_UINT16:
            return 2;
        case inference::TYPE_UINT32:
            return 4;
        case inference::TYPE_UINT64:
            return 8;
        case inference::TYPE_INT8:
            return 1;
        case inference::TYPE_INT16:
            return 2;
        case inference::TYPE_INT32:
            return 4;
        case inference::TYPE_INT64:
            return 8;
        case inference::TYPE_FP16:
            return 2;
        case inference::TYPE_FP32:
            return 4;
        case inference::TYPE_FP64:
            return 8;
        case inference::TYPE_STRING:
            return 0;
        default:
            break;
    }

    return 0;
}

TRITONSERVER_Error* ResponseAlloc(TRITONSERVER_ResponseAllocator* allocator,
                                  const char* tensor_name, size_t byte_size,
                                  TRITONSERVER_MemoryType preferred_memory_type,
                                  int64_t preferred_memory_type_id, void* userp, void** buffer,
                                  void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
                                  int64_t* actual_memory_type_id)
{
    // If 'byte_size' is zero just return 'buffer'==nullptr, we don't
    // need to do any other book-keeping.
    if(byte_size == 0)
    {
        *buffer = nullptr;
        *buffer_userp = nullptr;
        // LOG_INFO << "allocated " << byte_size << " bytes for result tensor "
        //         << tensor_name;
    }
    else
    {
        auto pool = reinterpret_cast<trtis_frontend::PinnedMemoryPool*>(userp);
        auto block = pool->Obtain();
        // Use CPU instead of CPU_PINNED to trigger internal pinned memory buffer
        // in Triton
        *actual_memory_type = TRITONSERVER_MEMORY_CPU;
        *actual_memory_type_id = 0;
        *buffer = block->m_Data;
        *buffer_userp = new trtis_frontend::PoolBlockPair(pool, block);
    }

    return nullptr; // Success
}

TRITONSERVER_Error* ResponseRelease(TRITONSERVER_ResponseAllocator* allocator, void* buffer,
                                    void* buffer_userp, size_t byte_size,
                                    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
{
    if(buffer_userp != nullptr)
    {
        auto pool_block = reinterpret_cast<trtis_frontend::PoolBlockPair*>(buffer_userp);
        pool_block->first->Release(pool_block->second);
        delete pool_block;
    }
    return nullptr; // Success
}

size_t GetSampleLength(qsl::SampleLibraryPtr_t qsl_ptr, const mlperf::QuerySampleIndex idx)
{
    // Get sample length by checking where the input_mask change from 1 to 0
    size_t start{0};
    size_t end{BERT_MAX_SEQ_LENGTH};
    size_t cursor{(start + end) / 2};
    auto& input_mask =
        *static_cast<std::array<int32_t, BERT_MAX_SEQ_LENGTH>*>(qsl_ptr->GetSampleAddress(idx, 2));
    while(cursor != start)
    {
        if(input_mask[cursor])
        {
            start = cursor;
        }
        else
        {
            end = cursor;
        }
        cursor = (start + end) / 2;
    }
    return end;
}
} // namespace

namespace trtis_frontend {

std::unique_ptr<RequestPool> RequestPool::instance_;

PinnedMemoryPool::PinnedMemoryPool(const size_t element_count, const size_t element_byte_size)
    : m_Head(nullptr), m_Blocks(element_count), m_Buffer(nullptr)
{
    FAIL_IF_CUDA_ERR(
        cudaHostAlloc(&m_Buffer, element_count * element_byte_size, cudaHostAllocPortable),
        "failed to allocate pinned memory");
    char* next_buffer = m_Buffer;
    for(auto& block : m_Blocks)
    {
        if(m_Head != nullptr)
        {
            block.m_NextBlock = m_Head;
        }
        m_Head = &block;
        block.m_Data = next_buffer;
        next_buffer += element_byte_size;
    }
}

PinnedMemoryPool::~PinnedMemoryPool()
{
    if(m_Buffer != nullptr)
    {
        FAIL_IF_CUDA_ERR(cudaFreeHost(m_Buffer), "failed to free pinned memory");
    }
}

void RequestPool::Create(const size_t initial_element_count, TRITONSERVER_Server* server,
                         Server_SUT* server_sut, const std::string& model_name,
                         const uint32_t model_version, const std::vector<InputMetaData>& inputs,
                         const std::vector<std::string>& outputs)
{
    instance_.reset(new RequestPool(initial_element_count, server, server_sut, model_name,
                                    model_version, inputs, outputs));
}

void RequestPool::Destroy()
{
    instance_.reset(nullptr);
}

RequestPool::RequestPool(const size_t initial_element_count, TRITONSERVER_Server* server,
                         Server_SUT* server_sut, const std::string& model_name,
                         const uint32_t model_version, const std::vector<InputMetaData>& inputs,
                         const std::vector<std::string>& outputs)
    : m_Head(nullptr), m_Blocks(initial_element_count), m_Server(server), m_ServerSUT(server_sut),
      m_ModelName(model_name), m_ModelVersion(model_version), m_Inputs(inputs), m_Outputs(outputs)
{
    Block* prev_block = nullptr;
    // Build free block list in the same order as in std list
    for(auto& block : m_Blocks)
    {
        if(m_Head == nullptr)
        {
            m_Head = &block;
        }
        if(prev_block != nullptr)
        {
            prev_block->m_NextBlock = &block;
        }
        prev_block = &block;
        InternalInitInferenceRequest(&block);
    }
}

RequestPool::~RequestPool()
{
    for(auto& block : m_Blocks)
    {
        if(block.m_Data != nullptr)
        {
            FAIL_IF_ERR(TRITONSERVER_InferenceRequestDelete(block.m_Data),
                        "deleting inference request");
        }
    }
}

void RequestPool::InitInferenceRequest(RequestPool::Block* block)
{
    instance_->InternalInitInferenceRequest(block);
}

void RequestPool::InternalInitInferenceRequest(RequestPool::Block* block)
{
    // Init m_Data (TRITONSERVER_InferenceRequest)
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestNew(&block->m_Data, m_Server, m_ModelName.c_str(),
                                                 m_ModelVersion),
                "creating new inference request");
    for(const auto& input : m_Inputs)
    {
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                        block->m_Data, std::get<0>(input).c_str(), std::get<1>(input),
                        std::get<2>(input).data(), std::get<2>(input).size()),
                    "setting input meta-data for the request");
    }
    for(const auto& output : m_Outputs)
    {
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(block->m_Data, output.c_str()),
                    "requesting output for the request");
    }
    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestSetReleaseCallback(block->m_Data, RequestRelease, block),
        "setting request release callback");
    block->m_ResponseMetadata = ResponseMetaData(m_ServerSUT);
}

void Server_SUT::Init(size_t min_sample_size, size_t max_sample_size)
{
#if TRTIS_FRONTEND_TRACE
    /* Set up trace manager */
    ni::TraceManager* manager = nullptr;
    FAIL_IF_ERR(nvidia::inferenceserver::TraceManager::Create(
                    &manager, TRITONSERVER_TRACE_LEVEL_MAX, 80 /* rate, one sample per batch*/,
                    "trtis_trace.log"),
                "creating trace manger");
    m_TraceManager.reset(manager);
#endif // TRTIS_FRONTEND_TRACE

    /* Create the options for the server */
    TRITONSERVER_ServerOptions* server_options = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsNew(&server_options), "creating server options");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options,
                                                                 m_ModelRepositoryPath.c_str()),
                "setting model repository path");

    /* Actually create the server now */
    TRITONSERVER_Server* server_ptr = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerNew(&server_ptr, server_options), "creating server");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsDelete(server_options), "deleting server options");
    m_Server = std::shared_ptr<TRITONSERVER_Server>(server_ptr, TRITONSERVER_ServerDelete);

    /* Wait until the server is both live and ready, and the model is ready. */
    size_t health_iters = 0;
    while(true)
    {
        bool live, ready, model_ready;
        FAIL_IF_ERR(TRITONSERVER_ServerIsLive(m_Server.get(), &live),
                    "unable to get server liveness");
        FAIL_IF_ERR(TRITONSERVER_ServerIsReady(m_Server.get(), &ready),
                    "unable to get server readiness");
        FAIL_IF_ERR(TRITONSERVER_ServerModelIsReady(m_Server.get(), m_ModelName.c_str(),
                                                    m_ModelVersion, &model_ready),
                    "unable to get model readiness");
        std::cout << "Server Health status: live " << live << ", ready " << ready
                  << ", model ready " << model_ready << std::endl;
        if(live && ready && model_ready)
        {
            std::cout << "Server is live and ready. Model is ready" << std::endl;
            break;
        }

        if(++health_iters >= 200)
        {
            FAIL("failed to find healthy inference server within 200 tries");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Initalize enough pinned output buffer in front
    // We want to have instance # sets of output buffers, each set has
    // output # buffers, and each buffer has max batch bytes for the output
    TRITONSERVER_Message* model_config_message = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerModelConfig(m_Server.get(), m_ModelName.c_str(), m_ModelVersion,
                                               1 /* config_version */, &model_config_message),
                "unable to get model config message");
    auto lcm = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        model_config_message, TRITONSERVER_MessageDelete);

    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_config_message, &buffer, &byte_size),
                "unable to serialize model metadata message to JSON");

    inference::ModelConfig config;
    ::google::protobuf::util::JsonStringToMessage({buffer, (int)byte_size}, &config);
    std::cout << "Model Config:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;
    size_t max_batch1_byte_size = 0;
    std::vector<std::string> output_names;
    // Pre-allocate a memory pool for output buffers
    // Use the largest possible size among the outputs as size for each block
    for(const auto& output : config.output())
    {
        int batch1_byte_size = 1;
        for(const auto& dim : output.dims())
        {
            // FIXME: hard-coded value for variable dims
            if(dim == -1)
            {
                batch1_byte_size *= BERT_MAX_SEQ_LENGTH;
            }
            else
            {
                batch1_byte_size *= dim;
            }
        }
        batch1_byte_size *= GetDataTypeByteSize(output.data_type());
        if(batch1_byte_size <= 0)
        {
            FAIL("can't preallocate memory for variable size data type");
        }
        max_batch1_byte_size = std::max(max_batch1_byte_size, (size_t)batch1_byte_size);
        output_names.emplace_back(output.name());
    }

    size_t instance_count = 0;
    for(const auto& instance_group : config.instance_group())
    {
        // Don't really care about which gpu it is since we are allocating
        // pinned memory buffer with cudaHostAllocPortable.
        for(const auto& gpu : instance_group.gpus())
        {
            instance_count += instance_group.count();
        }
    }
    size_t max_batch_size = config.max_batch_size() == 0 ? 1 : config.max_batch_size();
    size_t pool_item_count =
        2 * instance_count * (max_batch_size / min_sample_size + 1) * output_names.size();
    size_t pool_item_size = max_batch1_byte_size * max_sample_size;
    m_OutputBufferPool.reset(new PinnedMemoryPool(pool_item_count, pool_item_size));

    // Pre-allocate a growable request pool for inference requests
    std::vector<InputMetaData> inputs;
    m_IsDynamic = false;
    for(const auto& io : config.input())
    {
        InputMetaData input;
        std::get<0>(input) = io.name();
        std::get<1>(input) = DataTypeToTriton(io.data_type());
        auto& shape = std::get<2>(input);
        if(config.max_batch_size() != 0)
        {
            shape.push_back(1);
        }
        for(const auto& dim : io.dims())
        {
            m_IsDynamic |= (dim == -1);
            shape.push_back(dim);
        }
        m_InputTensors.emplace_back(input);
        inputs.emplace_back(std::move(input));
    }

    /*  Create the allocator that will be used to allocate buffers for
        the result tensors. */
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(&m_Allocator, ResponseAlloc, ResponseRelease,
                                                  nullptr /* start_fn */),
                "creating response allocator");

    RequestPool::Create(50000 /* initial_element_count */, m_Server.get(), this, m_ModelName,
                        m_ModelVersion, inputs, output_names);

    // Prepare padding buffer in the case of DLRM. The model assumes
    // even batch size but some sample has odd batch size
    if(m_UseDlrmQsl)
    {
        for(const auto& input_tensor : m_InputTensors)
        {
            int64_t batch1_byte_size = TRITONSERVER_DataTypeByteSize(std::get<1>(input_tensor));
            for(const auto& dim : std::get<2>(input_tensor))
            {
                batch1_byte_size *= dim;
            }
            m_InputPaddings.emplace_back(batch1_byte_size);
        }

        for(const auto& io : config.output())
        {
            int64_t batch1_byte_size =
                TRITONSERVER_DataTypeByteSize(DataTypeToTriton(io.data_type()));
            for(const auto& dim : io.dims())
            {
                batch1_byte_size *= dim;
            }
            m_OutputPaddingSize = (size_t)batch1_byte_size;
        }
    }

    /* Set the number of warmup responses to 0 to prepare for next warmup */
    m_NumWarmupResponses = 0;
}

void Server_SUT::ModelStats()
{
    TRITONSERVER_Message* model_stats_message = nullptr;
    auto err = TRITONSERVER_ServerModelStatistics(m_Server.get(), m_ModelName.c_str(),
                                                  m_ModelVersion, &model_stats_message);
    if(err != nullptr)
    {
        std::cerr << "failed to obtain stats message: " << TRITONSERVER_ErrorMessage(err)
                  << std::endl;
        TRITONSERVER_ErrorDelete(err);
        return;
    };
    auto lms = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        model_stats_message, TRITONSERVER_MessageDelete);
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_stats_message, &buffer, &byte_size),
                "serializing stats message");

    std::cout << "Model Stats:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;
}

void Server_SUT::ModelMetadata()
{
    TRITONSERVER_Message* model_metadata_message = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerModelMetadata(m_Server.get(), m_ModelName.c_str(),
                                                 m_ModelVersion, &model_metadata_message),
                "obtaining metadata message");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_metadata_message, &buffer, &byte_size),
                "serializing model metadata message");

    std::cout << "Model Metadata:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;
    auto lmm = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        model_metadata_message, TRITONSERVER_MessageDelete);
}

void Server_SUT::Warmup(double duration_sec, double expected_qps)
{
    /* Notify user that we are starting the warmup */
    std::cout << "Starting TRTIS warmup" << std::endl;

    /* Calculate the number of inferences to send */
    int num_inferences = (int)(duration_sec * expected_qps);

    /* Keep track of the number of inferences that we have sent so far */
    int inferences_sent = 0;

    // Load a sample to RAM to use
    mlperf::QuerySampleIndex index{0}; // Arbitrary sample index
    std::vector<mlperf::QuerySampleIndex> samples;
    samples.push_back(index);
    m_SampleLibrary->LoadSamplesToRam(samples);

    while(inferences_sent < num_inferences)
    {
        /* Create the inference request provider, which provides the request
            header information as well as the actual data. */
        auto request_block = RequestPool::Obtain();
        if(request_block->m_Data == nullptr)
        {
            RequestPool::InitInferenceRequest(request_block);
            request_block->m_ResponseMetadata.m_ServerPtr = this;
        }
        if(m_UseDlrmQsl)
        {
            // Inputs will need to be re-added as the shape is different from run
            // to run
            TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

            auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibrary.get());
            // new batch size for the request
            auto num_pairs = qsl->GetNumUserItemPairs(index);
            for(size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data =
                    (int8_t*)qsl->GetSampleAddress(index, idx); // Get address of the query
                size_t input_size = qsl->GetSampleSize(idx) * num_pairs;

                auto& shape = std::get<2>(m_InputTensors[idx]);
                shape[0] = num_pairs;
                if(num_pairs % 2)
                {
                    shape[0] += 1;
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                                "re-adding input");
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, input_size, m_InputMemoryType, 0),
                                "appending input data");
                    // Add padding buffer
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    m_InputPaddings[idx].data(), m_InputPaddings[idx].size(),
                                    TRITONSERVER_MEMORY_CPU, 0),
                                "appending input data padding");
                }
                else
                {
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                                "re-adding input");
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, input_size, m_InputMemoryType, 0),
                                "appending input data");
                }
            }
        }
        else if(m_IsDynamic)
        {
            // Special handling as BERT is the only model uses dynamic shape
            //
            // Inputs will need to be re-added as the shape is different from run
            // to run
            TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

            size_t seq_len = GetSampleLength(m_SampleLibrary, index);
            for(size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*)m_SampleLibrary->GetSampleAddress(
                    index, idx); // Get address of the query
                // Need to calculate the shape from data for dynamic case
                size_t input_size =
                    seq_len * TRITONSERVER_DataTypeByteSize(std::get<1>(m_InputTensors[idx]));

                thread_local std::vector<int64_t> shape{1, 0};
                shape[1] = seq_len;
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                            "re-adding input");
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                input_data, input_size, m_InputMemoryType, 0),
                            "appending input data");
            }
        }
        else
        {
            for(size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*)m_SampleLibrary->GetSampleAddress(
                    index, idx); // Get address of the query
                size_t input_size = m_SampleLibrary->GetSampleSize(idx);

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
                            "removing input data");
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                input_data, input_size, m_InputMemoryType, 0),
                            "appending input data");
            }
        }

        /* Set response callback for warmup */
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
                        request_block->m_Data, m_Allocator, m_OutputBufferPool.get(),
                        WarmupResponseComplete, this),
                    "appending input data");

        /* Actually perform inferences (asynchronously) */
        FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, nullptr),
                    "running inference");
        inferences_sent += 1;
    }

    /* Wait for all the warmup inferences to complete */
    while(m_NumWarmupResponses < inferences_sent)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    /* Unload sample from RAM */
    m_SampleLibrary->UnloadSamplesFromRam(samples);

    /* Reset the number of warmup responses */
    m_NumWarmupResponses = 0;

    /* Notify user that we are done with the warmup */
    std::cout << "Finished TRTIS warmup" << std::endl;
}

void Server_SUT::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    for(size_t i = 0; i < samples.size(); i++)
    {
        TRITONSERVER_InferenceTrace* trace = nullptr;
#if TRTIS_FRONTEND_TRACE
        if(m_TraceManager != nullptr)
        {
            trace = m_TraceManager->SampleTrace();
            if(trace != nullptr)
            {
                uint64_t trace_id = 0;
                TRITONSERVER_InferenceTraceId(trace, &trace_id);
                struct timespec ts;
                clock_gettime(CLOCK_MONOTONIC, &ts);
                m_TraceManager->CaptureTimestamp(trace_id, TRITONSERVER_TRACE_LEVEL_MIN,
                                                 "request issued", TIMESPEC_TO_NANOS(ts));
            }
        }
#endif // TRTIS_FRONTEND_TRACE
       // Set the Request Provider
        auto request_block = RequestPool::Obtain();
        if(request_block->m_Data == nullptr)
        {
            RequestPool::InitInferenceRequest(request_block);
            request_block->m_ResponseMetadata.m_ServerPtr = this;
        }
        request_block->m_ResponseMetadata.m_ResponseId = samples[i].id;
        request_block->m_ResponseMetadata.m_QuerySampleIdx = samples[i].index;
        request_block->m_ResponseMetadata.m_TracePtr = trace;
        if(m_UseDlrmQsl)
        {
            // Inputs will need to be re-added as the shape is different from run
            // to run
            TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

            auto qsl = reinterpret_cast<DLRMSampleLibrary*>(m_SampleLibrary.get());
            // new batch size for the request
            auto num_pairs = qsl->GetNumUserItemPairs(samples[i].index);
            for(size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*)qsl->GetSampleAddress(
                    samples[i].index, idx); // Get address of the query
                size_t input_size = qsl->GetSampleSize(idx) * num_pairs;

                auto& shape = std::get<2>(m_InputTensors[idx]);
                shape[0] = num_pairs;
                if(num_pairs % 2)
                {
                    shape[0] += 1;
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                                "re-adding input");
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, input_size, m_InputMemoryType, 0),
                                "appending input data");
                    // Add padding buffer
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    m_InputPaddings[idx].data(), m_InputPaddings[idx].size(),
                                    TRITONSERVER_MEMORY_CPU, 0),
                                "appending input data padding");
                }
                else
                {
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                                "re-adding input");
                    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                    request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                    input_data, input_size, m_InputMemoryType, 0),
                                "appending input data");
                }
            }
            request_block->m_ResponseMetadata.m_PaddingSize =
                (num_pairs % 2) ? m_OutputPaddingSize : 0;
        }
        else if(m_IsDynamic)
        {
            // Special handling as BERT is the only model uses dynamic shape
            //
            // Inputs will need to be re-added as the shape is different from run
            // to run
            TRITONSERVER_InferenceRequestRemoveAllInputs(request_block->m_Data);

            size_t seq_len = GetSampleLength(m_SampleLibrary, samples[i].index);
            for(size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*)m_SampleLibrary->GetSampleAddress(
                    samples[i].index, idx); // Get address of the query
                // Need to calculate the shape from data for dynamic case
                size_t input_size =
                    seq_len * TRITONSERVER_DataTypeByteSize(std::get<1>(m_InputTensors[idx]));

                thread_local std::vector<int64_t> shape{1, 0};
                shape[1] = seq_len;
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                std::get<1>(m_InputTensors[idx]), shape.data(), shape.size()),
                            "re-adding input");
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                input_data, input_size, m_InputMemoryType, 0),
                            "appending input data");
            }
        }
        else
        {
            for(size_t idx = 0; idx < m_InputTensors.size(); idx++)
            {
                // Get a pointer to the input data
                int8_t* input_data = (int8_t*)m_SampleLibrary->GetSampleAddress(
                    samples[i].index, idx); // Get address of the query
                size_t input_size = m_SampleLibrary->GetSampleSize(idx);

                FAIL_IF_ERR(TRITONSERVER_InferenceRequestRemoveAllInputData(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str()),
                            "removing input data");
                FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
                                request_block->m_Data, std::get<0>(m_InputTensors[idx]).c_str(),
                                input_data, input_size, m_InputMemoryType, 0),
                            "appending input data");
            }
        }

        /* Actually perform inference (asynchronously) */
        // For the userp field, pass in a pointer to a tuple with a pointer to the SUT, a pointer to
        // the request provider, and the LoadGen response ID
        auto buffer_pool = m_OutputBufferPool.get();
        /* Set response callback for this request */
        FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(
                        request_block->m_Data, m_Allocator, m_OutputBufferPool.get(),
                        ResponseComplete, &request_block->m_ResponseMetadata),
                    "appending input data");

        FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(m_Server.get(), request_block->m_Data, trace),
                    "running inference");
    }
}

void Server_SUT::Completion(TRITONSERVER_InferenceResponse* response,
                            const ResponseMetaData* response_metadata)
{
    /* Make sure we have a valid response */
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseError(response), "response");

    /* Extract and process the response data */
    const char* name;
    TRITONSERVER_DataType datatype;
    void* userp;
    const int64_t* shape;
    uint64_t dim_count;
    const void* output0_content;
    size_t output0_byte_size;
    TRITONSERVER_MemoryType output0_memory_type;
    int64_t output0_memory_type_id;
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutput(
                    response, 0 /* index */, &name, &datatype, &shape, &dim_count, &output0_content,
                    &output0_byte_size, &output0_memory_type, &output0_memory_type_id, &userp),
                "getting output0 result");
    // Recast the output pointer as a uintptr_t (for LoadGen)
    uintptr_t output0_result = reinterpret_cast<uintptr_t>(output0_content);

    /* Call QuerySamplesComplete */
    mlperf::QuerySampleResponse loadgen_response{response_metadata->m_ResponseId, output0_result,
                                                 output0_byte_size -
                                                     response_metadata->m_PaddingSize};

    // callback if it exists
    if(m_ResponseCallback)
    {
        std::vector<::mlperf::QuerySampleIndex> response_indices = {
            response_metadata->m_QuerySampleIdx};
        m_ResponseCallback(&loadgen_response, response_indices, 1);
    }

    mlperf::QuerySamplesComplete(&loadgen_response,
                                 1); // We always send one inference response at a time
}

void Server_SUT::Done()
{
    RequestPool::Destroy();

    /* Delete the response allocator since we are done with it */
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(m_Allocator), "deleting response allocator");

    /* Reset the server pointer to nullptr to ensure Init() is called before the server is used
     * again */
    m_Server = nullptr;
}

void Server_SUT::FlushQueries() {}

void Server_SUT::ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns)
{
}

}; // namespace trtis_frontend
