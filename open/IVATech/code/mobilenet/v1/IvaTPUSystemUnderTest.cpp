//
// Created by mmoroz on 7/9/20.
//

#include <iostream>
#include <utility>
#include <stdexcept>
#include <loadgen.h>
#include <cstring>
#include <thread>
#include <chrono>
#include <tuple>

#include <spdlog/spdlog.h>

#include "IvaTPUSystemUnderTest.h"
#include "util.h"

const std::string &ivatpu::IvaTPUSystemUnderTest::Name() const {
    return name_;
}

void ivatpu::IvaTPUSystemUnderTest::submitBatch(const std::vector<mlperf::QuerySample> &queries, size_t offset, size_t count) {
    const size_t inputIndex = 0;
    // copy sample to device buffer
    tpu_io_descriptor *ioDescriptor = this->ioDescriptors_[this->inferenceSubmitedCount_ % this->ioDescriptors_.size()];

    uint8_t *deviceBuffer = (uint8_t *)tpu_program_get_input_buffer(ioDescriptor, inputIndex);
    for(size_t i = 0; i < count; ++i) {
        ivatpu::SampleBuffer *sample = this->querySampleLibrary_.GetSample(queries[offset + i].index);
        memcpy(deviceBuffer + i * sample->size, sample->data, sample->size);
    }

    // submit inference
    int rc = tpu_inference_submit(this->device_.get(), ioDescriptor);
    while (rc != 0) {
        rc = tpu_inference_submit(this->device_.get(), ioDescriptor);
    }
}

void ivatpu::IvaTPUSystemUnderTest::IssueQuery(const std::vector<mlperf::QuerySample> &queries) {
    spdlog::debug("Issue {} queries", queries.size());
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        classificationQueue.emplace(queries);
        lock.unlock();
        inferenceQueueEmtpyCV_.notify_one();
        spdlog::debug("Notify worker");
    }

    for(size_t i = 0; i < queries.size(); i += batch_size) {
        size_t count = (i + batch_size > queries.size()) ? queries.size() - i : batch_size;
        submitBatch(queries, i, count);

        {
            std::unique_lock<std::mutex> deviceLk(deviceQueueMutex_);
            ++inferenceSubmitedCount_;
            deviceLk.unlock();
            // notify consumer. Should I optimize?
            deviceQueueEmtpyCV_.notify_one();
        }
        spdlog::debug("TPU QUEUE SIZE: {}. SUBMITED {}", inferenceSubmitedCount_ - inferenceCompletedCount_, inferenceSubmitedCount_);

        // wait on device queue full
        if(inferenceSubmitedCount_ - inferenceCompletedCount_ >= TPU_PROGRAM_QUEUE_LENGTH) {
            std::unique_lock<std::mutex> lk(deviceQueueMutex_);
            deviceQueueFullCV_.wait(lk, [this] {
                return (inferenceSubmitedCount_ - inferenceCompletedCount_) < TPU_PROGRAM_QUEUE_LENGTH;
            });
        }
    }
}

void ivatpu::IvaTPUSystemUnderTest::FlushQueries() {
    spdlog::info("Flush queries");
}

void ivatpu::IvaTPUSystemUnderTest::ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency> &samples) {
    spdlog::info("Return latency results for {} samples", samples.size());
}

template<class T>
int get_max_index(T *data) {
    size_t idx = 0;
    T v = 0;
    for (size_t i = 0; i < 1001; ++i) {
        if (v < data[i]) {
            idx = i;
            v = data[i];
        }
    }
    return idx;
}

int get_resnet50_class(const TPUTensor *tensor, size_t N)
{
    size_t tensorOffset = N * tpu_tensor_get_dimension_size(&tensor->shape[1], tensor->shape_len - 1, 1);
    switch (tensor->dtype) {
        case TPU_FLOAT32:
            return get_max_index(reinterpret_cast<float *>(tensor->data) + tensorOffset);
        case TPU_INT8:
            return get_max_index(reinterpret_cast<char *>(tensor->data) + tensorOffset);
        case TPU_FLOAT64:
            return get_max_index(reinterpret_cast<double *>(tensor->data) + tensorOffset);
        default:
            throw std::runtime_error(std::string("can't find max for ") + tpu_tensor_get_dtype_name(tensor->dtype));
    }
}


std::tuple<size_t, size_t, size_t, size_t>
ivatpu::IvaTPUSystemUnderTest::processInferences(std::shared_ptr<TPUTensor> out_tensor,
                                                 ivatpu::ClassificationQuery &classificationQuery, size_t queriesLeft,
                                                 size_t queryOffset, size_t devicePendingBatches) {
    size_t completedBatches = completeInferences(out_tensor, classificationQuery, queriesLeft, queryOffset,
                                                 devicePendingBatches);
    queryOffset += completedBatches * batch_size;
    if (queriesLeft > completedBatches * batch_size) {
        queriesLeft -= completedBatches * batch_size;
    } else {
        queriesLeft = 0;
    }
    devicePendingBatches -= completedBatches;
    return std::make_tuple(completedBatches, queryOffset, queriesLeft, devicePendingBatches);
}

void ivatpu::IvaTPUSystemUnderTest::inferenceWorkerThread()
{
    auto node = tpu_program_get_output_node(program_.get(), 0);
    std::shared_ptr<TPUTensor> out_tensor(tpu_tensor_allocate(node->user_shape, node->user_shape_len, TPU_INT8, malloc),
                                          tpu_tensor_free);

    spdlog::info("Start worker");

    size_t devicePendingBatches = 0;
    while(!shutdown) {
        // wait for inference queue
        spdlog::debug("Wait inference query");
        std::unique_lock<std::mutex> lk(queueMutex_);
        inferenceQueueEmtpyCV_.wait(lk, [&] { return shutdown || !classificationQueue.empty(); });
        if(shutdown) break;
        auto classificationQuery = classificationQueue.front(); // single consumer
        size_t queriesLeft = classificationQuery.queries_.size();
        lk.unlock();
        spdlog::debug("Inference query detected");

        size_t queryOffset = 0;

        if(devicePendingBatches) {
            size_t completedBatches;

            std::tie(completedBatches, queryOffset,
                     queriesLeft, devicePendingBatches) = processInferences(out_tensor,
                                                                            classificationQuery, queriesLeft, queryOffset,
                                                                            devicePendingBatches);

            if(devicePendingBatches > 0) { // still has pending inferences
                mlperf::QuerySamplesComplete(classificationQuery.querySampleResponse.data(), classificationQuery.queries_.size());
                {
                    std::unique_lock<std::mutex> lk(queueMutex_);
                    classificationQueue.pop();
                }
                continue;
            };
        }

        while (queriesLeft > 0) {
            // wait for inference queue
            if(inferenceSubmitedCount_ - inferenceCompletedCount_ == 0){
                spdlog::debug("Wait device queue");
                std::unique_lock<std::mutex> deviceLk(deviceQueueMutex_);
                deviceQueueEmtpyCV_.wait(deviceLk, [&] { return inferenceSubmitedCount_ > inferenceCompletedCount_; });
            }

            // read inference results
            uint32_t inferenceNr = 0;
            spdlog::debug("Wait for inference {}. {} submitted, {} completed", inferenceNr, inferenceSubmitedCount_, inferenceCompletedCount_);
            int rc = tpu_inference_wait(device_.get(), &inferenceNr);
            if(rc != 0) {
                throw std::runtime_error("Inference error");
            }
            spdlog::debug("Inferences up to {} completed. Currently handled {}", inferenceNr, inferenceCompletedCount_);
            inferenceNr -= inferenceCompletedCount_;

            size_t completedBatches;

            std::tie(completedBatches, queryOffset,
                     queriesLeft, devicePendingBatches) = processInferences(out_tensor,
                                                                            classificationQuery, queriesLeft, queryOffset,
                                                                            inferenceNr);
            spdlog::debug("completed batches {}, query offset {}, queries left {}, device pending batches {}",
                          completedBatches, queryOffset,
                          queriesLeft, devicePendingBatches);
        }
        for(size_t i = 0; i < classificationQuery.queries_.size(); ++i) {
            volatile int v = classificationQuery.querySampleResponse[i].id;
            volatile uintptr_t p = classificationQuery.querySampleResponse[i].data;
            if(p != reinterpret_cast<uintptr_t>(&classificationQuery.data[i])) {
                spdlog::error("unxpected data at offset {}: {}", i, classificationQuery.data[i]);
                throw std::runtime_error("unexpected data");
            }
        }
        spdlog::debug("complete classification query");
        {
            std::unique_lock<std::mutex> lk(queueMutex_);
            mlperf::QuerySamplesComplete(classificationQuery.querySampleResponse.data(), classificationQuery.queries_.size());
            classificationQueue.pop();
        }
        spdlog::debug("completed classification query");
    }
}

size_t ivatpu::IvaTPUSystemUnderTest::completeInferences(const std::shared_ptr<TPUTensor> &out_tensor,
                                                         ivatpu::ClassificationQuery &classificationQuery,
                                                         size_t queriesLeft, size_t queryOffset,
                                                         uint32_t devicePendingBatches) {
    size_t batchesLeft = queriesLeft / batch_size;
    if(queriesLeft % batch_size) batchesLeft += 1;

    size_t completed = std::min<size_t>(batchesLeft, devicePendingBatches);
    int rc;
    spdlog::debug("Handle {} batches", completed);
    for(size_t i = 0; i < completed; ++i) {
        tpu_io_descriptor *sampleDescriptor = this->ioDescriptors_[this->inferenceCompletedCount_ % this->ioDescriptors_.size()];

        rc = tpu_program_get_output_tensor(this->program_.get(), sampleDescriptor, out_tensor.get(), 0);
        if(rc != 0) {
            throw std::runtime_error("Failed to get output tensor");
        }

        for(size_t j = 0; j < batch_size; ++j, ++queryOffset) {
            const mlperf::QuerySample &sample = classificationQuery.queries_[queryOffset];
            assert(queryOffset < classificationQuery.querySampleResponse.size());

            int sample_class = get_resnet50_class(out_tensor.get(), j);
            classificationQuery.data[queryOffset] = sample_class;
            classificationQuery.querySampleResponse[queryOffset].data = reinterpret_cast<uintptr_t>(&classificationQuery.data[queryOffset]);
            spdlog::debug("response {} data {}", queryOffset, classificationQuery.querySampleResponse[queryOffset].data);
            queriesLeft -= 1;
            if(!queriesLeft) break;
        }


        {
            std::unique_lock<std::mutex> deviceLk(deviceQueueMutex_);
            this->inferenceCompletedCount_++;
        }

        // unlock producer. Should I optimize?
        this->deviceQueueFullCV_.notify_one();
    }
    return completed;
}

void ivatpu::IvaTPUSystemUnderTest::startInferenceWorker()
{
    inferenceWorker_ = std::thread(&ivatpu::IvaTPUSystemUnderTest::inferenceWorkerThread, this);
}

ivatpu::IvaTPUSystemUnderTest::IvaTPUSystemUnderTest(std::shared_ptr<TPUDevice> device,
                                                     std::shared_ptr<TPUProgram> program,
        IvaTPUQuerySampleLibrary& querySampleLibrary)
        : device_(device), program_(program), querySampleLibrary_(querySampleLibrary),
          shutdown(false), inferenceSubmitedCount_(0), inferenceCompletedCount_(0)
{
    // allocate TPU buffers
    ioDescriptors_.resize(ivatpu::TPU_PROGRAM_QUEUE_LENGTH + 1);
    for(auto &i: ioDescriptors_) {
        i = tpu_io_descriptor_create(program.get());
        if(i == nullptr) {
            throw std::runtime_error("failed to allocate IO descriptor");
        }
    }
    auto input_node = tpu_program_get_input_node(program.get(), 0);
    batch_size = input_node->tpu_shape[0];
    startInferenceWorker();
}

void ivatpu::IvaTPUSystemUnderTest::stopInferenceWorker()
{
    std::cout << "stop inference worker" << std::endl;
    {
        std::unique_lock<std::mutex> lk(queueMutex_);
        shutdown = true;
    }
    inferenceQueueEmtpyCV_.notify_one();
    inferenceWorker_.join();
}

ivatpu::IvaTPUSystemUnderTest::~IvaTPUSystemUnderTest()
{
    stopInferenceWorker();
    for(auto &i: ioDescriptors_) {
        tpu_io_descriptor_free(i);
    }
}

ivatpu::ClassificationQuery::ClassificationQuery(const std::vector<mlperf::QuerySample> &queries): queries_(queries), inferences_completed(0) {
    querySampleResponse.reserve(queries.size());
    data.resize(queries.size());
    for(const auto &q: queries) {
        querySampleResponse.push_back(mlperf::QuerySampleResponse{q.id, 0, sizeof(int)});
    }
    if(querySampleResponse.size() != queries.size()) {
        throw std::runtime_error("Unexpected response size");
    }
}
