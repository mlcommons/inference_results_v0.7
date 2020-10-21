//
// Created by mmoroz on 7/9/20.
//

#ifndef IVA_RESNET50_IVATPUSYSTEMUNDERTEST_H
#define IVA_RESNET50_IVATPUSYSTEMUNDERTEST_H


#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <loadgen.h>
#include <system_under_test.h>
#include <tpu.h>
#include <atomic>

#include "IvaTPUQuerySampleLibrary.h"

namespace ivatpu {

    struct ClassificationQuery {
        explicit ClassificationQuery(const std::vector<mlperf::QuerySample> &queries);
        const std::vector<mlperf::QuerySample> &queries_;
        std::vector<mlperf::QuerySampleResponse> querySampleResponse;
        std::vector<int> data;
        size_t inferences_completed;
        void reportSamples();
    };

    class IvaTPUSystemUnderTest : public mlperf::SystemUnderTest {
    public:
        explicit IvaTPUSystemUnderTest(std::shared_ptr<TPUDevice> device, std::shared_ptr<TPUProgram> program, IvaTPUQuerySampleLibrary& querySampleLibrary);
        virtual ~IvaTPUSystemUnderTest();
        const std::string& Name() const override;
        void IssueQuery(const std::vector<mlperf::QuerySample>& queries) override;
        void FlushQueries() override;
        void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>&) override;
        void startInferenceWorker();
        void stopInferenceWorker();

    private:
        void submitBatch(const std::vector<mlperf::QuerySample> &queries, size_t offset, size_t count);
        void inferenceWorkerThread();
        size_t completeInferences(const std::shared_ptr<TPUTensor> &out_tensor,
                                  ivatpu::ClassificationQuery &classificationQuery,
                                  size_t queriesLeft, size_t queryOffset,
                                  uint32_t devicePendingBatches);
    private:
        std::shared_ptr<TPUDevice> device_;
        std::shared_ptr<TPUProgram> program_;
        std::vector<tpu_io_descriptor *> ioDescriptors_;
        IvaTPUQuerySampleLibrary& querySampleLibrary_;
        const std::string name_ = "IVA FPGA1";
        std::mutex queueMutex_;
        std::condition_variable inferenceQueueEmtpyCV_;
        std::queue<ClassificationQuery> classificationQueue;
        std::thread inferenceWorker_;
        std::mutex deviceQueueMutex_;
        std::condition_variable deviceQueueEmtpyCV_;
        std::condition_variable deviceQueueFullCV_;
        size_t inferenceSubmitedCount_;
        size_t inferenceCompletedCount_;
        bool shutdown;
        size_t batch_size;
    private:
        std::tuple<size_t, size_t, size_t, size_t>
                processInferences(std::shared_ptr<TPUTensor> out_tensor, ivatpu::ClassificationQuery& classificationQuery, size_t queriesLeft, size_t queryOffset, size_t devicePendingBatches);
    };
}


#endif //IVA_RESNET50_IVATPUSYSTEMUNDERTEST_H
