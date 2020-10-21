//
// Created by mmoroz on 7/9/20.
//

#ifndef IVA_RESNET50_IVATPUQUERYSAMPLELIBRARY_H
#define IVA_RESNET50_IVATPUQUERYSAMPLELIBRARY_H

#include <unordered_map>
#include <tpu.h>
#include <query_sample_library.h>

namespace ivatpu {
    struct SampleBuffer {
        void *data;
        size_t size;
    };

    class IvaTPUQuerySampleLibrary: public mlperf::QuerySampleLibrary {
    public:
        IvaTPUQuerySampleLibrary(std::shared_ptr<TPUProgram> tpuProgram, const std::string& datasetPath);
        const std::string& Name() const override;
        size_t TotalSampleCount() override;
        size_t PerformanceSampleCount() override;
        void LoadSamplesToRam(const std::vector<long unsigned int>& samples) override;
        void UnloadSamplesFromRam(const std::vector<long unsigned int>& samples) override;
        std::string GetSamplePath(long unsigned int sample) const;
        SampleBuffer * GetSample(long unsigned sample);
        void LoadValMap(const std::string& valMapPath);
        std::shared_ptr<TPUProgram> GetProgram();
    private:
        bool AllocateSamples(const std::vector<unsigned long> &samples, const TPUIONode *node);
        void LoadSampleToRam(unsigned long sample);
        void UnloadSampleFromRam(unsigned long sample);

    private:
        const std::string name_ = "IVA TPU QSL";
        std::string datasetPath_;
        std::vector<std::pair<std::string, int>> items_;
        std::shared_ptr<TPUProgram> tpuProgram_;
        std::unordered_map<unsigned long, SampleBuffer *> samples_;
    };
}


#endif //IVA_RESNET50_IVATPUQUERYSAMPLELIBRARY_H
