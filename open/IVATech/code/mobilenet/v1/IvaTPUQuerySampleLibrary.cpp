//
// Created by mmoroz on 7/9/20.
//

#include <fstream>
#include <iostream>
#include <algorithm>
#include <execution>
#include <stdexcept>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <spdlog/spdlog.h>
#include <tpu_tensor.h>

#include <spdlog/spdlog.h>

#include "imagenet/ImageNetUtils.h"

#include "IvaTPUQuerySampleLibrary.h"

ivatpu::IvaTPUQuerySampleLibrary::IvaTPUQuerySampleLibrary(std::shared_ptr<TPUProgram> tpuProgram,
                                                           const std::string &datasetPath) : tpuProgram_(tpuProgram),
                                                                                             datasetPath_(
                                                                                                     datasetPath) {

}

const std::string &ivatpu::IvaTPUQuerySampleLibrary::Name() const {
    return name_;
}

size_t ivatpu::IvaTPUQuerySampleLibrary::TotalSampleCount() {
    return items_.size();
}

size_t ivatpu::IvaTPUQuerySampleLibrary::PerformanceSampleCount() {
    return 10;
}

void ivatpu::IvaTPUQuerySampleLibrary::LoadSamplesToRam(const std::vector<unsigned long> &samples) {
    spdlog::info("Load {} samples", samples.size());
    // allocate memory for samples
    if(!AllocateSamples(samples, tpu_program_get_input_node(tpuProgram_.get(), 0))) {
        throw std::runtime_error("Can't allocate samples memory");
    };
    std::for_each(std::execution::par_unseq, samples.begin(), samples.end(), [this](auto s){LoadSampleToRam(s);});
}

void ivatpu::IvaTPUQuerySampleLibrary::UnloadSamplesFromRam(const std::vector<unsigned long> &samples) {
//    std::cout << "Unload " << samples.size() << " from RAM" << std::endl;
    for(auto s: samples) {
        UnloadSampleFromRam(s);
        samples_.erase(s);
    }
}

void ivatpu::IvaTPUQuerySampleLibrary::LoadValMap(const std::string &valMapPath) {
    std::ifstream mapFile(valMapPath);
    if(!mapFile) throw std::runtime_error(std::string("failed to load values map from ") + valMapPath);
    while(!mapFile.eof()) {
        std::string fileName;
        int label;
        mapFile >> fileName >> label;
        if(!fileName.empty()) {
            items_.emplace_back(fileName, label);
        }
    }
}

bool ivatpu::IvaTPUQuerySampleLibrary::AllocateSamples(const std::vector<unsigned long> &samples, const TPUIONode *node)
{
    const size_t dataSize = tpu_tensor_get_dimension_size(&node->tpu_shape[1],
                                                          node->tpu_shape_len - 1,
                                                          tpu_tensor_get_elem_size(node->dtype));
    size_t sampleMemorySize = sizeof(SampleBuffer) + dataSize;

    for(size_t i = 0; i < samples.size(); ++i) {
        // pointer arithmetics :)
        SampleBuffer *s = (ivatpu::SampleBuffer *)malloc(sampleMemorySize);
        if(s == nullptr) {
            throw std::runtime_error("failed to allocate sample memory");
        }
        s->data = (void *)(s + 1);
        s->size = dataSize;
        auto sampleId = samples[i];
        samples_[sampleId] = s;
    }

    return true;
}

void ivatpu::IvaTPUQuerySampleLibrary::LoadSampleToRam(unsigned long sample) {
    const std::string path = GetSamplePath(sample);

    try {
        //spdlog::debug("Sample {} path {}", sample, path);
        auto image = cv::imread(path);
        if ( !image.data ) {
            throw std::runtime_error(std::string("Can't open image ") + path) ;
        }

        const size_t WIDTH = 224;
        const size_t HEIGHT = 224;
        const size_t CHANNELS = 3;
        int shape[] = {1, HEIGHT, WIDTH, CHANNELS};

        std::shared_ptr<TPUTensor> input_tensor(tpu_tensor_allocate(shape, 4, TPU_INT8, malloc), tpu_tensor_free);
        auto input_node = tpu_program_get_input_node(tpuProgram_.get(), 0);
        imagenet::image_to_tensor(image, input_tensor.get(), input_node->scale[0]);
        // write tensor to buffer
        SampleBuffer *sampleBuffer = samples_[sample];

        const struct TPUIONode *node = tpu_program_get_input_node(tpuProgram_.get(), 0);
        int tpu_sample_shape[8];
        memcpy(tpu_sample_shape, node->tpu_shape, sizeof(int) * node->tpu_shape_len);
        tpu_sample_shape[0] = 1; // FIXME: sorry, I need to implement it in libtpu
        std::shared_ptr<TPUTensor> tpu_tensor(tpu_tensor_allocate(tpu_sample_shape, node->tpu_shape_len, node->dtype, malloc), tpu_tensor_free);
        int rc = tpu_tensor_copy_with_padding(tpu_tensor.get(), input_tensor.get(), node->padding);
        if(rc != 0) {
            throw std::runtime_error("can't convert user tensor to tpu tensor");
        }
        memcpy(sampleBuffer->data, tpu_tensor->data, tpu_tensor->size);
    } catch (std::out_of_range& e) {
        throw std::runtime_error(std::string("failed to load ") + path);
    }
}

ivatpu::SampleBuffer * ivatpu::IvaTPUQuerySampleLibrary::GetSample(unsigned long sample)
{
    return samples_[sample];
}

std::string ivatpu::IvaTPUQuerySampleLibrary::GetSamplePath(unsigned long sample) const {
    return datasetPath_ + "/" + items_[sample].first;
}

std::shared_ptr<TPUProgram> ivatpu::IvaTPUQuerySampleLibrary::GetProgram() {
    return tpuProgram_;
}

void ivatpu::IvaTPUQuerySampleLibrary::UnloadSampleFromRam(unsigned long sample) {
    free(samples_[sample]);
}
