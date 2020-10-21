//
// Created by mmoroz on 8/18/20.
//

#ifndef MLPERF_RESNET50_UTIL_H
#define MLPERF_RESNET50_UTIL_H

#include <string>
#include <memory>

#include <tpu.h>

namespace ivatpu {
    using TPUDevicePtr = std::shared_ptr<TPUDevice>;
    using TPUProgramPtr = std::shared_ptr<TPUProgram>;
    const size_t TPU_PROGRAM_QUEUE_LENGTH = 8;
    std::pair<TPUDevicePtr, TPUProgramPtr> device_program_load(const std::string& model_path);
}

#endif //MLPERF_RESNET50_UTIL_H
