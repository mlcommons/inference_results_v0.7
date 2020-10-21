//
// Created by mmoroz on 8/18/20.
//
#include <iostream>
#include <stdexcept>
#include "util.h"

namespace ivatpu {
    std::pair<TPUDevicePtr, TPUProgramPtr> device_program_load(const std::string &model_path) {
        // Load TPU program
        std::shared_ptr<TPUProgramZipLoader> loader(tpu_program_zip_loader_open(model_path.c_str()),
                                                    tpu_program_zip_loader_close);
        if (!loader) {
            throw std::runtime_error(std::string("Failed to open program ") + model_path);
        }

        std::shared_ptr<TPUProgram> program(tpu_program_open(loader.get()), tpu_program_close);

        if (!program) {
            throw std::runtime_error(std::string("Failed to load program ") + model_path);
        }

        // Open TPU device
        std::shared_ptr<TPUDevice> device(tpu_device_build(), tpu_device_close);
        if (!device) {
            throw std::runtime_error("TPU initialization failed");
        }

        // load program to device
        int rc = tpu_program_load(device.get(), program.get());
        if (rc != 0) {
            throw std::runtime_error("failed to load tpu program to device");
        }

        tpu_program_set_queue_size(program.get(), TPU_PROGRAM_QUEUE_LENGTH);
        return std::make_pair(device, program);
    }
}

