/***************************************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE 
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once 

#include <samples_with_cudnn.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Convolution_kernel;
struct Convolution_kernel_desc {

    // The name of the kernel.
    const char name_[256];
    // The tile size.
    int m_, n_, k_;
    // The number of stages.
    int stages_;
    // Alignment_A
    int Alignment_A;
    // Alignment_C
    int Alignment_C;
    // The function pointer to build the kernel.
    int (*build_)(Convolution_kernel *);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Convolution_kernel {

    // The function to compute the host workspace size.
    int (*compute_host_workspace_size_)(size_t *size_in_bytes);

    // The function to initialize the host workspace.
    int (*initialize_host_workspace_)(void *host_workspace, 
                                      const Convolution_params *params);

    // The function to compute the device workspace size.
    int (*compute_device_workspace_size_)(size_t *size_in_bytes, 
                                          const void *host_ptr);

    // The function to initialize the device workspace.
    int (*initialize_device_workspace_)(void *device_ptr, 
                                        const void *host_ptr, 
                                        cudaStream_t stream);

    // The function to launch the kernel.
    int (*launch_)(void *act,
                   void *flt,
                   void *out,
                   void *bias,
                   void *host_ptr,
                   void *device_ptr,
                   cudaStream_t stream);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

