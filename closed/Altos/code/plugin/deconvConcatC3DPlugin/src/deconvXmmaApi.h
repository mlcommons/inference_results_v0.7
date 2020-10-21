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

#include <stddef.h>

#include <cuda_runtime_api.h>

struct DeconvXmmaHandle_t {
    void* obj;
};

extern "C" int deconvXmmaCreate(DeconvXmmaHandle_t* handle,
                        int num_groups,
                        int c,
                        int stride_c,
                        int k,
                        int nbDims,
                        const int* kernel_size_nd, 
                        const int* stride_nd,
                        const int* padding_nd,
                        const int* dilation_nd,
                        bool is_acc_type_fp32 );

extern "C" int deconvXmmaConfigure(DeconvXmmaHandle_t handle, int n_max,
                                   int d_max, int h_max, int w_max, 
                                   int stride_c, bool use_idx_kernels, int* kernel_id);

extern "C" int deconvXmmaGetWorkspaceSize(DeconvXmmaHandle_t handle, int n, int d, int h, int w, size_t* size);

extern "C" int deconvXmmaEnqueue(DeconvXmmaHandle_t handle, void* act_d, const void* flt_d, 
                                 const void* out_d, const void* bias_d, void* workspace_d, 
                                 int n, int d, int h, int w, cudaStream_t stream);

extern "C" void deconvXmmaDestroy(DeconvXmmaHandle_t handle);

typedef int ( *deconvXmmaCreatePtr_t )
            ( DeconvXmmaHandle_t* handle,
              int num_groups,
              int c,
              int stride_c,
              int k,
              int nbDims,
              const int* kernel_size_nd, 
              const int* stride_nd,
              const int* padding_nd,
              const int* dilation_nd,
              bool is_acc_type_fp32 );

typedef void ( *deconvXmmaGetWorkspaceSizePtr_t )
             ( DeconvXmmaHandle_t handle, int n, 
               int d, int h, int w, size_t* size );

typedef int  ( *deconvXmmaConfigurePtr_t )
             ( DeconvXmmaHandle_t handle,
               int n_max, int d_max, int h_max, int w_max, 
               int stride_c, bool use_idx_kernels, int* kernel_id);

typedef int ( *deconvXmmaEnqueuePtr_t )
            ( DeconvXmmaHandle_t handle,
              void* act_d, const void* flt_d, 
              const void* out_d, const void* bias_d,
              void* workspace_d, int n, int d, int h, int w, 
              cudaStream_t stream);

typedef void ( *deconvXmmaDestroyPtr_t )
             ( DeconvXmmaHandle_t handle );