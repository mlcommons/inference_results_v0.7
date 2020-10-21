/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <stdint.h>

#include <cuda_runtime_api.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Input_Data_Type_ = uint16_t, typename Output_Data_Type_ = uint16_t,
          int THREADS_PER_CTA_ = 512,  int THREADS_PER_PIXEL_ = 16, int C_ELEMENTS_PER_CTA_ = 64>
struct Instance_norm_kernel_params {
    enum { USE_ONLINE_APPROACH = 1 };
    enum { THREADS_PER_CTA = THREADS_PER_CTA_ };
    enum { THREADS_PER_PIXEL = THREADS_PER_PIXEL_ }; // 8 or 16

    typedef Input_Data_Type_ Input_Data_Type;
    typedef Output_Data_Type_ Output_Data_Type;

#if 1
    typedef float StorageType;
    enum { PIXELS_PER_THREAD_IN_REGISTERS = 
                                sizeof(StorageType) == 4 ? 2 : 4 };
    enum { PIXELS_PER_THREAD_IN_SMEM = 4};
#else
#if 0
    typedef float StorageType;
#else
    typedef uint16_t StorageType;
#endif

    enum { PIXELS_PER_THREAD_IN_REGISTERS = 
        sizeof(StorageType) == 4 ? 
            (USE_ONLINE_APPROACH ? 16 : 16) :  // Config for 7x7
            (USE_ONLINE_APPROACH ? 33 : 18) };
    enum { PIXELS_PER_THREAD_IN_SMEM = 7 };
#endif

    enum { PIXELS_PER_THREAD = PIXELS_PER_THREAD_IN_REGISTERS + PIXELS_PER_THREAD_IN_SMEM };
    enum { C_ELEMENTS_PER_CTA = C_ELEMENTS_PER_CTA_ }; //64;
    enum { ELEMENTS_PER_LDG = C_ELEMENTS_PER_CTA / THREADS_PER_PIXEL };  // 4 default

    // Derived params.
    enum { PIXELS_PER_CTA = THREADS_PER_CTA/THREADS_PER_PIXEL * PIXELS_PER_THREAD };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct InstanceNormFwdParams {
    // The input/output tensors.
    void *gmem_src, *gmem_dst;
    // The bias/scale.
    float *gmem_bias, *gmem_scale;
    // running mean/var (refer BN API from cudnn doc)
    float *gmem_running_mean, *gmem_running_var;
    // saved mean/var (refer BN API from cudnn doc)
    float *gmem_saved_mean, *gmem_saved_var;
    // The dimensions.
    int nhw, c, n;
    // The buffer to do the reduction for mean, stddev and count.
    float *gmem_sums;
    // The buffer to count items in the different CTAs.
    int *gmem_counts;
    // The counters of retired CTAs.
    int *gmem_retired_ctas;
    // The epsilon to apply to the computation of the variance.
    float var_eps;
    // outer loop count
    int outer_loops;
    // exponential average factor
    float exp_avg_factor;
    // use relu as activation?
    bool use_relu;
    float relu_alpha;

    int sm_count;

    float in_scale;

    float out_scale;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void instance_norm_buffer_sizes_dispatch(InstanceNormFwdParams params, 
                                size_t &size_sums, size_t &size_counts, size_t &size_retired_ctas,
                                int input_data_type = 1, int output_data_type = 1);

int instance_norm_fwd_dispatch(InstanceNormFwdParams params, cudaStream_t stream,
                               int input_data_type = 1, int output_data_type = 1);