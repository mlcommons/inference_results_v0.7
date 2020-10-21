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
#include <limits>
#include <implicit_gemm.h>

#include <deconvXmmaApi.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

int implicit_gemm_dgrad_an_cn_list_kernels(int *, Convolution_kernel_desc *, const Convolution_traits_desc *, uint32_t, uint32_t);
int implicit_gemm_dgrad_at_ct_list_kernels(int *, Convolution_kernel_desc *, const Convolution_traits_desc *, uint32_t, uint32_t);
int implicit_gemm_dgrad_an_ct_list_kernels(int *, Convolution_kernel_desc *, const Convolution_traits_desc *, uint32_t, uint32_t);
int implicit_gemm_dgrad_at_cn_list_kernels(int *, Convolution_kernel_desc *, const Convolution_traits_desc *, uint32_t, uint32_t);

////////////////////////////////////////////////////////////////////////////////////////////////////

int implicit_gemm_dgrad_indexed_an_cn_list_kernels(int *, Convolution_kernel_desc *, const Convolution_traits_desc *, uint32_t, uint32_t);
int implicit_gemm_dgrad_indexed_at_ct_list_kernels(int *, Convolution_kernel_desc *, const Convolution_traits_desc *, uint32_t, uint32_t);
int implicit_gemm_dgrad_indexed_an_ct_list_kernels(int *, Convolution_kernel_desc *, const Convolution_traits_desc *, uint32_t, uint32_t);
int implicit_gemm_dgrad_indexed_at_cn_list_kernels(int *, Convolution_kernel_desc *, const Convolution_traits_desc *, uint32_t, uint32_t);

////////////////////////////////////////////////////////////////////////////////////////////////////

void implicit_gemm_dgrad_list_kernels(int *num_kernels,
    Convolution_kernel_desc *kernel_descs,
    const Convolution_traits_desc *traits_desc, uint32_t alignment_a, uint32_t alignment_c,
    bool is_img_nchw, bool is_out_nchw) {

    if (is_img_nchw) {
        if (is_out_nchw) {
            implicit_gemm_dgrad_an_cn_list_kernels(num_kernels, kernel_descs, traits_desc,
                alignment_a, alignment_c);
        } else {
            implicit_gemm_dgrad_an_ct_list_kernels(num_kernels, kernel_descs, traits_desc,
                alignment_a, alignment_c);
        }
    } else {
        if (is_out_nchw) {
            implicit_gemm_dgrad_at_cn_list_kernels(num_kernels, kernel_descs, traits_desc,
                alignment_a, alignment_c);
        } else {
            implicit_gemm_dgrad_at_ct_list_kernels(num_kernels, kernel_descs, traits_desc,
                alignment_a, alignment_c);
        }
    }
}

void implicit_gemm_dgrad_indexed_list_kernels(int *num_kernels,
    Convolution_kernel_desc *kernel_descs,
    const Convolution_traits_desc *traits_desc, uint32_t alignment_a, uint32_t alignment_c,
    bool is_img_nchw, bool is_out_nchw) {

    if (is_img_nchw) {
        if (is_out_nchw) {
            implicit_gemm_dgrad_indexed_an_cn_list_kernels(num_kernels, kernel_descs, traits_desc,
                alignment_a, alignment_c);
        } else {
            implicit_gemm_dgrad_indexed_an_ct_list_kernels(num_kernels, kernel_descs, traits_desc,
                alignment_a, alignment_c);
        }
    } else {
        if (is_out_nchw) {
            implicit_gemm_dgrad_indexed_at_cn_list_kernels(num_kernels, kernel_descs, traits_desc,
                alignment_a, alignment_c);
        } else {
            implicit_gemm_dgrad_indexed_at_ct_list_kernels(num_kernels, kernel_descs, traits_desc,
                alignment_a, alignment_c);
        }
    }
}

struct DeconvConcatXmma {

    Convolution_traits_desc traits_desc_;
    //Convolution_traits_desc traits_indexed_desc_;
    Convolution_params params_;

    Convolution_kernel kernel_;
    Convolution_kernel_desc *kernel_descs_;

    DeconvConcatXmma() {
        init();
    }

    DeconvConcatXmma( int num_groups,
                     int c,
                     int stride_c,
                     int k,
                     int nbDims,
                     const int* kernel_size_nd, 
                     const int* stride_nd,
                     const int* padding_nd,
                     const int* dilation_nd,
                     bool is_acc_type_fp32 ) {
        init();
        assert(nbDims = 3);
        params_.t_ = kernel_size_nd[0];
        params_.r_ = kernel_size_nd[1];
        params_.s_ = kernel_size_nd[2];

        // Strides for the convolution.
        params_.stride_d_ = stride_nd[0];
        params_.stride_h_ = stride_nd[1];
        params_.stride_w_ = stride_nd[2];

        // Padding.
        params_.pad_d_ = padding_nd[0];
        params_.pad_h_ = padding_nd[1];
        params_.pad_w_ = padding_nd[2];

        // Dilation for the convolution.
        params_.dilation_d_ = dilation_nd[0];
        params_.dilation_h_ = dilation_nd[1];
        params_.dilation_w_ = dilation_nd[2];

        traits_desc_.acc_type_ = (is_acc_type_fp32)? DATA_TYPE_FP32 :
                                                     DATA_TYPE_FP16;

        params_.c_ = c;
        params_.k_ = k;

        params_.nhwc_pitch_c_ = stride_c; // e.g. params.c_ * 2;
    }

    ~DeconvConcatXmma() {
        terminate();
    }

    void configure(int n_max, int d_max, int h_max, int w_max,
                   int stride_c, int use_idx_kernels, int& kernel_id)
    {
        params_.nhwc_pitch_c_ = stride_c;
        if (kernel_id == -1) {
    //****************** Find best kernel*************************
            void *bias_d  = nullptr;  // bias on device

            // Do we force a specific config?
            int cfg = -1;
            // The number of runs to evaluate performance.
            int runs = 1;
            // The range for random number generation.
            int range = 5;
            // The scaling to apply to random numbers.
            float scale = 1.f;
            // Do we use 1s to initialize the different tensors?
            bool act_1s = false, flt_1s = false, out_1s = false, use_1s=false;
            // Use random alpha beta.
            bool random_alpha_beta = false;
            // The tolerance to check the results.
            float epsilon = 1.e-2f;
            // Is it verbose?
            bool verbose = false;
            // Disable colors.
            bool without_colors = false;

            params_.d_ = d_max;
            params_.h_ = h_max;
            params_.w_ = w_max;
            params_.n_ = n_max;

            params_.compute_output_dimensions();

            bool is_3d = (params_.t_ > 1) || (d_max > 1);

            // if alignment_a should be 4 - need ti regenerate the kernel list
            assert(!is_img_nchw_ || 
                !((params_.d_ * params_.h_ * params_.w_ % 4 != 0)
                    || (params_.t_ * params_.r_ * params_.s_ != 1)
                    || (params_.stride_d_ * params_.stride_h_ * params_.stride_w_ != 1)));

            // if alignment c should be 4
            assert(!is_out_nchw_ ||
                ! (params_.d_ * params_.h_ * params_.w_ % 4 != 0));

            size_t ndhw2c = (size_t) params_.n_ * params_.d_ * params_.h_ * params_.w_ * params_.nhwc_pitch_c_;
            // Allocate the activations on the device.
            const size_t act_sz = get_size_in_bytes(ndhw2c, traits_desc_.act_type_);
            void *act_d = nullptr;
            XMMA_CHECK_CUDA(cudaMalloc(&act_d, act_sz));

            // Allocate the filters on the host.
            size_t ktrsc = (size_t) params_.k_ * params_.t_ * params_.r_ * params_.s_ * params_.c_;
            float *flt_h = (float*) malloc(ktrsc * sizeof(float));

            // Initialize if needed.
            bool init_flt = true;
            if( init_flt ) {
                if( verbose ) { 
                    printf("Flt...........:\n");
                }
                random_init(flt_h, ktrsc, range, scale, flt_1s, verbose);
            }

            // Allocate the filters on the device.
            const size_t flt_sz = get_size_in_bytes(ktrsc, traits_desc_.flt_type_);
            void *flt_d = nullptr;
            XMMA_CHECK_CUDA(cudaMalloc(&flt_d, flt_sz));


            size_t nopqk = (size_t) params_.n_ * params_.o_ * params_.p_ * params_.q_ * params_.k_;
            float *out_h = (float*) malloc(nopqk * sizeof(float));

            // Initialize if needed.
            bool init_out = true;
            if( init_out ) {
                if( verbose ) {
                    printf("Out...........:\n");
                }
                random_init(out_h, nopqk, range, scale, out_1s, verbose);
            }

        // Allocate the output on the device.
        const size_t out_sz = get_size_in_bytes(nopqk, traits_desc_.out_type_);
        void *out_d = nullptr;
        XMMA_CHECK_CUDA(cudaMalloc(&out_d, out_sz));

        // Copy the output to the device.
        if( init_out && is_3d ) {
            XMMA_CHECK_CUDA(cuda_memcpy_to_ncdhw_h2d(out_d, 
                                                    out_h, 
                                                    params_.n_, 
                                                    params_.o_, 
                                                    params_.p_, 
                                                    params_.q_, 
                                                    params_.k_, 
                                                    traits_desc_.out_type_));
        } else if( init_out ) {
            XMMA_CHECK_CUDA(cuda_memcpy_h2d(out_d, out_h, nopqk, traits_desc_.out_type_));
        }

            // CUDA events to time the code.
            cudaEvent_t start, stop;
            XMMA_CHECK_CUDA(cudaEventCreate(&start));
            XMMA_CHECK_CUDA(cudaEventCreate(&stop));

            // The best runtime.
            float best_elapsed = FLT_MAX;
            // The best config.
            int best_config = -1;

            // The number of kernels to run.
            int num_kernels_to_run = cfg_ != -1 ? 1 : num_kernels_ + ((use_idx_kernels)?num_idx_kernels_:0);
            // The number of failures.
            int failures = 0;

            // Launch the different kernels.
            for( int ii = 0; ii < num_kernels_to_run; ++ii ) {

                // The index of the kernel.
                int kernel_idx = cfg_ != -1 ? cfg_ : ii;

                // Reset the inputs on the device for 3D as we want to run NDHWC.
                // DGRAD
                if( is_3d ) {
                    XMMA_CHECK_CUDA(cuda_memcpy_h2d(out_d, out_h, nopqk, traits_desc_.out_type_));
                    XMMA_CHECK_CUDA(cuda_memcpy_h2d(flt_d, flt_h, ktrsc, traits_desc_.flt_type_));
                }

                // Reset C. DGRAD
                assert(params_.beta_[0] == 0);

                XMMA_CHECK_CUDA(cudaMemset(act_d, 0xdc, act_sz));

                // Info.
                if( verbose ) {
                    printf("\n");
                    printf("Config........: %d\n", kernel_idx);
                    printf("Kernel........: %s\n", kernel_descs_[kernel_idx].name_);
                }

                // Create a kernel structure.
                Convolution_kernel kernel;
                memset(&kernel, 0, sizeof(kernel));
                kernel_descs_[kernel_idx].build_(&kernel);
                
                // Allocate memory for the host workspace.
                size_t workspace_sz;
                kernel.compute_host_workspace_size_(&workspace_sz);
                void *workspace_h = malloc(workspace_sz);

                // Initialize the host workspace.
                kernel.initialize_host_workspace_(workspace_h, &params_);

                // Allocate memory for the device workspace.
                kernel.compute_device_workspace_size_(&workspace_sz, workspace_h);
                void *workspace_d = nullptr;
                if( workspace_sz > 0 ) {
                    XMMA_CHECK_CUDA(cudaMalloc(&workspace_d, workspace_sz));
                }

                // Initialize the device workspace.
                kernel.initialize_device_workspace_(workspace_d, workspace_h, cudaStreamDefault);

                // Run the kernel.
                XMMA_CHECK_CUDA(cudaEventRecord(start));
                bool failed = false;
                for( int i = 0; !failed && i < runs; ++i ) {
                    failed = kernel.launch_(act_d,
                                            flt_d,
                                            out_d,
                                            bias_d,
                                            workspace_h,
                                            workspace_d,
                                            cudaStreamDefault);
                }
                XMMA_CHECK_CUDA(cudaEventRecord(stop));
                XMMA_CHECK_CUDA(cudaDeviceSynchronize());

                // Print the runtime. TODO: Deal with "Not enough shared memory issues"!
                if( failed ) {
                    if( verbose ) {
                        printf("Xmma..........: Failed with error code = %d\n", failed);
                    }
                } else {
                    float cuda_elapsed;
                    XMMA_CHECK_CUDA(cudaEventElapsedTime(&cuda_elapsed, start, stop));
                    if( verbose ) {
                        printf("Xmma..........: %.3fms\n", cuda_elapsed / runs);
                    }

                    // Record the best config.
                    if( cuda_elapsed < best_elapsed ) {
                        best_elapsed = cuda_elapsed;
                        best_config  = kernel_idx;
                    }
                }

                if( verbose ) {
                    print_results(!without_colors, false);
                }

                // Clear the workspaces.
                XMMA_CHECK_CUDA(cudaFree(workspace_d));
                free(workspace_h);
            }

            // Release the cuda events.
            XMMA_CHECK_CUDA(cudaEventDestroy(start));
            XMMA_CHECK_CUDA(cudaEventDestroy(stop));

            // Print the best.
            if( verbose ) {
                printf("\n");
            }
            printf("Num.executed..: %d\n", num_kernels_to_run);
            bool success = failures == 0 && best_config >= 0;
            if( success ) {
                printf("Best..........: %.3fms\n", best_elapsed / runs);
                printf("Config........: %d\n",     best_config);
                printf("Kernel........: %s\n",     kernel_descs_[best_config].name_);
                print_results(!without_colors, false);
            } else if( success ) {
                printf("Best..........: %.3fms\n", best_elapsed / runs);
                printf("Config........: %d\n",     best_config);
                printf("Kernel........: %s\n",     kernel_descs_[best_config].name_);
                print_results(!without_colors, true, true);
            } else {
                printf("Failures......: %d\n", failures);
                print_results(!without_colors, true, false);
            }


            // Release memory.
            XMMA_CHECK_CUDA(cudaFree(act_d));
            XMMA_CHECK_CUDA(cudaFree(flt_d));
            XMMA_CHECK_CUDA(cudaFree(out_d));

            // Release memory.
            free(flt_h);
            free(out_h);

            kernel_id = best_config;

    //****************** End find best kernel*******************
        }
        // Initialize best kernel
        assert( kernel_id >= 0 );
        cfg_ = kernel_id;
        kernel_descs_[cfg_].build_(&kernel_);

        // Allocate memory for the host workspace.
        size_t workspace_sz;
        kernel_.compute_host_workspace_size_(&workspace_sz);
        workspace_h_ = malloc(workspace_sz);
        kernel_.initialize_host_workspace_(workspace_h_, &params_);
    }

    size_t get_device_workspace_size(int n, int d, int h, int w) {
        assert( cfg_ >= 0 );
        params_.n_ = n;
        params_.d_ = d;
        params_.h_ = h;
        params_.w_ = w;

        params_.compute_output_dimensions();

        size_t workspace_sz = 0;
        if (workspace_h_ != nullptr) free(workspace_h_);
        kernel_.compute_host_workspace_size_(&workspace_sz);
        workspace_h_ = malloc(workspace_sz);
        kernel_.initialize_host_workspace_(workspace_h_, &params_);
        kernel_.compute_device_workspace_size_(&workspace_sz, workspace_h_);
        return workspace_sz;
    }

    void enqueue(void* act_d, void* flt_d, void* out_d, void* bias_d, void* workspace_d,
                 int n, int d, int h, int w, cudaStream_t stream) {
        params_.n_ = n;
        params_.d_ = d;
        params_.h_ = h;
        params_.w_ = w;

        assert(cfg_ >= 0);
        assert(bias_d == nullptr);
        assert(workspace_h_ != nullptr);

        cudaMemsetAsync(act_d, 0xdc, n * d * h * w * params_.nhwc_pitch_c_, stream);

        // Compute the output dimensions.
        params_.compute_output_dimensions();

        // Initialize the host workspace. stride_c is already set in constructor
        kernel_.initialize_host_workspace_(workspace_h_, &params_);

        // Initialize the device workspace.
        kernel_.initialize_device_workspace_(workspace_d, workspace_h_, stream);

        // Run the kernel.
        bool failed = false;
        failed = kernel_.launch_(act_d,
                                flt_d,
                                out_d,
                                bias_d,
                                workspace_h_,
                                workspace_d,
                                stream);
    }
private:
    // Output in nchw layout (tf32 only)
    bool is_out_nchw_;
    // Input in nchw layout (tf32 only)
    bool is_img_nchw_;
    int cfg_;
    int num_kernels_;
    int num_idx_kernels_;
    
    void *workspace_h_;

    void terminate() {
        if (kernel_descs_ != nullptr) free(kernel_descs_);
        if (workspace_h_ != nullptr) free(workspace_h_);
    }

    void init(){
        is_out_nchw_ = false;
        is_img_nchw_ = false;
        cfg_ = -1;
        workspace_h_ = nullptr;
        kernel_descs_ = nullptr;

        memset(&kernel_, 0, sizeof(kernel_));

        // The device.
        cudaDeviceProp props;
        XMMA_CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
        int sm = props.major*10 + props.minor;
        memset(&traits_desc_, 0, sizeof(traits_desc_));
        traits_desc_.arch_     = sm_to_arch(sm);
        traits_desc_.act_type_ = DATA_TYPE_FP16;
        traits_desc_.flt_type_ = DATA_TYPE_FP16;
        traits_desc_.out_type_ = DATA_TYPE_FP16;
        traits_desc_.acc_type_ = DATA_TYPE_FP32;
        traits_desc_.bias_type_= DATA_TYPE_FP16;

        memset(&params_, 0, sizeof(params_));
        // The layer.
        params_.is_cross_correlation_ = true;
        params_.g_ = 1;
        params_.n_ = 2;
        params_.d_ = 64;
        params_.h_ = 64;
        params_.w_ = 64;
        params_.c_ = 32;
        params_.k_ = 64;
        params_.t_ = 1;
        params_.r_ = 1;
        params_.s_ = 1;

        // Padding.
        params_.pad_d_ = params_.t_ / 2;
        params_.pad_h_ = params_.r_ / 2;
        params_.pad_w_ = params_.s_ / 2;
        
        // Strides for the convolution.
        params_.stride_d_ = 1;
        params_.stride_h_ = 1;
        params_.stride_w_ = 1;

        // Dilation for the convolution.
        params_.dilation_d_ = 1;
        params_.dilation_h_ = 1;
        params_.dilation_w_ = 1;

        // Alpha/beta config
        params_.alpha_[0] = 1.f;
        params_.beta_[0]  = 0.f;
        params_.alpha_[1] = 1.f;
        params_.beta_[1]  = 0.f;

        // Use horizontal CTA rasterization by default.
        params_.use_horizontal_cta_rasterization_ = true;

        // For split-k we control the # of CTAs in the Z dimension and the # of accumulation buffers.
        params_.split_k_slices_  = 1;
        params_.split_k_buffers_ = 0;
        params_.split_k_kernels_ = 1;

        // Make sure the split-k params are consistent.
        if( params_.split_k_slices_ > 1 ) {
            params_.split_k_buffers_ = params_.split_k_buffers_ > 1 ? params_.split_k_buffers_ : 1;
        }

        // Make sure the split-k flags make sense.
        params_.split_k_c_ = params_.split_k_slices_ > 1 && !params_.split_k_t_ && !params_.split_k_r_;

        //if (mode == DGRAD) {
        params_.Layout_A = is_out_nchw_
            ? xmma::Convolution_layout::NCHW
            : xmma::Convolution_layout::NHWC;
        params_.Layout_B = xmma::Convolution_layout::NHWC;
        params_.Layout_C = is_img_nchw_
            ? xmma::Convolution_layout::NCHW
            : xmma::Convolution_layout::NHWC;


        // Allocate memory for the output tensor on the host.
        uint32_t alignment_c = 16;
        uint32_t alignment_a = 16;
        uint32_t alignment_b = 16;

        // List the kernels.
        // DGRAD
        // Create a list of non-indexed kernels
        num_kernels_ = 0;
        implicit_gemm_dgrad_list_kernels(&num_kernels_, nullptr, &traits_desc_,
            alignment_a, alignment_c, is_img_nchw_, is_out_nchw_);
        // Create a list of indexed kernels
        num_idx_kernels_ = 0;
        implicit_gemm_dgrad_indexed_list_kernels(&num_idx_kernels_, nullptr, &traits_desc_,
            alignment_a, alignment_c, is_img_nchw_, is_out_nchw_);

        // Info about the number of kernels.
        //printf("Num.kernels...: %d\n", num_kernels_);
        assert(num_kernels_ + num_idx_kernels_ > 0);

        // Create the list of kernels.
        size_t sz = (num_kernels_ + num_idx_kernels_) * sizeof(Convolution_kernel_desc);
        kernel_descs_ = (Convolution_kernel_desc*) malloc(sz);
        // DGRAD
        if (num_kernels_ > 0) {
            implicit_gemm_dgrad_list_kernels(nullptr, kernel_descs_, &traits_desc_,
                alignment_a, alignment_c, is_img_nchw_, is_out_nchw_);
        }
        if (num_idx_kernels_ > 0) {
            implicit_gemm_dgrad_indexed_list_kernels(nullptr, &kernel_descs_[num_kernels_], &traits_desc_,
                alignment_a, alignment_c, is_img_nchw_, is_out_nchw_);
        }
    }
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
                        bool is_acc_type_fp32 ) {
    handle->obj = new DeconvConcatXmma( num_groups,
                     c,
                     stride_c,
                     k,
                     nbDims,
                     kernel_size_nd, 
                     stride_nd,
                     padding_nd,
                     dilation_nd,
                     is_acc_type_fp32 );
    return 0;
}

extern "C" int deconvXmmaConfigure(DeconvXmmaHandle_t handle,int n_max,
                                   int d_max, int h_max, int w_max, 
                                   int stride_c, bool use_idx_kernels, int* kernel_id) {
    static_cast<DeconvConcatXmma *>(handle.obj)->configure(n_max, d_max, h_max, w_max, stride_c,
                                                           use_idx_kernels, *kernel_id);
    return 0;
}

extern "C" int deconvXmmaGetWorkspaceSize(DeconvXmmaHandle_t handle, int n, 
                                          int d, int h, int w, size_t* size) {
    *size = static_cast<DeconvConcatXmma *>(handle.obj)->get_device_workspace_size(n, d, h, w);
    return 0;
}

extern "C" int deconvXmmaEnqueue(DeconvXmmaHandle_t handle, void* act_d, const void* flt_d,
                                 const void* out_d, const void* bias_d, void* workspace_d, 
                                 int n, int d, int h, int w, cudaStream_t stream) {
    static_cast<DeconvConcatXmma *>(handle.obj)->enqueue(act_d, (void*)flt_d, (void*)out_d, 
                                                        (void*)bias_d, workspace_d,
                                                         n, d, h, w, stream);
    return 0;
}

extern "C" void deconvXmmaDestroy(DeconvXmmaHandle_t handle) {
    delete static_cast<DeconvConcatXmma *>(handle.obj);
}