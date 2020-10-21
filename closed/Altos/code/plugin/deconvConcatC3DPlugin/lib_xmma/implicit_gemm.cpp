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

#define DGRAD_ONLY

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

int main(int argc, char **argv) {

    // The device.
    cudaDeviceProp props;
    XMMA_CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
    int sm = props.major*10 + props.minor;

    // The traits.
    Convolution_traits_desc traits_desc;
    memset(&traits_desc, 0, sizeof(traits_desc));
    traits_desc.arch_     = sm_to_arch(sm);
    traits_desc.act_type_ = DATA_TYPE_FP16;
    traits_desc.flt_type_ = DATA_TYPE_FP16;
    traits_desc.out_type_ = DATA_TYPE_FP16;
    traits_desc.acc_type_ = DATA_TYPE_FP32;
    traits_desc.bias_type_= DATA_TYPE_FP16;

    // The convolution params_.
    Convolution_params params;
    memset(&params, 0, sizeof(params));

    // The layer.
    params.is_cross_correlation_ = true;
    params.g_ = 1;
    params.n_ = 2;
    params.d_ = 1;
    params.h_ = 56;
    params.w_ = 56;
    params.c_ = 64;
    params.k_ = 64;
    params.t_ = 1;
    params.r_ = 1;
    params.s_ = 1;

    // Padding.
    params.pad_d_ = params.t_ / 2;
    params.pad_h_ = params.r_ / 2;
    params.pad_w_ = params.s_ / 2;
    
    // Strides for the convolution.
    params.stride_d_ = 1;
    params.stride_h_ = 1;
    params.stride_w_ = 1;

    // Dilation for the convolution.
    params.dilation_d_ = 1;
    params.dilation_h_ = 1;
    params.dilation_w_ = 1;

    // Alpha/beta config
    params.alpha_[0] = 1.f;
    params.beta_[0]  = 0.f;

    // Use horizontal CTA rasterization by default.
    params.use_horizontal_cta_rasterization_ = true;

    // For split-k we control the # of CTAs in the Z dimension and the # of accumulation buffers.
    params.split_k_slices_  = 1;
    params.split_k_buffers_ = 0;
    params.split_k_kernels_ = 1;

    // The mode ( dgrad).
    // Use the idx kernels.
    bool use_idx_kernels = false;
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
    // Run the validation code on the CPU?
    bool use_cpu = false;
    // Disable result checks.
    bool without_checks = false;
    // Use random alpha beta.
    bool random_alpha_beta = false;
    // The tolerance to check the results.
    float epsilon = 1.e-2f;
    // Is it verbose?
    bool verbose = false;
    // Disable colors.
    bool without_colors = false;
    // Do we use TF32?
    bool use_tf32 = false;
    // Do we use BF16?
    bool use_bf16 = false;
    // Output in nchw layout (tf32 only)
    bool is_out_nchw = false;
    // Input in nchw layout (tf32 only)
    bool is_img_nchw = false;

    // Read the arguments.
    for( int i = 1; i < argc; ++i ) {
        if( !strcmp(argv[i], "-acc") && ++i < argc ) {
            traits_desc.acc_type_ = data_type_from_string(argv[i]);
        } else if( !strcmp(argv[i], "-act") && ++i < argc ) {
            traits_desc.act_type_ = data_type_from_string(argv[i]);
        } else if( !strcmp(argv[i], "-act-1s") ) {
            act_1s = true;
        } else if( !strcmp(argv[i], "-alpha") && ++i < argc ) {
            params.alpha_[0] = strtof(argv[i], nullptr);
        } else if( !strcmp(argv[i], "-beta") && ++i < argc ) {
            params.beta_[0] = strtof(argv[i], nullptr);
        } else if( !strcmp(argv[i], "-c") && ++i < argc ) {
            params.c_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-cfg") && ++i < argc ) {
            cfg = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-conv") ) {
            params.is_cross_correlation_ = false;
        } else if( !strcmp(argv[i], "-cpu") ) {
            use_cpu = true;
        } else if( !strcmp(argv[i], "-d") && ++i < argc ) {
            params.d_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-dilation-d") && ++i < argc ) {
            params.dilation_d_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-dilation-h") && ++i < argc ) {
            params.dilation_h_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-dilation-w") && ++i < argc ) {
            params.dilation_w_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-tf32") ) {
            traits_desc.act_type_ = DATA_TYPE_TF32;
            traits_desc.flt_type_ = DATA_TYPE_TF32;
            traits_desc.out_type_ = DATA_TYPE_TF32;
            traits_desc.acc_type_ = DATA_TYPE_FP32;
            traits_desc.bias_type_= DATA_TYPE_TF32;
            use_tf32             = true;
        } else if( !strcmp(argv[i], "-tf32-fp32") ) {
            traits_desc.act_type_ = DATA_TYPE_FP32;
            traits_desc.flt_type_ = DATA_TYPE_FP32;
            traits_desc.out_type_ = DATA_TYPE_FP32;
            traits_desc.acc_type_ = DATA_TYPE_FP32;
            traits_desc.bias_type_= DATA_TYPE_FP32;
            use_tf32             = true;
        } else if( !strcmp(argv[i], "-bf16") ) {
            traits_desc.act_type_ = DATA_TYPE_BF16;
            traits_desc.flt_type_ = DATA_TYPE_BF16;
            traits_desc.out_type_ = DATA_TYPE_BF16;
            traits_desc.acc_type_ = DATA_TYPE_FP32;
            traits_desc.bias_type_= DATA_TYPE_BF16;
            use_bf16             = true;
        } else if( !strcmp(argv[i], "-epsilon") && ++i < argc ) {
            epsilon = strtof(argv[i], nullptr);
        } else if( !strcmp(argv[i], "-flt") && ++i < argc ) {
            traits_desc.flt_type_ = data_type_from_string(argv[i]);
        } else if( !strcmp(argv[i], "-flt-1s") ) {
            flt_1s = true;
        } else if( !strcmp(argv[i], "-h") && ++i < argc ) {
            params.h_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-idx") ) {
            use_idx_kernels = true;
        } else if( !strcmp(argv[i], "-out-nchw") ) {
            is_out_nchw = true;
        } else if( !strcmp(argv[i], "-img-nchw") ) {
            is_img_nchw = true;
        } else if( !strcmp(argv[i], "-k") && ++i < argc ) {
            params.k_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-n") && ++i < argc ) {
            params.n_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-out") && ++i < argc ) {
            traits_desc.out_type_ = data_type_from_string(argv[i]);
        } else if( !strcmp(argv[i], "-out-1s") ) {
            out_1s = true;
        } else if( !strcmp(argv[i], "-pad-d") && ++i < argc ) {
            params.pad_d_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-pad-h") && ++i < argc ) {
            params.pad_h_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-pad-w") && ++i < argc ) {
            params.pad_w_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-r") && ++i < argc ) {
            params.r_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-random-alpha-beta") ) {
            random_alpha_beta = true;
        } else if( !strcmp(argv[i], "-range") && ++i < argc ) {
            range = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-runs") && ++i < argc ) {
            runs = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-s") && ++i < argc ) {
            params.s_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-scale") && ++i < argc ) {
            scale = strtof(argv[i], nullptr);
        } else if( !strcmp(argv[i], "-sm") && ++i < argc ) {
            traits_desc.arch_ = sm_to_arch(strtol(argv[i], nullptr, 10));
        } else if( !strcmp(argv[i], "-split-k-buffers") && ++i < argc ) {
            params.split_k_buffers_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-split-k-kernels") && ++i < argc ) {
            params.split_k_kernels_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-split-k-r") && ++i < argc ) {
            params.split_k_r_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-split-k-slices") && ++i < argc ) {
            params.split_k_slices_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-split-k-t") && ++i < argc ) {
            params.split_k_t_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-stride-d") && ++i < argc ) {
            params.stride_d_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-stride-h") && ++i < argc ) {
            params.stride_h_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-stride-w") && ++i < argc ) {
            params.stride_w_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-t") && ++i < argc ) {
            params.t_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-use-1s") ) {
            act_1s = flt_1s = out_1s = use_1s = true;
        } else if( !strcmp(argv[i], "-use-bias") ) {
            params.with_bias_ = true;
        } else if( !strcmp(argv[i], "-use-relu") ) {
            // we only support relu with clipping at zero with the command line parameters
            params.with_relu_ = true;
            params.relu_lb_ = 0;
            params.relu_ub_ = +std::numeric_limits<float>::infinity();
        } else if( !strcmp(argv[i], "-use-vertical-cta-rasterization") ) {
            params.use_horizontal_cta_rasterization_ = false;
        } else if( !strcmp(argv[i], "-v") ) {
            verbose = true;
        } else if( !strcmp(argv[i], "-w") && ++i < argc ) {
            params.w_ = strtol(argv[i], nullptr, 10);
        } else if( !strcmp(argv[i], "-without-checks") ) {
            without_checks = true;
        } else if( !strcmp(argv[i], "-without-colors") ) {
            without_colors = true;
        } else {
            fprintf(stderr, "Unrecognized option: %s. Aborting!\n", argv[i]);
            return -1;
        }
    }

    // Running the following command.
    printf("Command.......: %s", argv[0]);
    for( int ii = 1; ii < argc; ++ii ) {
        printf(" %s", argv[ii]);
    }
    printf("\n");

    // Device info.
    if( verbose ) {
        printf("Device........: %s\n", props.name);
        printf("Arch.(sm).....: %d\n", sm);
        printf("#.of.SMs......: %d\n", props.multiProcessorCount);
    }

    // Compute the output dimensions.
    params.compute_output_dimensions();

    // Make sure the split-k params are consistent.
    if( params.split_k_slices_ > 1 ) {
        params.split_k_buffers_ = params.split_k_buffers_ > 1 ? params.split_k_buffers_ : 1;
    }

    // Make sure the split-k flags make sense.
    params.split_k_c_ = params.split_k_slices_ > 1 && !params.split_k_t_ && !params.split_k_r_;

    //if (mode == DGRAD) {
    params.Layout_A = is_out_nchw
        ? xmma::Convolution_layout::NCHW
        : xmma::Convolution_layout::NHWC;
    params.Layout_B = xmma::Convolution_layout::NHWC;
    params.Layout_C = is_img_nchw
        ? xmma::Convolution_layout::NCHW
        : xmma::Convolution_layout::NHWC;

    // Print the layer info.
    if( verbose ) {
        printf("N.............: %d\n", params.n_); 
        printf("D.............: %d\n", params.d_); 
        printf("H.............: %d\n", params.h_); 
        printf("W.............: %d\n", params.w_); 
        printf("C.............: %d\n", params.c_); 
        printf("K.............: %d\n", params.k_); 
        printf("T.............: %d\n", params.t_); 
        printf("R.............: %d\n", params.r_); 
        printf("S.............: %d\n", params.s_); 
        printf("O.............: %d\n", params.o_); 
        printf("P.............: %d\n", params.p_); 
        printf("Q.............: %d\n", params.q_); 
        printf("Pad.d.........: %d\n", params.pad_d_); 
        printf("Pad.h.........: %d\n", params.pad_h_); 
        printf("Pad.w.........: %d\n", params.pad_w_); 
        printf("Stride.d......: %d\n", params.stride_d_); 
        printf("Stride.h......: %d\n", params.stride_h_); 
        printf("Stride.w......: %d\n", params.stride_w_); 
        printf("Dilation.d....: %d\n", params.dilation_d_); 
        printf("Dilation.h....: %d\n", params.dilation_h_); 
        printf("Dilation.w....: %d\n", params.dilation_w_); 
        printf("Convolution...: %s\n", params.is_cross_correlation_ ? "false" : "true");
        printf("Runs..........: %d\n", runs);
    }

    // Make sure we set the seed for reproducible results.
    srand(1234UL);

    // Generate alpha/beta if needed.
    if( random_alpha_beta ) {
        params.alpha_[0] = (double) ((int) rand() % range - range / 2);
        params.alpha_[1] = (double) ((int) rand() % range - range / 2);
        params.beta_ [0] = (double) ((int) rand() % range - range / 2);
        params.beta_ [1] = (double) ((int) rand() % range - range / 2);
    }

    // Print alpha/beta.
    if( verbose ) {
        printf("Alpha.........: Re=%.6lf Im=%.6lf\n", params.alpha_[0], params.alpha_[1]);
        printf("Beta..........: Re=%.6lf Im=%.6lf\n", params.beta_ [0], params.beta_ [1]);
    }

    // Are we doing a 3D convolution?
    bool is_3d = params.t_ > 1;

    // Allocate the activations on the host.
    size_t ndhwc = (size_t) params.n_ * params.d_ * params.h_ * params.w_ * params.c_;
    float *act_h = (float*) malloc(ndhwc * sizeof(float));

    // Allocate the activations on the device.
    const size_t act_sz = get_size_in_bytes(ndhwc, traits_desc.act_type_);
    void *act_d = nullptr;
    XMMA_CHECK_CUDA(cudaMalloc(&act_d, act_sz));

    // Allocate the filters on the host.
    size_t ktrsc = (size_t) params.k_ * params.t_ * params.r_ * params.s_ * params.c_;
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
    const size_t flt_sz = get_size_in_bytes(ktrsc, traits_desc.flt_type_);
    void *flt_d = nullptr;
    XMMA_CHECK_CUDA(cudaMalloc(&flt_d, flt_sz));

    // Copy the filters to the device.
    if( init_flt && is_3d ) {
        XMMA_CHECK_CUDA(cuda_memcpy_to_ncdhw_h2d(flt_d, 
                                                 flt_h, 
                                                 params.k_, 
                                                 params.t_, 
                                                 params.r_, 
                                                 params.s_, 
                                                 params.c_, 
                                                 traits_desc.flt_type_));
    } else if( init_flt ) {
        XMMA_CHECK_CUDA(cuda_memcpy_h2d(flt_d, flt_h, ktrsc, traits_desc.flt_type_));
    }

    float *bias_h = nullptr;  // bias on host
    void *bias_d  = nullptr;  // bias on device

    // Allocate memory for the output tensor on the host.
    uint32_t alignment_c = 16;
    uint32_t alignment_a = 16;
    uint32_t alignment_b = 16;

    if (is_img_nchw) {
        if ((params.d_ * params.h_ * params.w_ % 4 != 0)
            || (params.t_ * params.r_ * params.s_ != 1)
            || (params.stride_d_ * params.stride_h_ * params.stride_w_ != 1)) {
            alignment_a = 4; //bytes
        }
    }

    if (is_out_nchw) {
        if (params.d_ * params.h_ * params.w_ % 4 != 0) {
            alignment_c = 4; //bytes
        }
    }

    size_t nopqk = (size_t) params.n_ * params.o_ * params.p_ * params.q_ * params.k_;
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
    const size_t out_sz = get_size_in_bytes(nopqk, traits_desc.out_type_);
    void *out_d = nullptr;
    XMMA_CHECK_CUDA(cudaMalloc(&out_d, out_sz));

    // Copy the output to the device.
    if( init_out && is_3d ) {
        XMMA_CHECK_CUDA(cuda_memcpy_to_ncdhw_h2d(out_d, 
                                                 out_h, 
                                                 params.n_, 
                                                 params.o_, 
                                                 params.p_, 
                                                 params.q_, 
                                                 params.k_, 
                                                 traits_desc.out_type_));
    } else if( init_out ) {
        XMMA_CHECK_CUDA(cuda_memcpy_h2d(out_d, out_h, nopqk, traits_desc.out_type_));
    }

    // The reference output computed by cuDNN as well as the result.
    float *ref_h = nullptr, *res_h = nullptr;

    // DGRAD
    ref_h = (float*) malloc(ndhwc * sizeof(float));
    res_h = (float*) malloc(ndhwc * sizeof(float));


    // CUDA events to time the code.
    cudaEvent_t start, stop;
    XMMA_CHECK_CUDA(cudaEventCreate(&start));
    XMMA_CHECK_CUDA(cudaEventCreate(&stop));

    // We either run on the CPU or call into cuDNN.
    float cudnn_elapsed = 0.f;
    if( without_checks ) {
        ;
    } else if( use_cpu ) {
        dgrad_ndhwc(ref_h, out_h, flt_h, params, is_img_nchw, is_out_nchw);
    } else {
        // Create the handle for cuDNN.
        cudnnHandle_t handle;
        XMMA_CHECK_CUDNN(cudnnCreate(&handle));

        // Create the descriptor for the convolutions.
        cudnnDataType_t cudnn_acc_type = data_type_to_cudnn(traits_desc.acc_type_);
        cudnnConvolutionDescriptor_t conv_desc = nullptr;
        XMMA_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
        cudnnConvolutionMode_t conv_mode = params.is_cross_correlation_ ? CUDNN_CROSS_CORRELATION :
                                                                          CUDNN_CONVOLUTION;
        if( is_3d ) {
            int conv_pad[] = { params.pad_d_, params.pad_h_, params.pad_w_ };
            int conv_stride[] = { params.stride_d_, params.stride_h_, params.stride_w_ };
            int conv_dilation[] = { params.dilation_d_, params.dilation_h_, params.dilation_w_ };
            XMMA_CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(conv_desc,
                                                             3,
                                                             conv_pad,
                                                             conv_stride,
                                                             conv_dilation,
                                                             conv_mode,
                                                             cudnn_acc_type));
        } else {
            XMMA_CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                             params.pad_h_,
                                                             params.pad_w_,
                                                             params.stride_h_,
                                                             params.stride_w_,
                                                             params.dilation_h_,
                                                             params.dilation_w_,
                                                             conv_mode,
                                                             cudnn_acc_type));
        }

        // cudnnConvolutionBiasActivationForward broken if only bias is specified (it applies relu
        // even when IDENTITY activation is specified. When tensor math is enabled, kernels are
        // selected which properly support bias with IDENTITY activation. Also note that
        // cudnnConvolutionBiasActivationForward works only with algo 1 (IPG).
        if( params.with_bias_ || params.with_relu_ )
            cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);

        // Create the input tensor descriptors.
        cudnnDataType_t act_type = data_type_to_cudnn(traits_desc.act_type_);
        cudnnTensorDescriptor_t act_desc = nullptr;
        XMMA_CHECK_CUDNN(cudnnCreateTensorDescriptor(&act_desc));
        if( is_3d ) {
            int act_dim[] = { params.n_, params.c_, params.d_, params.h_, params.w_ };
            XMMA_CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(act_desc,
                                                          CUDNN_TENSOR_NCHW,
                                                          act_type,
                                                          5,
                                                          act_dim));
        } else {
            XMMA_CHECK_CUDNN(cudnnSetTensor4dDescriptor(act_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        act_type,
                                                        params.n_,
                                                        params.c_,
                                                        params.h_,
                                                        params.w_));
        }

        // Create the filter descriptors.
        cudnnDataType_t flt_type = data_type_to_cudnn(traits_desc.flt_type_);
        cudnnFilterDescriptor_t flt_desc = nullptr;
        XMMA_CHECK_CUDNN(cudnnCreateFilterDescriptor(&flt_desc));
        if( is_3d ) {
            int flt_dim[] = { params.k_, params.c_, params.t_, params.r_, params.s_ };
            XMMA_CHECK_CUDNN(cudnnSetFilterNdDescriptor(flt_desc,
                                                        flt_type,
                                                        CUDNN_TENSOR_NCHW,
                                                        5,
                                                        flt_dim));
        } else {
            XMMA_CHECK_CUDNN(cudnnSetFilter4dDescriptor(flt_desc,
                                                        flt_type,
                                                        CUDNN_TENSOR_NHWC,
                                                        params.k_,
                                                        params.c_,
                                                        params.r_,
                                                        params.s_));
        }

        // Create the bias and relu activation descriptors
        cudnnTensorDescriptor_t bias_desc = nullptr;
        cudnnActivationDescriptor_t activation_desc = nullptr;

        if( params.with_bias_ || params.with_relu_ ) {
            cudnnDataType_t bias_type = data_type_to_cudnn(traits_desc.bias_type_);
            // bias data type for cudnn convolution must be identical to output descriptor according to documentation
            XMMA_CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
            XMMA_CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        data_type_to_cudnn(traits_desc.bias_type_),
                                                        params.n_,
                                                        params.k_,
                                                        1,
                                                        1));
            // Create the activation descriptor (relu)
            XMMA_CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));
            if( params.with_relu_ )
                XMMA_CHECK_CUDNN(cudnnSetActivationDescriptor( activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0 ) );
            else
                XMMA_CHECK_CUDNN(cudnnSetActivationDescriptor( activation_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0 ) );
        }

        // Create the output tensor descriptors.
        cudnnDataType_t out_type = data_type_to_cudnn(traits_desc.out_type_);
        cudnnTensorDescriptor_t out_desc = nullptr;  
        XMMA_CHECK_CUDNN(cudnnCreateTensorDescriptor(&out_desc));
        if( is_3d ) {
            int out_dim[] = { params.n_, params.k_, params.o_, params.p_, params.q_ };
            XMMA_CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(out_desc,
                                                          CUDNN_TENSOR_NCHW,
                                                          out_type,
                                                          5,
                                                          out_dim));
        } else {
            XMMA_CHECK_CUDNN(cudnnSetTensor4dDescriptor(out_desc,
                                                        CUDNN_TENSOR_NHWC,
                                                        out_type,
                                                        params.n_,
                                                        params.k_,
                                                        params.p_,
                                                        params.q_));
        }

        // Always use PRECOMP_GEMM.
        int algo = 0;
        if( use_tf32 || use_bf16 ) {
            algo = 0;
        }else 
        {
            // DGRAD
            algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        }

        // Determine workspace sizes for the different convolutions.
        size_t workspace_sz = 0;
        cudnnConvolutionBwdDataAlgo_t conv_algo = (cudnnConvolutionBwdDataAlgo_t) algo;
        XMMA_CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
                                                                        flt_desc,
                                                                        out_desc,
                                                                        conv_desc,
                                                                        act_desc,
                                                                        conv_algo,
                                                                        &workspace_sz));

        // Allocate workspaces.
        void *workspace_d = nullptr;
        if( workspace_sz > 0 ) {
            XMMA_CHECK_CUDA(cudaMalloc(&workspace_d, workspace_sz));
        }

        // Run cudnn.
        XMMA_CHECK_CUDA(cudaEventRecord(start));
        float alpha = (float) params.alpha_[0], beta = (float) params.beta_[0];
        for( int i = 0; i < runs; ++i ) {
            // DGRAD
            cudnnConvolutionBwdDataAlgo_t conv_algo = (cudnnConvolutionBwdDataAlgo_t) algo;
            XMMA_CHECK_CUDNN(cudnnConvolutionBackwardData(handle,
                                                            &alpha,
                                                            flt_desc,
                                                            flt_d,
                                                            out_desc,
                                                            out_d,
                                                            conv_desc,
                                                            conv_algo,
                                                            workspace_d,
                                                            workspace_sz,
                                                            &beta,
                                                            act_desc,
                                                            act_d));
        }
        XMMA_CHECK_CUDA(cudaEventRecord(stop));
        XMMA_CHECK_CUDA(cudaDeviceSynchronize());

        // Time cuDnn.
        XMMA_CHECK_CUDA(cudaEventElapsedTime(&cudnn_elapsed, start, stop));
        if( verbose ) {
            printf("Cudnn.........: %.3fms\n", cudnn_elapsed / runs);
        }

        // Release cuDNN descriptors and handle.
        XMMA_CHECK_CUDA(cudaFree(workspace_d));
        XMMA_CHECK_CUDNN(cudnnDestroyTensorDescriptor(act_desc));
        XMMA_CHECK_CUDNN(cudnnDestroyFilterDescriptor(flt_desc));
        XMMA_CHECK_CUDNN(cudnnDestroyTensorDescriptor(out_desc));
        XMMA_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
        if( bias_desc ) {
            XMMA_CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
        }
        if( activation_desc ) {
            XMMA_CHECK_CUDNN(cudnnDestroyActivationDescriptor(activation_desc));
        }
        XMMA_CHECK_CUDNN(cudnnDestroy(handle));

        // Copy the results. DGRAD
        if( is_3d || is_out_nchw) {
            float *tmp = (float*) malloc(ndhwc * sizeof(float));
            XMMA_CHECK_CUDA(cuda_memcpy_d2h(tmp, act_d, ndhwc, traits_desc.act_type_));
            ncdhw_to_ndhwc(ref_h, tmp, params.n_, params.c_, params.d_, params.h_, params.w_);
            free(tmp);
        } else {
            XMMA_CHECK_CUDA(cuda_memcpy_d2h(ref_h, act_d, ndhwc, traits_desc.act_type_));
        }
    } // cpu

    // Info.
    if( verbose ) {
        printf("Act.type......: %s\n", data_type_to_string(traits_desc.act_type_));
        printf("Flt.type......: %s\n", data_type_to_string(traits_desc.flt_type_));
        printf("Out.type......: %s\n", data_type_to_string(traits_desc.out_type_));
        printf("Acc.type......: %s\n", data_type_to_string(traits_desc.acc_type_));
        printf("Bias.type.....: %s\n", data_type_to_string(traits_desc.bias_type_));
        printf("Use.idx.......: %s\n", use_idx_kernels ? "true" : "false");
    }

    // Is it a simple 1x1x1 kernel?
    bool simple_1x1x1 = 
        params.t_ * params.stride_d_ * params.dilation_d_ == 1 && params.pad_d_ == 0 && 
        params.r_ * params.stride_h_ * params.dilation_h_ == 1 && params.pad_h_ == 0 && 
        params.s_ * params.stride_w_ * params.dilation_w_ == 1 && params.pad_w_ == 0; 

    // List the kernels.
    int num_kernels;
    // DGRAD
    if( use_idx_kernels ) {
        implicit_gemm_dgrad_indexed_list_kernels(&num_kernels, nullptr, &traits_desc,
            alignment_a, alignment_c, is_img_nchw, is_out_nchw);
    } else {
        implicit_gemm_dgrad_list_kernels(&num_kernels, nullptr, &traits_desc,
            alignment_a, alignment_c, is_img_nchw, is_out_nchw);
    }

    // Info about the number of kernels.
    printf("Num.kernels...: %d\n", num_kernels);
    if( num_kernels == 0 ) {
        return 1;
    }

    // If we have an invalid config.
    if( cfg > 0 && cfg >= num_kernels ) {
        fprintf(stderr, "Invalid configuration %d. Aborting!\n", cfg);
        return 1;
    }

    // Create the list of kernels.
    size_t sz = num_kernels * sizeof(Convolution_kernel_desc);
    Convolution_kernel_desc *kernel_descs = (Convolution_kernel_desc*) malloc(sz);
    // DGRAD
    if(use_idx_kernels ) {
        implicit_gemm_dgrad_indexed_list_kernels(nullptr, kernel_descs, &traits_desc,
            alignment_a, alignment_c, is_img_nchw, is_out_nchw);
    } else {
        implicit_gemm_dgrad_list_kernels(nullptr, kernel_descs, &traits_desc,
            alignment_a, alignment_c, is_img_nchw, is_out_nchw);
    }

    // The best runtime.
    float best_elapsed = FLT_MAX;
    // The best config.
    int best_config = -1;

    // The number of kernels to run.
    int num_kernels_to_run = cfg != -1 ? 1 : num_kernels;
    // The number of failures.
    int failures = 0;

    // Launch the different kernels.
    for( int ii = 0; ii < num_kernels_to_run; ++ii ) {

        // The index of the kernel.
        int kernel_idx = cfg != -1 ? cfg : ii;

        // Reset the inputs on the device for 3D as we want to run NDHWC.
        // DGRAD
        if( is_3d ) {
            XMMA_CHECK_CUDA(cuda_memcpy_h2d(out_d, out_h, nopqk, traits_desc.out_type_));
            XMMA_CHECK_CUDA(cuda_memcpy_h2d(flt_d, flt_h, ktrsc, traits_desc.flt_type_));
        }

        // Reset C. DGRAD

        if( params.beta_[0] != 0.0 && is_3d ) {
            XMMA_CHECK_CUDA(cuda_memcpy_to_ncdhw_h2d(act_d, 
                                                     act_h, 
                                                     params.n_, 
                                                     params.d_, 
                                                     params.h_, 
                                                     params.w_, 
                                                     params.c_, 
                                                     traits_desc.act_type_));
        } else if( params.beta_[0] != 0.0 ) {
            XMMA_CHECK_CUDA(cuda_memcpy_h2d(act_d, act_h, ndhwc, traits_desc.act_type_));
        } else {
            XMMA_CHECK_CUDA(cudaMemset(act_d, 0xdc, act_sz));
        }

        // Info.
        if( verbose ) {
            printf("\n");
            printf("Config........: %d\n", kernel_idx);
            printf("Kernel........: %s\n", kernel_descs[kernel_idx].name_);
        }

        // Create a kernel structure.
        Convolution_kernel kernel;
        memset(&kernel, 0, sizeof(kernel));
        kernel_descs[kernel_idx].build_(&kernel);

        params.nhwc_pitch_c_ = params.c_;// * 2;

        // Allocate memory for the host workspace.
        size_t workspace_sz;
        kernel.compute_host_workspace_size_(&workspace_sz);
        void *workspace_h = malloc(workspace_sz);

        // Initialize the host workspace.
        kernel.initialize_host_workspace_(workspace_h, &params);

        // Allocate memory for the device workspace.
        kernel.compute_device_workspace_size_(&workspace_sz, workspace_h);
        void *workspace_d = nullptr;
        if( workspace_sz > 0 ) {
            XMMA_CHECK_CUDA(cudaMalloc(&workspace_d, workspace_sz));
        }

        // Initialize the device workspace.
        kernel.initialize_device_workspace_(workspace_d, workspace_h, cudaStreamDefault);

        // !!! DGRAD
        assert(params.beta_[0] == 0);
        void *act_pitched_d = nullptr;
        XMMA_CHECK_CUDA(cudaMalloc(&act_pitched_d, 2 * act_sz));
        XMMA_CHECK_CUDA(cudaMemset(act_pitched_d, 0xdc, 2 * act_sz));

        // Run the kernel.
        XMMA_CHECK_CUDA(cudaEventRecord(start));
        bool failed = false;
        for( int i = 0; !failed && i < runs; ++i ) {
            failed = kernel.launch_(act_d,//act_pitched_d,
                                    flt_d,
                                    out_d,
                                    bias_d,
                                    workspace_h,
                                    workspace_d,
                                    cudaStreamDefault);
        }
        XMMA_CHECK_CUDA(cudaEventRecord(stop));
        XMMA_CHECK_CUDA(cudaDeviceSynchronize());

        //!!!
/*         XMMA_CHECK_CUDA(cudaMemcpy2D(act_d, params.c_ * sizeof(uint16_t), act_pitched_d, 
                              params.nhwc_pitch_c_ * sizeof(uint16_t), params.c_ * sizeof(uint16_t),
                              params.n_ * params.d_ * params.h_ * params.w_, cudaMemcpyDeviceToDevice)); */
        cudaFree(act_pitched_d);

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

            // Compare to cudnn (if available).
            if( verbose && !without_checks && !use_cpu ) {
                printf("Ratio.........: %.3fx\n",  cudnn_elapsed / cuda_elapsed);
            }

            // Record the best config.
            if( cuda_elapsed < best_elapsed ) {
                best_elapsed = cuda_elapsed;
                best_config  = kernel_idx;
            }
        }

        // Check the results.
        bool with_checks = !without_checks && !failed;
        if( with_checks) {
            XMMA_CHECK_CUDA(cuda_memcpy_d2h(res_h, act_d, ndhwc, traits_desc.act_type_));
            failures += check_results(res_h, ref_h, ndhwc, 1, 1, epsilon, verbose, !without_colors);
        } else if( verbose ) {
            print_results(!without_colors, false);
        }

        // Clear the workspaces.
        XMMA_CHECK_CUDA(cudaFree(workspace_d));
        free(workspace_h);
    }

    // Print the best.
    if( verbose ) {
        printf("\n");
    }
    printf("Num.executed..: %d\n", num_kernels_to_run);
    bool success = failures == 0 && best_config >= 0;
    if( success && without_checks ) {
        printf("Best..........: %.3fms\n", best_elapsed / runs);
        printf("Config........: %d\n",     best_config);
        printf("Kernel........: %s\n",     kernel_descs[best_config].name_);
        print_results(!without_colors, false);
    } else if( success ) {
        printf("Cudnn.........: %.3fms\n", cudnn_elapsed / runs);
        printf("Best..........: %.3fms\n", best_elapsed / runs);
        printf("Ratio.........: %.3fx\n",  cudnn_elapsed / best_elapsed);
        printf("Config........: %d\n",     best_config);
        printf("Kernel........: %s\n",     kernel_descs[best_config].name_);
        print_results(!without_colors, true, true);
    } else {
        printf("Failures......: %d\n", failures);
        print_results(!without_colors, true, false);
    }

    // Release the cuda events.
    XMMA_CHECK_CUDA(cudaEventDestroy(start));
    XMMA_CHECK_CUDA(cudaEventDestroy(stop));

    // Release memory.
    XMMA_CHECK_CUDA(cudaFree(act_d));
    XMMA_CHECK_CUDA(cudaFree(flt_d));
    XMMA_CHECK_CUDA(cudaFree(out_d));
    if( bias_d ) { XMMA_CHECK_CUDA(cudaFree(bias_d)); }

    // Release memory.
    free(act_h);
    free(flt_h);
    free(out_h);
    free(ref_h);
    free(res_h);
    free(kernel_descs);

    // Reset the device and quit.
    XMMA_CHECK_CUDA(cudaDeviceReset());
    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

