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

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <xmma/numeric_types.h>
#include <xmma/params.h>

////////////////////////////////////////////////////////////////////////////////////////////////////


#define XMMA_CHECK_CUDA(call) do { \
  cudaError status_ = call; \
  if( status_ != cudaSuccess ) { \
    fprintf(stderr, "Cuda error in file \"%s\" at line %d: %s\n", \
        __FILE__, \
        __LINE__, \
        cudaGetErrorString(status_)); \
    exit(1); \
  } \
} while(0)


////////////////////////////////////////////////////////////////////////////////////////////////////
//
// A R C H / D A T A   T Y P E S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

enum Arch {
    ARCH_VOLTA  = 0x1,
    ARCH_TURING = 0x2,
    ARCH_AMPERE = 0x4,
    ARCH_HOPPER = 0x8
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline Arch sm_to_arch(int sm) {
    if( sm == 70 || sm == 72 ) {
        return ARCH_VOLTA;
    } else if( sm == 75 ) {
        return ARCH_TURING;
    } else if( sm == 80 || sm == 82 ) {
        return ARCH_AMPERE;
    } else if( sm == 90 ) {
        return ARCH_HOPPER;
    } else {
        assert(false);
    }
    return ARCH_VOLTA;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

enum Data_type {
    DATA_TYPE_BOOL,
    DATA_TYPE_TF32,
    DATA_TYPE_BF16,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT8x32,
    DATA_TYPE_INT32,
    DATA_TYPE_FP64
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Gemm_traits_desc {

    // The architecture.
    unsigned arch_;
    // The type of the elements of A.
    Data_type a_type_;
    // The type of the elements of B.
    Data_type b_type_;
    // The type of the elements of C.
    Data_type c_type_;
    // The type of the accumulators.
    Data_type acc_type_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Gemm_params {

    // Ctor.
    Gemm_params() {
    }

    // The dimensions of the matrices.
    int m_, n_, k_;
    // The dimensions of the leading dimensions.
    int lda_, ldb_, ldc_, ldd_;
    // The transposition.
    bool ta_, tb_;
    // The alpha and beta values.
    double alpha_[2], beta_[2];
    // Do we use horizontal CTA rasterization?
    bool use_horizontal_cta_rasterization_;
    // The number of split-k slices.
    int split_k_slices_;
    // The number of split-k buffers (between 1 and split-k-slices).
    int split_k_buffers_;
    // Are we doing the final reduction in a separate kernel?
    int split_k_kernels_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Convolution_traits_desc {

    // The architecture.
    unsigned arch_;
    // The type of the elements of the activations.
    Data_type act_type_;
    // The type of the elements of the filters.
    Data_type flt_type_;
    // The type of the elements of the output.
    Data_type out_type_;
    // The type of the accumulators.
    Data_type acc_type_;
    // The type of the elements if bias
    Data_type bias_type_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Convolution_params {

    // Ctor.
    Convolution_params() {
        memset(this, 0, sizeof(Convolution_params));
    }

    // Compute the output dimensions.
    inline void compute_output_dimensions() {
        o_ = (d_ + 2*pad_d_ - (t_-1)*dilation_d_ - 1) / stride_d_ + 1;
        p_ = (h_ + 2*pad_h_ - (r_-1)*dilation_h_ - 1) / stride_h_ + 1;
        q_ = (w_ + 2*pad_w_ - (s_-1)*dilation_w_ - 1) / stride_w_ + 1;
    }

    // Data layouts
    xmma::Convolution_layout Layout_A, Layout_B, Layout_C;
    // The dimensions of the input activation tensor.
    int g_, n_, c_, d_, h_, w_;
    // The dimensions of the filter tensor.
    int k_, t_, r_, s_;
    // The dimensions of the output tensor.
    int o_, p_, q_;
    // The padding.
    int pad_d_, pad_h_, pad_w_;
    // The strides.
    int stride_d_, stride_h_, stride_w_;
    // The dilation.
    int dilation_d_, dilation_h_, dilation_w_;
    // The alpha and beta values.
    double alpha_[2], beta_[2];
    // Is it a cross correlation?
    bool is_cross_correlation_;
    // The activation.
    bool with_relu_;
    float relu_lb_, relu_ub_;
    // The bias.
    bool with_bias_;
    // Do we use horizontal CTA rasterization?
    bool use_horizontal_cta_rasterization_;
    // The number of split-k slices.
    int split_k_slices_;
    // The number of split-k buffers (between 1 and split-k-slices).
    int split_k_buffers_;
    // Are we doing the final reduction in a separate kernel?
    int split_k_kernels_;
    // The parameters to control how to split the TRSC dimension.
    int split_k_c_, split_k_t_, split_k_r_;

    int nhwc_pitch_c_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// C H E C K S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined SAMPLES

static inline void print_results(bool with_colors, bool enabled, bool success = false) {
    // The opening tag.
    char beg[16];
    if( with_colors && enabled && success ) {  // Succeeded -> green
        strcpy(beg, "\033[0;32m");
    } else if( with_colors && enabled ) { // Failed -> red
        strcpy(beg, "\033[0;31m");
    } else if( with_colors ) {            // Disabled -> yellow
        strcpy(beg, "\033[0;33m");
    }

    // The message.
    char msg[16];
    if( enabled && success ) {
        strcpy(msg, "SUCCESS");
    } else if( enabled ) {
        strcpy(msg, "FAILED");
    } else {
        strcpy(msg, "DISABLED");
    }

    // The closing tag.
    char end[16];
    if( with_colors ) {
        strcpy(end, "\033[0m");
    }

    // Print the results.
    if( with_colors ) {
        printf("Checks........: %s%s%s\n", beg, msg, end);
    } else {
        printf("Checks........: %s\n", msg);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int check_results(const float *out,
                                const float *ref,
                                size_t m,
                                size_t n,
                                size_t ld,
                                float epsilon,
                                bool verbose,
                                bool with_colors) {
    int failed = 0, infs = 0;
    float min_val = +FLT_MAX, max_val = -FLT_MAX, min_err = +FLT_MAX, max_err = -FLT_MAX;
    double avg_val = 0.0, sqr_val = 0.0, avg_err = 0.0, sqr_err = 0.0;
    double inv_mn = 1.0 / (double) m / (double) n;
    for( size_t ni = 0; ni < n; ++ni ) {
        for( size_t mi = 0; mi < m; ++mi ) {
            // The offset.
            size_t ii = (size_t) ni * ld + mi;

            // The elements.
            float a = out[ii];
            float b = ref[ii];

            // Compute the error.
            float den = fabsf(a) + fabsf(b);
            float err = den <= epsilon ? fabsf(a-b) : fabsf(a-b) / den;

            // Min/max values.
            min_val = fminf(a,   min_val);
            max_val = fmaxf(a,   max_val);
            min_err = fminf(err, min_err);
            max_err = fmaxf(err, max_err);

            // Sums to compute the average value.
            avg_val += (double) a         * inv_mn;
            sqr_val += (double) a * a     * inv_mn;
            avg_err += (double) err       * inv_mn;
            sqr_err += (double) err * err * inv_mn;

            // Does it fail?
            if( isnan(a) || isnan(b) || err > epsilon ) {
                if( failed < 8 ) {
                    printf("\tInvalid result for ni=%lu mi=%lu ii=%lu:\n", ni, mi, ii);
                    printf("\t    Found...: 0x%08x (%10.6f)\n", *(const int*) &out[ii], a);
                    printf("\t    Expected: 0x%08x (%10.6f)\n", *(const int*) &ref[ii], b);
                    printf("\t    Error...: %10.6f\n", err);
                }
                failed++;
            }
            infs += !isfinite(a);
            infs += !isfinite(b);
        }
    }

    double std_val = sqrtf(sqr_val - avg_val * avg_val);
    double std_err = sqrtf(sqr_err - avg_err * avg_err);

    if( verbose ) {
        printf("Epsilon.......: %.8f\n", epsilon);
        printf("Tested........: %lu\n", m*n);
        printf("Failed........: %d\n", failed);
        printf("Values........: Min=%12.6f, Max=%12.6f, Avg=%10.6lf, Std=%10.6lf\n", min_val,
                                                                                     max_val,
                                                                                     avg_val,
                                                                                     std_val);
        printf("Error.........: Min=%12.6f, Max=%12.6f, Avg=%10.6lf, Std=%10.6lf\n", min_err,
                                                                                     max_err,
                                                                                     avg_err,
                                                                                     std_err);
        printf("Epsilon.......: %.6f\n", epsilon);
        printf("Infs..........: %d\n", infs);
        print_results(with_colors, true, !failed);
    }
    return failed ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// C O N V E R S I O N   F R O M   F L O A T
//
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_bf16_from_float_(cutlass::float_bf16_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        // Decompose the float into 2 uint16.
        union { uint16_t u16[2]; float f32; } tmp;
        tmp.f32 = src[ii];

        // Decompose x into lo/hi parts.
        uint16_t lo = tmp.u16[0];
        uint16_t hi = tmp.u16[1];

        // Tweak the hi part if needed.
        if( lo == 0x8000 ) {
            hi += hi & 0x1;
        } else if( lo > 0x8000 ) {
            hi++;
        }
        dst[ii] = hi;
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_tf32_from_float_(uint32_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        uint32_t x = reinterpret_cast<const uint32_t*>(src)[ii];
        dst[ii] = x & 0xffffd000u;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_from_float_(cutlass::half_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        reinterpret_cast<__half*>(dst)[ii] = __float2half_rn(src[ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_from_float_(int32_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = (int32_t) src[ii];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_from_float_(int8_t *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        float x = src[ii];
        dst[ii] = (int8_t) (int32_t) (x < -128.f ? -128.f : (x > 127.f ? 127.f : x));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_fp64_from_float_(double *dst, const float *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = src[ii];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_from_float(void *dst, const float *src, size_t n, Data_type dtype) {
    switch( dtype ) {
    case DATA_TYPE_BF16:
        convert_bf16_from_float_(reinterpret_cast<cutlass::float_bf16_t*>(dst), src, n);
        break;
    case DATA_TYPE_TF32:
        convert_tf32_from_float_(reinterpret_cast<uint32_t*>(dst), src, n);
        break;
    case DATA_TYPE_FP32:
        memcpy(dst, src, n*sizeof(float));
        break;
    case DATA_TYPE_FP16:
        convert_from_float_(reinterpret_cast<cutlass::half_t*>(dst), src, n);
        break;
    case DATA_TYPE_INT32:
        convert_from_float_(reinterpret_cast<int32_t*>(dst), src, n);
        break;
    case DATA_TYPE_INT8:
        convert_from_float_(reinterpret_cast<int8_t*>(dst), src, n);
        break;
    case DATA_TYPE_INT8x32:
        convert_from_float_(reinterpret_cast<int8_t*>(dst), src, n);
        break;
    case DATA_TYPE_FP64:
        convert_fp64_from_float_(reinterpret_cast<double*>(dst), src, n);
        break;
    default:
        assert(false); // Not implemented!
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// C O N V E R S I O N   T O   F L O A T
//
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_bf16_to_float_(float *dst, const cutlass::float_bf16_t *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        union { uint16_t u16[2]; uint32_t u32; } tmp;
        tmp.u16[0] = uint16_t(0);
        tmp.u16[1] = src[ii];
        reinterpret_cast<uint32_t*>(dst)[ii] = tmp.u32;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_to_float_(float *dst, const cutlass::half_t *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = __half2float(reinterpret_cast<const __half*>(src)[ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_to_float_(float *dst, const int32_t *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = (float) src[ii];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_to_float_(float *dst, const int8_t *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = (float) (int32_t) src[ii];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_double_to_float_(float *dst, const double *src, size_t n) {
    for( size_t ii = 0; ii < n; ++ii ) {
        dst[ii] = src[ii];
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void convert_to_float(float *dst, const void *src, size_t n, Data_type dtype) {
    switch( dtype ) {
    case DATA_TYPE_BF16:
        convert_bf16_to_float_(dst, reinterpret_cast<const cutlass::float_bf16_t*>(dst), n);
        break;
    case DATA_TYPE_TF32:
    case DATA_TYPE_FP32:
        memcpy(dst, src, n*sizeof(float));
        break;
    case DATA_TYPE_FP16:
        convert_to_float_(dst, reinterpret_cast<const cutlass::half_t*>(src), n);
        break;
    case DATA_TYPE_INT32:
        convert_to_float_(dst, reinterpret_cast<const int32_t*>(src), n);
        break;
    case DATA_TYPE_INT8:
        convert_to_float_(dst, reinterpret_cast<const int8_t*>(src), n);
        break;
    case DATA_TYPE_INT8x32:
        convert_to_float_(dst, reinterpret_cast<const int8_t*>(src), n);
        break;
    case DATA_TYPE_FP64:
        convert_double_to_float_(dst, reinterpret_cast<const double*>(src), n);
        break;
    default:
        assert(false); // Not implemented!
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// G E M M S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Params >
static inline void gemm(float *d,
                        const float *a,
                        const float *b,
                        const float *c,
                        const Params &params) {

    // Use floats for alpha/beta.
    float alpha = (float) params.alpha_[0], beta = (float) params.beta_[0];

    // #pragma omp parallel for
    for( int ni = 0; ni < params.n_; ++ni ) {
        for( int mi = 0; mi < params.m_; ++mi ) {
            float sum = 0.f;
            for( int ki = 0; ki < params.k_; ++ki ) {

                // The offsets for A and B.
                size_t a_offset = params.ta_ ? (size_t) mi*params.lda_ + ki :
                                               (size_t) ki*params.lda_ + mi;
                size_t b_offset = params.tb_ ? (size_t) ki*params.ldb_ + ni :
                                               (size_t) ni*params.ldb_ + ki;

                // Read the elements.
                float x = a[a_offset];
                float y = b[b_offset];

                // Update the sum.
                sum += x * y;
//                if (ni==0 && mi==0) {
//                    printf("ki=%d y=%1.0f %1.0f\n",ki,x,sum);
//                }
            } // ki

            // Update the result.
            float z = 0.f;
            if( beta != 0.f ) {
                z = c[(size_t) ni*params.ldc_ + mi];
            }
            d[(size_t) ni*params.ldd_ + mi] = alpha * sum + beta * z;
        } // mi
    } // ni
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// C O N V O L U T I O N S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Identity_functor {
    inline float operator()(int, int, int, int, int, float val) const {
        return val;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fprop_bias_functor {
    // Ctor.
    Fprop_bias_functor(const float *bias) : bias_(bias) {
    }

    // Add the bias and apply relu.
    inline float operator()(int, int, int, int, int ki, float val) const {
        return val + bias_[ki];
    }

    // The bias tensor.
    const float *bias_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fprop_bias_and_relu_functor : public Fprop_bias_functor {
    // Ctor.
    Fprop_bias_and_relu_functor(const float *bias) : Fprop_bias_functor(bias) {
    }

    // Add the bias and apply relu.
    inline float operator()(int ni, int oi, int pi, int qi, int ki, float val) const {
        return fmaxf(Fprop_bias_functor::operator()(ni, oi, pi, qi, ki, val), 0.f);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename Params >
static inline void print_converted_inputs(const float* act, const float* flt, const Params &params) {

    float* converted_act = (float*) malloc(params.n_ * params.c_ * params.h_ * params.h_);
    float* converted_flt = (float*) malloc(params.k_ * params.c_ * params.r_ * params.s_);


    for (int ni=0; ni < params.n_; ni++) {
        for (int ci=0; ci < params.c_ / 32; ci++) {
            for (int hi=0; hi < params.h_;  hi++) {
                for (int wi=0; wi < params.w_;  wi++) {
                    for (int i=0; i < 32;  i++) {

                        int act_idx = ni * params.c_  * params.h_ * params.w_ +
                                      ci * params.h_ * params.w_ * 32 +
                                      hi * params.w_ * 32 +
                                      wi * 32 +
                                      i;

                        int converted_act_idx = ni * params.h_ * params.w_ * params.c_ +
                                                hi * params.w_ * params.c_;
                                                wi * params.c_ +
                                                ci * 32 + i;

                        converted_act[converted_act_idx] = act[act_idx];

                    }
                }
            }
        }
    }

    for (int ni=0; ni < params.n_; ni++) {
        for (int hi=0; hi < params.h_;  hi++) {
            for (int wi=0; wi < params.w_;  wi++) {
                for (int ci=0; ci < params.c_; ci++) {
                    int idx = ni * params.h_ * params.w_ * params.c_ +
                              hi * params.w_ * params.c_;
                              wi * params.c_ +
                              ci;

                    printf("%f ", converted_act[idx]);
                }
                printf("\n");
            }
        }
    }


    for (int ki=0; ki < params.k_; ki++) {
        for (int ci=0; ci < params.c_ / 32; ci++) {
            for (int ri=0; ri < params.r_;  ri++) {
                for (int si=0; si < params.s_;  si++) {
                    for (int i=0; i < 32;  i++) {

                        int flt_idx = ki * params.c_  * params.r_ * params.s_ +
                                      ci * params.r_ * params.s_ * 32 +
                                      ri * params.s_ * 32 +
                                      si * 32 +
                                      i;

                        int converted_flt_idx = ki * params.r_ * params.s_ * params.c_ +
                                                ri * params.s_ * params.c_;
                                                si * params.c_ +
                                                ci * 32 + i;

                        converted_flt[converted_flt_idx] = flt[flt_idx];
                    }
                }
            }
        }
    }

    for (int ki=0; ki < params.k_; ki++) {
        for (int ri=0; ri < params.r_;  ri++) {
            for (int si=0; si < params.s_;  si++) {
                for (int ci=0; ci < params.c_; ci++) {
                    int idx = ki * params.r_ * params.s_ * params.c_ +
                              ri * params.s_ * params.c_;
                              si * params.c_ +
                              ci;

                    printf("%f ", converted_flt[idx]);
                }
                printf("\n");
            }
        }
    }

    free(converted_act);
    free(converted_flt);
}

template< typename Params >
static inline void fprop_interleaved(float *out,
                               const float *act,
                               const float *flt,
                               const Params &params) {


    print_converted_inputs(act, flt, params);

    // Use floats for alpha/beta.
    float alpha = (float) params.alpha_[0], beta = (float) params.beta_[0];

    #pragma omp parallel for
    for( int ni = 0; ni < params.n_; ++ni ) {
        for( int pi = 0; pi < params.p_; ++pi ) {
            for( int qi = 0; qi < params.q_; ++qi ) {
                for( int ki = 0; ki < params.k_; ++ki ) {
                    float sum = 0.f;
                    for( int ri = 0; ri < params.r_; ++ri ) {
                        for( int si = 0; si < params.s_; ++si ) {
                            for( int ci = 0; ci < params.c_ ; ++ci ) {

                                // The filter shift.
                                int rj = ri;
                                int sj = si;

                                // Deal with convolution.
                                if( !params.is_cross_correlation_ ) {
                                    rj = params.r_ - ri - 1;
                                    sj = params.s_ - si - 1;
                                }

                                // The coordinates of the pixel in the image.
                                int hi = pi * params.stride_h_ + rj * params.dilation_h_ - params.pad_h_;
                                int wi = qi*params.stride_w_ + sj*params.dilation_w_ - params.pad_w_;

                                // Is the pixel in the image?
                                bool is_in_image = (unsigned) hi < (unsigned) params.h_ &&
                                                   (unsigned) wi < (unsigned) params.w_;

                                // Deal with convolution.
                                if( !params.is_cross_correlation_ ) {
                                    rj = params.r_ - ri - 1;
                                    sj = params.s_ - si - 1;
                                }

                                // The offsets. (assuming NC/32HW32)
                                int act_offset = (size_t) ni * params.h_ * params.w_ * params.c_ +
                                                    int(ci / 32) * params.h_ * params.w_ * 32 +
                                                    hi * params.w_ * 32 +
                                                    wi * 32 +
                                                    ci % 32;

                                size_t flt_offset = (size_t) ki * params.r_ * params.s_ * params.c_ +
                                                    int(ci / 32) * params.r_ * params.s_ * 32 +
                                                    ri * params.s_ * 32 +
                                                    si * 32 +
                                                    ci % 32;

                                //printf("n, p, q, k, r, s, c = (%d %d %d %d %d %d %d) ; act offset = %d \n", ni, pi, qi, ki, ri, si, ci, act_offset);

                                // The two values.
                                float a = is_in_image ? act[act_offset] : 0.f;
                                float b = is_in_image ? flt[flt_offset] : 0.f;

                                // Update the output value.
                                sum += a * b;

                            } // ci
                        } // si
                    } // ri

                    // Store the output value for real.
                    size_t out_offset = (size_t) ni * params.p_ * params.q_ * params.k_ +
                                        int(ki / 32) * params.p_ * params.q_ * 32 +
                                        pi * params.q_ * 32 +
                                        qi * 32 +
                                        ki % 32;

                    // Update the value.
                    float val;
                    if( beta != 0.f ) {
                        val = alpha * sum + beta * out[out_offset];
                    } else {
                        val = alpha * sum;
                    }

                    // Store the value.
                    out[out_offset] = val;

                } // ki
            } // qi
        } // pi
    } // ni
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Params, typename Functor = Identity_functor >
static inline void fprop_ndhwc(float *out,
                               const float *act,
                               const float *flt,
                               const Params &params,
                               const bool is_img_ncdhw = false,
                               const bool is_out_ncdhw = false,
                               const Functor &fct = Functor(),
                               Data_type dtype_out = DATA_TYPE_FP32) {

    // Use floats for alpha/beta.
    float alpha = (float) params.alpha_[0], beta = (float) params.beta_[0];

    #pragma omp parallel for
    for( int ni = 0; ni < params.n_; ++ni ) {
    for( int gi = 0; gi < params.g_; ++gi ) {
        for( int oi = 0; oi < params.o_; ++oi ) {
            for( int pi = 0; pi < params.p_; ++pi ) {
                for( int qi = 0; qi < params.q_; ++qi ) {
                    for( int ki = 0; ki < (params.k_ / params.g_); ++ki ) {
                        float sum = 0.f;
                        for( int ti = 0; ti < params.t_; ++ti ) {
                            for( int ri = 0; ri < params.r_; ++ri ) {
                                for( int si = 0; si < params.s_; ++si ) {
                                    for( int ci = 0; ci < params.c_ / params.g_; ++ci ) {

                                        // The filter shift.
                                        int tj = ti;
                                        int rj = ri;
                                        int sj = si;

                                        // Deal with convolution.
                                        if( !params.is_cross_correlation_ ) {
                                            tj = params.t_ - ti - 1;
                                            rj = params.r_ - ri - 1;
                                            sj = params.s_ - si - 1;
                                        }

                                        // The coordinates of the pixel in the image.
                                        int di = oi*params.stride_d_ + tj*params.dilation_d_ -
                                                 params.pad_d_;
                                        int hi = pi*params.stride_h_ + rj*params.dilation_h_ -
                                                 params.pad_h_;
                                        int wi = qi*params.stride_w_ + sj*params.dilation_w_ -
                                                 params.pad_w_;

                                        // Is the pixel in the image?
                                        bool is_in_image = (unsigned) di < (unsigned) params.d_ &&
                                                           (unsigned) hi < (unsigned) params.h_ &&
                                                           (unsigned) wi < (unsigned) params.w_;

                                        // The offsets.
                                        size_t act_offset = is_img_ncdhw
                                            ? (size_t) ni*params.d_*params.h_*params.w_*params.c_ +
                                                       (gi * (params.c_ / params.g_) + ci)*params.d_*params.h_*params.w_ +
                                                       di*params.h_*params.w_ +
                                                       hi*params.w_ +
                                                       wi
                                            : (size_t) ni*params.d_*params.h_*params.w_*params.c_ +
                                                       di*params.h_*params.w_*params.c_ +
                                                       hi*params.w_*params.c_ +
                                                       wi*params.c_ +
                                                       (gi * (params.c_ / params.g_) + ci);
                                        size_t flt_offset =
                                            (size_t) gi * (params.k_ / params.g_)*params.t_*params.r_*params.s_*(params.c_ / params.g_) +
                                                     ki*params.t_*params.r_*params.s_*(params.c_ / params.g_) +
                                                     ti*params.r_*params.s_*(params.c_ / params.g_) +
                                                     ri*params.s_*(params.c_ / params.g_) +
                                                     si*(params.c_ / params.g_) +
                                                     ci;

                                        // The two values.
                                        float a = is_in_image ? act[act_offset] : 0.f;
                                        float b = is_in_image ? flt[flt_offset] : 0.f;

                                        // Update the output value.
                                        sum += a*b;
                                    } // ci
                                } // si
                            } // ri
                        } // ti

                        // Store the output value for real.
                        size_t out_offset = is_out_ncdhw
                            ? (size_t) ni*params.o_*params.p_*params.q_*params.k_ +
                                       (gi * (params.k_ / params.g_) + ki)*params.o_*params.p_*params.q_ +
                                       oi*params.p_*params.q_ +
                                       pi*params.q_ +
                                       qi
                            : (size_t) ni*params.o_*params.p_*params.q_*params.k_ +
                                       oi*params.p_*params.q_*params.k_ +
                                       pi*params.q_*params.k_ +
                                       qi*params.k_ +
                                       (gi * (params.k_ / params.g_) + ki);

                        // Update the value.
                        float val;
                        if( beta != 0.f ) {
                            val = alpha * sum + beta * out[out_offset];
                        } else {
                            val = alpha * sum;
                        }

                        if( dtype_out == DATA_TYPE_INT8) 
                        { 
                            if( val > 128 ) {
                                val = 127;
                            } else if( val < -128) {
                                val = -128;
                            }
                        }

                        // Store the value.
                        out[out_offset] = fct(ni, oi, pi, qi, ki, val);
                    } // ki
                } // qi
            } // pi
        } // oi
    } // gi
    } // ni
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Params, typename Functor = Identity_functor >
static inline void dgrad_ndhwc(float *act,
                               const float *out,
                               const float *flt,
                               const Params &params,
                               const bool is_img_ncdhw = false,
                               const bool is_out_ncdhw = false,
                               const Functor &fct = Functor()) {

    // Use floats for alpha/beta.
    float alpha = (float) params.alpha_[0], beta = (float) params.beta_[0];

    // Clear the output as we perform scattered writes.
    size_t ndhwc = (size_t) params.n_ * params.d_ * params.h_ * params.w_ * params.c_;
    for( size_t ii = 0; ii < ndhwc; ++ii ) {
        act[ii] = beta != 0.f ? beta * act[ii] : 0.f;
    }

    // Do dgrad...
    for( int ni = 0; ni < params.n_; ++ni ) {
        for( int oi = 0; oi < params.o_; ++oi ) {
            for( int pi = 0; pi < params.p_; ++pi ) {
                for( int qi = 0; qi < params.q_; ++qi ) {
                    for( int ki = 0; ki < params.k_; ++ki ) {
                        for( int ti = 0; ti < params.t_; ++ti ) {
                            for( int ri = 0; ri < params.r_; ++ri ) {
                                for( int si = 0; si < params.s_; ++si ) {
                                    for( int ci = 0; ci < params.c_; ++ci ) {

                                        // The filter shift.
                                        int tj = ti;
                                        int rj = ri;
                                        int sj = si;

                                        // Deal with convolution.
                                        if( !params.is_cross_correlation_ ) {
                                            tj = params.t_ - ti - 1;
                                            rj = params.r_ - ri - 1;
                                            sj = params.s_ - si - 1;
                                        }

                                        // The coordinates of the pixel in the image.
                                        int di = oi*params.stride_d_ + tj*params.dilation_d_ -
                                                 params.pad_d_;
                                        int hi = pi*params.stride_h_ + rj*params.dilation_h_ -
                                                 params.pad_h_;
                                        int wi = qi*params.stride_w_ + sj*params.dilation_w_ -
                                                 params.pad_w_;

                                        // Is the pixel in the image?
                                        bool is_in_image = (unsigned) di < params.d_ &&
                                                           (unsigned) hi < params.h_ &&
                                                           (unsigned) wi < params.w_;

                                        // The input offsets.
                                        size_t out_offset = is_img_ncdhw
                                            ? (size_t) ni*params.o_*params.p_*params.q_*params.k_ +
                                                       ki*params.o_*params.p_*params.q_ +
                                                       oi*params.p_*params.q_ +
                                                       pi*params.q_ +
                                                       qi
                                            : (size_t) ni*params.o_*params.p_*params.q_*params.k_ +
                                                       oi*params.p_*params.q_*params.k_ +
                                                       pi*params.q_*params.k_ +
                                                       qi*params.k_ +
                                                       ki;
                                        size_t flt_offset =
                                            (size_t) ki*params.t_*params.r_*params.s_*params.c_ +
                                                     ti*params.r_*params.s_*params.c_ +
                                                     ri*params.s_*params.c_ +
                                                     si*params.c_ +
                                                     ci;

                                        // The two values.
                                        float a = is_in_image ? out[out_offset] : 0.f;
                                        float b = is_in_image ? flt[flt_offset] : 0.f;

                                        // The destination offset.
                                        size_t act_offset = is_out_ncdhw
                                             ? (size_t) ni*params.d_*params.h_*params.w_*params.c_ +
                                                     ci*params.d_*params.h_*params.w_ +
                                                     di*params.h_*params.w_ +
                                                     hi*params.w_ +
                                                     wi
                                             : (size_t) ni*params.d_*params.h_*params.w_*params.c_ +
                                                     di*params.h_*params.w_*params.c_ +
                                                     hi*params.w_*params.c_ +
                                                     wi*params.c_ +
                                                     ci;

                                        // Update the gradient of the pixel.
                                        act[act_offset] += alpha * (a * b);

                                    } // ci
                                } // si
                            } // ri
                        } // ti
                    } // ki
                } // qi
            } // pi
        } // oi
    } // ni

    // Apply the functor.
    for( int ni = 0; ni < params.n_; ++ni ) {
        for( int di = 0; di < params.d_; ++di ) {
            for( int hi = 0; hi < params.h_; ++hi ) {
                for( int wi = 0; wi < params.w_; ++wi ) {
                    for( int ci = 0; ci < params.c_; ++ci ) {
                        size_t act_offset = is_out_ncdhw
                            ? (size_t) ni*params.d_*params.h_*params.w_*params.c_ +
                                     ci*params.d_*params.h_*params.w_ +
                                     di*params.h_*params.w_ +
                                     hi*params.w_ +
                                     wi
                            : (size_t) ni*params.d_*params.h_*params.w_*params.c_ +
                                     di*params.h_*params.w_*params.c_ +
                                     hi*params.w_*params.c_ +
                                     wi*params.c_ +
                                     ci;
                        act[act_offset] = fct(ni, di, hi, wi, ci, act[act_offset]);
                    } // ci
                } // wi
            } // hi
        } // di
    } // ni
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Params, typename Functor = Identity_functor >
static inline void wgrad_ndhwc(float *flt,
                               const float *act,
                               const float *out,
                               const Params &params,
                               const bool is_img_ncdhw = false,
                               const bool is_out_ncdhw = false,
                               const Functor &fct = Functor()) {

    // Use floats for alpha/beta.
    float alpha = (float) params.alpha_[0], beta = (float) params.beta_[0];

    #pragma omp parallel for
    for( int ki = 0; ki < params.k_; ++ki ) {
        for( int ti = 0; ti < params.t_; ++ti ) {
            for( int ri = 0; ri < params.r_; ++ri ) {
                for( int si = 0; si < params.s_; ++si ) {
                    for( int ci = 0; ci < params.c_; ++ci ) {
                        float sum = 0.f;
                        for( int ni = 0; ni < params.n_; ++ni ) {
                            for( int oi = 0; oi < params.o_; ++oi ) {
                                for( int pi = 0; pi < params.p_; ++pi ) {
                                    for( int qi = 0; qi < params.q_; ++qi ) {

                                        // The filter shift.
                                        int tj = ti;
                                        int rj = ri;
                                        int sj = si;

                                        // Deal with convolution.
                                        if( !params.is_cross_correlation_ ) {
                                            tj = params.t_ - ti - 1;
                                            rj = params.r_ - ri - 1;
                                            sj = params.s_ - si - 1;
                                        }

                                        // The coordinates of the pixel in the image.
                                        int di = oi*params.stride_d_ + tj*params.dilation_d_ -
                                                 params.pad_d_;
                                        int hi = pi*params.stride_h_ + rj*params.dilation_h_ -
                                                 params.pad_h_;
                                        int wi = qi*params.stride_w_ + sj*params.dilation_w_ -
                                                 params.pad_w_;

                                        // Is the pixel in the image?
                                        bool is_in_image = (unsigned) di < params.d_ &&
                                                           (unsigned) hi < params.h_ &&
                                                           (unsigned) wi < params.w_;

                                        // The offsets.
                                        size_t act_offset = is_img_ncdhw
                                             ? (size_t) ni*params.d_*params.h_*params.w_*params.c_ +
                                                     ci*params.d_*params.h_*params.w_ +
                                                     di*params.h_*params.w_ +
                                                     hi*params.w_ +
                                                     wi
                                             : (size_t) ni*params.d_*params.h_*params.w_*params.c_ +
                                                     di*params.h_*params.w_*params.c_ +
                                                     hi*params.w_*params.c_ +
                                                     wi*params.c_ +
                                                     ci;

                                        size_t out_offset = is_out_ncdhw
                                            ? (size_t) ni*params.o_*params.p_*params.q_*params.k_ +
                                                       ki*params.o_*params.p_*params.q_ +
                                                       oi*params.p_*params.q_ +
                                                       pi*params.q_ +
                                                       qi
                                            : (size_t) ni*params.o_*params.p_*params.q_*params.k_ +
                                                       oi*params.p_*params.q_*params.k_ +
                                                       pi*params.q_*params.k_ +
                                                       qi*params.k_ +
                                                       ki;

                                        // The two values.
                                        float a = is_in_image ? act[act_offset] : 0.f;
                                        float b = is_in_image ? out[out_offset] : 0.f;

                                        // Update the output value.
                                        sum += a*b;
                                    } // qi
                                } // pi
                            } // oi
                        } // ni

                        // Store the output value for real.
                        size_t flt_offset =
                            (size_t) ki*params.t_*params.r_*params.s_*params.c_ +
                                     ti*params.r_*params.s_*params.c_ +
                                     ri*params.s_*params.c_ +
                                     si*params.c_ +
                                     ci;

                        // Update the value.
                        float val;
                        if( beta != 0.f ) {
                            val = alpha * sum + beta * flt[flt_offset];
                        } else {
                            val = alpha * sum;
                        }

                        // Apply the functor.
                        flt[flt_offset] = fct(ki, ti, ri, si, ci, val);
                    } // ci
                } // si
            } // ri
        } // ti
    } // ki
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// D A T A   T Y P E   F R O M / T O   A   S T R I N G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline Data_type data_type_from_string(const char *name) {
    if( !strcmp(name, "tf32") ) {
        return DATA_TYPE_TF32;
    } else if( !strcmp(name, "bf16") ) {
        return DATA_TYPE_BF16;
    } else if( !strcmp(name, "fp32") ) {
        return DATA_TYPE_FP32;
    } else if( !strcmp(name, "fp16") ) {
        return DATA_TYPE_FP16;
    } else if( !strcmp(name, "int32") ) {
        return DATA_TYPE_INT32;
    } else if( !strcmp(name, "int8") ) {
        return DATA_TYPE_INT8;
    } else if( !strcmp(name, "int4") ) {
        return DATA_TYPE_INT4;
    } else if( !strcmp(name, "bool") ) {
        return DATA_TYPE_BOOL;
    } else if( !strcmp(name, "fp64") ) {
        return DATA_TYPE_FP64;
    } else {
        assert(false);
        return DATA_TYPE_FP32;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline const char* data_type_to_string(Data_type dtype) {
    switch( dtype ) {
    case DATA_TYPE_BF16:
        return "bf16";
    case DATA_TYPE_TF32:
        return "tf32";
    case DATA_TYPE_FP32:
        return "fp32";
    case DATA_TYPE_FP16:
        return "fp16";
    case DATA_TYPE_INT32:
        return "int32";
    case DATA_TYPE_INT8:
        return "int8";
    case DATA_TYPE_INT4:
        return "int4";
    case DATA_TYPE_BOOL:
        return "bool";
    case DATA_TYPE_FP64:
        return "fp64";
    default:
        assert(false);
        return "unknown";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// T R A N S P O S E S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void ncdhw_to_ndhwc(float *dst, const float *src, int n, int c, int d, int h, int w) {
    for( int ni = 0; ni < n; ++ni ) {
        for( int ci = 0; ci < c; ++ci ) {
            for( int di = 0; di < d; ++di ) {
                for( int hi = 0; hi < h; ++hi ) {
                    for( int wi = 0; wi < w; ++wi ) {
                        size_t src_offset = (size_t) ni*c*d*h*w + ci*d*h*w + di*h*w + hi*w + wi;
                        size_t dst_offset = (size_t) ni*d*h*w*c + di*h*w*c + hi*w*c + wi*c + ci;
                        dst[dst_offset] = src[src_offset];
                    } // wi
                } // hi
            } // di
        } // ci
    } // ni
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void ndhwc_to_ncdhw(float *dst, const float *src, int n, int d, int h, int w, int c) {
    for( int ni = 0; ni < n; ++ni ) {
        for( int di = 0; di < d; ++di ) {
            for( int hi = 0; hi < h; ++hi ) {
                for( int wi = 0; wi < w; ++wi ) {
                    for( int ci = 0; ci < c; ++ci ) {
                        size_t src_offset = (size_t) ni*d*h*w*c + di*h*w*c + hi*w*c + wi*c + ci;
                        size_t dst_offset = (size_t) ni*c*d*h*w + ci*d*h*w + di*h*w + hi*w + wi;
                        dst[dst_offset] = src[src_offset];
                    } // ci
                } // wi
            } // hi
        } // di
    } // ni
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// R A N D O M   I N I T I A L I Z A T I O N
//
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void random_init(float *dst,
                               size_t n,
                               int range,
                               float scale,
                               bool use_1s,
                               bool verbose) {
    if( verbose ) {
        printf("Range.........: %d\n", range);
        printf("Scale.........: %f\n", scale);
        printf("Use.1s........: %s\n", use_1s ? "true" : "false");
        printf("Address.......: 0x%016lx\n", (size_t) dst);
        printf("Values........: ");
    }
    for( size_t ii = 0; ii < n; ++ii ) {
        float x = 1.f;
        if( !use_1s ) {
            x = (float) (rand() % range - range / 2) * scale;
            //x = (float) (ii % 32);
        }
        if( verbose && ii < 8 ) {
            printf("%.3f ", x);
        }
        dst[ii] = x;
    }
    if( verbose ) {
        printf("...\n");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S I Z E   O F   A   D A T A   T Y P E
//
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline size_t get_size_in_bytes(size_t n, Data_type dtype) {
    switch( dtype ) {
    case DATA_TYPE_FP64:
        return n * 8;
    case DATA_TYPE_TF32:
        return n * 4;
    case DATA_TYPE_FP32:
        return n * 4;
    case DATA_TYPE_FP16:
        return n * 2;
    case DATA_TYPE_INT32:
        return n * 4;
    case DATA_TYPE_INT8:
        return n;
    case DATA_TYPE_INT4:
        return n / 2;
    case DATA_TYPE_BOOL:
        return n / 8;
    case DATA_TYPE_BF16:
        return n * 2;
    case DATA_TYPE_INT8x32:
        return n;
    default:
        assert(false);
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int elements_per_ldg(int size_in_bytes, Data_type dtype) {
    switch( dtype ) {
    case DATA_TYPE_FP64:
        return size_in_bytes / 8;
    case DATA_TYPE_TF32:
        return size_in_bytes / 4;
    case DATA_TYPE_BF16:
        return size_in_bytes / 2;
    case DATA_TYPE_FP32:
        return size_in_bytes / 4;
    case DATA_TYPE_FP16:
        return size_in_bytes / 2;
    case DATA_TYPE_INT32:
        return size_in_bytes / 4;
    case DATA_TYPE_INT8:
        return size_in_bytes;
    case DATA_TYPE_INT4:
        return size_in_bytes * 2;
    case DATA_TYPE_BOOL:
        return size_in_bytes * 8;
    default:
        assert(false);
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// C U D A   C O P I E S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

static inline cudaError cuda_memcpy_h2d(void *dst, const float *src, size_t n, Data_type dtype) {
    size_t sz = get_size_in_bytes(n, dtype);
    void *tmp = malloc(sz);
    convert_from_float(tmp, src, n, dtype);
    cudaError err = cudaMemcpy(dst, tmp, sz, cudaMemcpyHostToDevice);
    free(tmp);
    return err;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline cudaError cuda_memcpy_d2h(float *dst, const void *src, size_t n, Data_type dtype) {
    size_t sz = get_size_in_bytes(n, dtype);
    void *tmp = malloc(sz);
    cudaError err = cudaMemcpy(tmp, src, sz, cudaMemcpyDeviceToHost);
    convert_to_float(dst, tmp, n, dtype);
    free(tmp);
    return err;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline cudaError cuda_memcpy_to_ncdhw_h2d(void *dst,
                                                 const float *src,
                                                 int n,
                                                 int d,
                                                 int h,
                                                 int w,
                                                 int c,
                                                 Data_type dtype) {
    size_t ndhwc = (size_t) n * d * h * w * c;
    float *tmp = (float*) malloc(ndhwc * sizeof(float));
    ndhwc_to_ncdhw(tmp, src, n, d, h, w, c);
    cudaError err = cuda_memcpy_h2d(dst, tmp, ndhwc, dtype);
    free(tmp);
    return err;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // SAMPLES
