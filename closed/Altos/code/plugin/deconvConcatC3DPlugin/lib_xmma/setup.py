# ***************************************************************************************************
# * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
# *
# * Redistribution and use in source and binary forms, with or without modification, are not permit-
# * ted.
# *
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
# **************************************************************************************************/

import os
import shutil
import sys

####################################################################################################

license_header = """
%(comment)s **************************************************************************************************
%(comment)s  Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
%(comment)s
%(comment)s  Redistribution and use in source and binary forms, with or without modification, are not permit-
%(comment)s  ted.
%(comment)s
%(comment)s  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
%(comment)s  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
%(comment)s  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
%(comment)s  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
%(comment)s  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
%(comment)s  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
%(comment)s  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%(comment)s  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%(comment)s
%(comment)s **************************************************************************************************
 """

####################################################################################################

def get_license_header(comment):
    return license_header % locals()

####################################################################################################

def generate_separation_line(file):
    file.write("\n")
    file.write("/" * 100)
    file.write("\n")

####################################################################################################

def get_dispatch_name(sample_name, mode, flags, ta, tc):
    name = sample_name + "_" + mode
    if flags:
        name += "_" + flags
    name += "_a" + ta
    name += "_c" + tc
    return name

####################################################################################################

def get_sm_version(arch):
    arch_to_sm = {
        "Volta" : 70,
        "Turing": 75,
        "Ampere": 80,
    }
    return arch_to_sm[arch]

####################################################################################################

def get_kernel_name(dispatch_name, arch, inst, args, align_a, align_c, m, n, k, stages):
    traits = "%s_%s" % (arch.lower(), inst)
    if args:
        traits += "_" + "_".join(args)
    if align_a:
        traits += "_a%s" % align_a
    if align_c:
        traits += "_c%s" % align_c
    kernel_name = "%s_%s_%03d_%03d_%03d_%d" % (dispatch_name, traits, m, n, k, stages)
    return kernel_name, get_sm_version(arch)

####################################################################################################

def get_arg_from_name(name):
    arg_from_name = {
        "tf32" : "cutlass::float_tf32_t",
        "bf16" : "cutlass::float_bf16_t",
        "fp32" : "float",
    }
    return arg_from_name[name]

####################################################################################################

def get_traits(arch, inst, args):
    traits = "%s_%s_traits" % (arch, inst)
    if args:
        traits += "<" + ", ".join([get_arg_from_name(arg) for arg in args]) + ">"
    return traits

def get_alignment_c(inst, align_c):
    if align_c:
        return align_c
    else:
        return 16

def get_alignment_a(inst, align_a):
    if align_a:
        return align_a
    else:
        return 16

def skip_meaningless_cfg(mode, inst, align_a, align_c, ta, tc):
    if tc in ["t"] and align_c in [4]:
        return 1
    if mode in "wgrad_indexed":
        if ta in ["n"] and align_a in [4]:
            return 1
    else:
        if ta in ["t"] and align_a in [4]:
            return 1

    if mode in "wgrad_indexed":
        if ta in ["t"] and inst not in ["hmma_tf32"]:
            return 1
    else:
        if ta in ["n"] and inst not in ["hmma_tf32"]:
            return 1

    if tc in ["n"] and inst not in ["hmma_tf32"]:
        return 1
    return 0

####################################################################################################

def generate_kernel_file(sample_name, mode, flags, ta, tc, arch, inst, args, align_a, align_c, m, n, k, stages):
    dispatch_name = get_dispatch_name(sample_name, mode, flags, ta, tc)
    kernel_name, sm = get_kernel_name(dispatch_name, arch, inst, args, align_a, align_c, m, n, k, stages)
    with open(os.path.join("generated", "%s.sm%d.cu" % (kernel_name, sm)), "w") as file:
        file.write(get_license_header("//"))
        file.write("\n")
        file.write("// !!! This file was generated -- do not edit of changes will be lost !!!\n")

        # The includes.
        arch_lower = arch.lower()
        file.write("#include <xmma/params.h>")
        file.write("""
#include <%(sample_name)s.h>
#include <limits.h>
""" % locals())

        # TODO: Fixme - we should not have to include all archs for wgrad.
        archs = ["volta", "turing", "ampere"] if "wgrad" in mode else [arch_lower]
        for included_arch in archs:
            file.write("""
#include <xmma/%(included_arch)s/traits.h>
#include <xmma/%(included_arch)s/fragment.h>
#include <xmma/%(included_arch)s/smem_tile.h>
""" % locals())
        file.write("""
#include <xmma/ampere/gmem_wo_smem_tile.h>
#include <xmma/gemm/kernel.h>
#include <xmma/implicit_gemm/%(mode)s/traits.h>
#include <xmma/implicit_gemm/%(mode)s/params.h>
#include <xmma/implicit_gemm/host_runtime.h>
""" % locals())

        # Include the utils if they exist.
        if mode not in ["fprop_indexed", "dgrad_indexed"]:
            file.write("#include <xmma/implicit_gemm/%s/utils.h>\n" % mode)

        # For Input_related
        if mode in ["fprop", "fprop_indexed", "dgrad", "dgrad_indexed"]:
            file.write("#include <xmma/implicit_gemm/utils.h>\n")

        # Add a line of '/'.
        generate_separation_line(file)

        # Declare the useful types.
        traits = get_traits(arch, inst, args)

        # Special mode for wgrad.
        if "wgrad" in mode:
            kernel_traits_args = "Traits, Cta_tile, Gmem_tile_a, Gmem_tile_b, "
        else:
            kernel_traits_args = "Traits, Cta_tile, Gmem_tile_a, Gmem_tile_c, "

        # For Input_related
        if mode in ["fprop", "fprop_indexed", "dgrad", "dgrad_indexed"]:
            # TODO : find a way to support tests of template filter size kernels.
            kernel_traits_args += "typename xmma::implicit_gemm::Input_related<0, 0, 0, false>, "
        if "wgrad" in mode and "simple_1x1x1" in flags:
            kernel_traits_args += "true, "
        elif "wgrad" in mode:
            kernel_traits_args += "false, "
        file.write("""
// The instruction traits.
using Traits = xmma::%(traits)s;
// The CTA tile.
using Cta_tile = typename Traits::Cta_tile<%(m)d, %(n)d, %(k)d>;
""" % locals())

        if "wgrad" not in mode:
            file.write("""
// The global memory tile for Epilogue.
using Gmem_tile_a = xmma::implicit_gemm::%(mode)s::Gmem_tile_a_%(ta)s<Traits,
                                                  Cta_tile,
                                                  typename xmma::implicit_gemm::Input_related<0, 0, 0, false>,
                                                  %(align_a)d>;
// The global memory tile for Epilogue.
using Gmem_tile_c = xmma::implicit_gemm::%(mode)s::Gmem_tile_c_%(tc)s<Traits,
                                                  Cta_tile, %(align_c)d>;
""" % locals())

        if "wgrad" in mode:
            file.write("""
// The global memory tile for Epilogue.
using Gmem_tile_a = xmma::implicit_gemm::%(mode)s::Gmem_tile_a_%(ta)s<Traits,
                                                  Cta_tile,
                                                  %(align_a)d>;
""" % locals())
        if "wgrad" in mode and "simple_1x1x1" in flags:
            file.write("""
// The global memory tile for Epilogue.
using Gmem_tile_b = xmma::implicit_gemm::%(mode)s::Gmem_tile_b_%(tc)s<Traits,
                                                  Cta_tile,
                                                  true,
                                                  %(align_c)d>;
""" % locals())
        elif "wgrad" in mode:
            file.write("""
// The global memory tile for Epilogue.
using Gmem_tile_b = xmma::implicit_gemm::%(mode)s::Gmem_tile_b_%(tc)s<Traits,
                                                  Cta_tile,
                                                  false,
                                                  %(align_c)d>;
""" % locals())
        kernel_traits_args += str(stages)

        file.write("""
// The kernel traits.
using Kernel_traits = xmma::implicit_gemm::%(mode)s::Kernel_traits<%(kernel_traits_args)s>;
// The host workspace.
using Host_workspace = xmma::Host_workspace<Kernel_traits>;
""" % locals())

        # Add a line of '/'.
        generate_separation_line(file)

        # Generate the function to compute the host size.
        file.write("""
extern "C" int %(kernel_name)s_compute_host_workspace_size(
    size_t *size_in_bytes) {
    size_in_bytes[0] = xmma::implicit_gemm::get_host_workspace_size<Kernel_traits>();
    return 0;
}
""" % locals())

        # Add a line of '/'.
        generate_separation_line(file)

        # Generate the function to initialize the host parameters.
        cta_raster = "1" if "wgrad_indexed" in mode else "params->use_horizontal_cta_rasterization_"
        # The Ampere flag. TODO: Remove that!
        ampere = 1 if arch == "Ampere" else 0

        file.write("""
extern "C" int %(kernel_name)s_initialize_host_workspace(
    void *host_ptr,
    const Convolution_params *params) {

    // The host workspace.
    Host_workspace *host_workspace = static_cast<Host_workspace*>(host_ptr);

    // Reset the XMMA parameters.
    memset(&host_workspace->xmma_params, 0, sizeof(host_workspace->xmma_params));

    // Is it Ampere?
    host_workspace->xmma_params.ampere = %(ampere)d;

    // Set the problem size.
    host_workspace->xmma_params.n = params->n_;
    host_workspace->xmma_params.g = 1;
    host_workspace->xmma_params.c = params->c_;
    host_workspace->xmma_params.d = params->d_;
    host_workspace->xmma_params.h = params->h_;
    host_workspace->xmma_params.w = params->w_;
    host_workspace->xmma_params.k = params->k_;
    host_workspace->xmma_params.t = params->t_;
    host_workspace->xmma_params.r = params->r_;
    host_workspace->xmma_params.s = params->s_;
    host_workspace->xmma_params.o = params->o_;
    host_workspace->xmma_params.p = params->p_;
    host_workspace->xmma_params.q = params->q_;

    // Set the tensor strides.
    if (params->Layout_A == xmma::Convolution_layout::NHWC) {
    host_workspace->xmma_params.img_stride_c = 1;
    host_workspace->xmma_params.img_stride_w =
        host_workspace->xmma_params.img_stride_c * params->nhwc_pitch_c_;
    host_workspace->xmma_params.img_stride_h =
        host_workspace->xmma_params.img_stride_w * params->w_;
    host_workspace->xmma_params.img_stride_d =
        host_workspace->xmma_params.img_stride_h * params->h_;
    host_workspace->xmma_params.img_stride_n =
        host_workspace->xmma_params.img_stride_d * params->d_;
    } else {
    host_workspace->xmma_params.img_stride_w = 1;
    host_workspace->xmma_params.img_stride_h =
        host_workspace->xmma_params.img_stride_w * params->w_;
    host_workspace->xmma_params.img_stride_d =
        host_workspace->xmma_params.img_stride_h * params->h_;
    host_workspace->xmma_params.img_stride_c =
        host_workspace->xmma_params.img_stride_d * params->d_;
    host_workspace->xmma_params.img_stride_n =
        host_workspace->xmma_params.img_stride_c * params->c_;
    }

    if (params->Layout_C == xmma::Convolution_layout::NHWC) {
    host_workspace->xmma_params.out_stride_c = 1;
    host_workspace->xmma_params.out_stride_w =
        host_workspace->xmma_params.out_stride_c * params->k_;
    host_workspace->xmma_params.out_stride_h =
        host_workspace->xmma_params.out_stride_w * params->q_;
    host_workspace->xmma_params.out_stride_d =
        host_workspace->xmma_params.out_stride_h * params->p_;
    host_workspace->xmma_params.out_stride_n =
        host_workspace->xmma_params.out_stride_d * params->o_;
    } else {
    host_workspace->xmma_params.out_stride_w = 1;
    host_workspace->xmma_params.out_stride_h =
        host_workspace->xmma_params.out_stride_w * params->q_;
    host_workspace->xmma_params.out_stride_d =
        host_workspace->xmma_params.out_stride_h * params->p_;
    host_workspace->xmma_params.out_stride_c =
        host_workspace->xmma_params.out_stride_d * params->o_;
    host_workspace->xmma_params.out_stride_n =
        host_workspace->xmma_params.out_stride_c * params->k_;
    }

    // Set the padding.
    host_workspace->xmma_params.pad[0][0] = params->pad_d_;
    host_workspace->xmma_params.pad[0][1] = params->pad_d_;
    host_workspace->xmma_params.pad[1][0] = params->pad_h_;
    host_workspace->xmma_params.pad[1][1] = params->pad_h_;
    host_workspace->xmma_params.pad[2][0] = params->pad_w_;
    host_workspace->xmma_params.pad[2][1] = params->pad_w_;

    // Set the strides.
    host_workspace->xmma_params.stride[0] = params->stride_d_;
    host_workspace->xmma_params.stride[1] = params->stride_h_;
    host_workspace->xmma_params.stride[2] = params->stride_w_;

    // Set the dilation.
    host_workspace->xmma_params.dilation[0] = params->dilation_d_;
    host_workspace->xmma_params.dilation[1] = params->dilation_h_;
    host_workspace->xmma_params.dilation[2] = params->dilation_w_;

    // Set alpha/beta.
    // using Epilogue_type = typename Traits::Epilogue_type;
    host_workspace->xmma_params.alpha = params->alpha_[0];
    host_workspace->xmma_params.beta  = params->beta_ [0];

    // Set relu params.
    host_workspace->xmma_params.with_relu = params->with_relu_;
    if (!params->with_relu_) {
        host_workspace->xmma_params.relu_lb = -std::numeric_limits<float>::infinity();
        host_workspace->xmma_params.relu_ub = +std::numeric_limits<float>::infinity();
    } else {
        host_workspace->xmma_params.relu_lb = params->relu_lb_;
        host_workspace->xmma_params.relu_ub = params->relu_ub_;
    }

    // Set bias param.
    host_workspace->xmma_params.with_bias = params->with_bias_ ? 1 : 0;

    // Is it a cross-correlation or a convolution.
    host_workspace->xmma_params.cross_correlation = params->is_cross_correlation_;

    // Force CTA rasterization to horizontal for wgrad.
    host_workspace->xmma_params.use_horizontal_cta_rasterization = %(cta_raster)s ? 1 : 0;

    // Set split-k parameters.
    host_workspace->xmma_params.split_k.slices  = params->split_k_slices_;
    host_workspace->xmma_params.split_k.buffers = params->split_k_buffers_;
    host_workspace->xmma_params.split_k.kernels = params->split_k_kernels_;
""" % locals())

        # Add split_k_t/r.
        if mode in ["fprop", "dgrad"]:
            file.write("""
    // How do we split the filter coordinates T*R*S*C.
    host_workspace->xmma_params.split_k_t = params->split_k_t_;
    host_workspace->xmma_params.split_k_r = params->split_k_r_;
""")

        # TODO: Fixme - it should be the same code for all kernels.
        if "fprop" in mode:
            file.write("""
    host_workspace->xmma_params.split_k_c = params->split_k_c_;
    // Compute the grid dimension.
    xmma::implicit_gemm::fprop::compute_grid_dimensions(host_workspace->grid,
                                                            host_workspace->xmma_params,
                                                            %(m)d,
                                                            %(n)d);
""" % locals())
        elif "dgrad" in mode:
            file.write("""
    host_workspace->xmma_params.split_k_k = params->split_k_c_;
    // Compute the grid dimension.
    xmma::implicit_gemm::dgrad::compute_grid_dimensions(host_workspace->grid,
                                                            host_workspace->xmma_params,
                                                            %(m)d,
                                                            %(n)d,
                                                            1);
""" % locals())
        elif "wgrad" in mode:
            file.write("""
    // Compute the grid dimension.
    xmma::implicit_gemm::%(mode)s::compute_grid_dimensions(host_workspace->grid,
                                                               host_workspace->xmma_params,
                                                               %(m)d,
                                                               %(n)d,
                                                               %(k)d,
                                                               1);
""" % locals())

        file.write("""
    // Finalize the initialization of the device parameters.
    host_workspace->xmma_params.initialize(host_workspace);
    return 0;
}
""" % locals())

        # Add a line of '/'.
        generate_separation_line(file)

        # Generate the function to determine the device workspace size.
        file.write("""
extern "C" int %(kernel_name)s_compute_device_workspace_size(
    size_t *size_in_bytes,
    const void *host_ptr) {
    size_in_bytes[0] = xmma::implicit_gemm::get_device_workspace_size<Kernel_traits>(host_ptr);
    return 0;
}
""" % locals())

        # Add a line of '/'.
        generate_separation_line(file)

        # Generate the function to initialize the device workspace size.
        file.write("""
extern "C" int %(kernel_name)s_initialize_device_workspace(
    void *device_ptr,
    const void *host_ptr,
    cudaStream_t stream) {
    auto *host_workspace = static_cast<const xmma::Host_workspace<Kernel_traits> *>(host_ptr);
    xmma::implicit_gemm::initialize_device_workspace<Kernel_traits>(
        host_workspace, device_ptr, stream);
    return 0;
}
""" % locals())

        # Add a line of '/'.
        generate_separation_line(file)

        # Generate the function to launch the kernel
        if "fprop" in mode:
            x_ptr, y_ptr, z_ptr, res_ptr = "act", "flt", "out", "out"
        elif "dgrad" in mode:
            x_ptr, y_ptr, z_ptr, res_ptr = "out", "flt", "act", "act"
        elif "wgrad" in mode:
            x_ptr, y_ptr, z_ptr, res_ptr = "act", "flt", "out", "flt"
        file.write("""
extern "C" int %(kernel_name)s_launch(
    void *act,
    void *flt,
    void *out,
    void *bias,
    void *host_ptr,
    void *device_ptr,
    cudaStream_t stream) {
    xmma::implicit_gemm::Runtime_params<Kernel_traits> p;
    memset(&p, 0, sizeof(p));
    p.descriptor_a  = 0x10000000u;
    p.descriptor_b  = 0x10000000u;
    p.descriptor_c1 = 0x10000000u;
    p.descriptor_d1 = 0x10000000u;
    xmma::Error err = xmma::implicit_gemm::run_kernel<Kernel_traits>(%(x_ptr)s,
                                                                             %(y_ptr)s,
                                                                             %(z_ptr)s,
                                                                             %(res_ptr)s,
                                                                             bias,
                                                                             nullptr,
                                                                             nullptr,
                                                                             host_ptr,
                                                                             device_ptr,
                                                                             p,
                                                                             stream);
    return err == xmma::Error::SUCCESS ? 0 : (int) err;
}
""" % locals())

        # Add a line of '/'.
        generate_separation_line(file)

        # Generate the function to create the kernel structure.
        file.write("""
extern "C" int %(kernel_name)s_build(
    Convolution_kernel *kernel) {
""" % locals())

        for fct in [
            "compute_host_workspace_size",
            "initialize_host_workspace",
            "compute_device_workspace_size",
            "initialize_device_workspace",
            "launch"]:
            file.write("    kernel->%(fct)s_ = \n" \
                       "        &%(kernel_name)s_%(fct)s;\n" % locals())

        file.write("    return 0;\n")
        file.write("}\n")

        # Add a line of '/'.
        generate_separation_line(file)
        file.write("\n");

####################################################################################################

def get_arch_flags(arch):
    arch_to_flags = {
        "Volta"  : "ARCH_VOLTA  | ARCH_TURING",
        "Turing" : "ARCH_TURING | ARCH_AMPERE",
        "Ampere" : "ARCH_AMPERE",
    }
    return arch_to_flags[arch]

####################################################################################################

def get_types_from_inst(inst, args):
    types = {
        "hmma_fp16"             : ("FP16",  "FP16",  "FP16",  "FP16"),
        "hmma_fp32"             : ("FP16",  "FP16",  "FP16",  "FP32"),
        "hmma_bf16_bf16_bf16"   : ("BF16",  "BF16",  "BF16",  "FP32"),
        "hmma_bf16_bf16_fp32"   : ("BF16",  "FP32",  "BF16",  "FP32"),
        "hmma_tf32_tf32_tf32"   : ("TF32",  "TF32",  "TF32",  "FP32"),
        "hmma_tf32_fp32_fp32"   : ("FP32",  "FP32",  "FP32",  "FP32"),
        "hmma_nhwc_nchw_tf32_tf32_tf32"  : ("TF32",  "TF32",  "TF32",  "FP32"),
        "hmma_nhwc_nchw_tf32_fp32_fp32"  : ("FP32",  "FP32",  "FP32",  "FP32"),
        "imma_int8_int32"       : ("INT8",  "INT8",  "INT32", "INT32"),
        "imma_int4_int32"       : ("INT4",  "INT4",  "INT32", "INT32"),
    }
    name = inst
    if not args is None:
        name += "_" + "_".join(args)
    return tuple("DATA_TYPE_" + t for t in types[name])

####################################################################################################

def generate_dispatch_file(sample_name, mode, flags, ta, tc, configs):
    dispatch_name = get_dispatch_name(sample_name, mode, flags, ta, tc)
    with open(os.path.join("generated", dispatch_name + ".cpp"), "w") as file:
        file.write(get_license_header("//"))
        file.write("\n")
        file.write("// !!! This file was generated -- do not edit of changes will be lost !!!\n")
        file.write("\n")

        # The includes.
        file.write("#include <%s.h>\n" % sample_name)

        # Add a line of '/'.
        generate_separation_line(file)
        file.write("\n")

        # Declare the different kernel "builders".
        file.write("extern \"C\" {\n")
        for arch, inst, arg, align_a, align_c, m, n, k, stages in configs:
            if skip_meaningless_cfg(mode, inst, align_a, align_c, ta, tc):
                continue
            kernel_name, _ = get_kernel_name(dispatch_name, arch, inst, arg, align_a, align_c, m, n, k, stages)
            file.write("int %(kernel_name)s_build(Convolution_kernel*);\n" % locals())
        file.write("} // extern \"C\"\n")

        # Add a line of '/'.
        generate_separation_line(file)
        file.write("\n")

        # Create the structure of kernels.
        file.write("""
struct Config_ {
    Convolution_traits_desc traits_desc_;
    Convolution_kernel_desc kernel_desc_;
};
""")
        # Add a line of '/'.
        generate_separation_line(file)

        # Create the list of configs.
        file.write("""
static const Config_ configs[] = {""")
        num_configs = 0
        for arch, inst, arg, align_a, align_c, m, n, k, stages in configs:
            if skip_meaningless_cfg(mode, inst, align_a, align_c, ta, tc):
                continue
            num_configs += 1
            kernel_name, _ = get_kernel_name(dispatch_name, arch, inst, arg, align_a, align_c, m, n, k, stages)
            arch_flags = get_arch_flags(arch)
            act_type, flt_type, out_type, acc_type = get_types_from_inst(inst, arg)
            alignment_a = get_alignment_a(inst, align_a)
            alignment_c = get_alignment_c(inst, align_c)
            file.write("""
    /* Kernel: %(kernel_name)s */
    { { %(arch_flags)s,
        %(act_type)s,
        %(flt_type)s,
        %(out_type)s,
        %(acc_type)s },
      { "%(kernel_name)s",
        %(m)d,
        %(n)d,
        %(k)d,
        %(stages)d,
        %(alignment_a)d,
        %(alignment_c)d,
        %(kernel_name)s_build } },
""" % locals())
        file.write("};\n")
        # Add a line of '/'.
        generate_separation_line(file)

        # Compute the kernels.
        file.write("""
int %(dispatch_name)s_list_kernels(
    int *num_kernels,
    Convolution_kernel_desc *kernel_descs,
    const Convolution_traits_desc *traits_desc,
    const uint32_t alignment_a,
    const uint32_t alignment_c) {

    int count = 0;
    for( int ii = 0; ii < %(num_configs)d; ++ii ) {
        const Config_ &cfg = configs[ii];
        if( (cfg.traits_desc_.arch_     &  traits_desc->arch_    ) &&
            (cfg.traits_desc_.act_type_ == traits_desc->act_type_) &&
            (cfg.traits_desc_.flt_type_ == traits_desc->flt_type_) &&
            (cfg.traits_desc_.out_type_ == traits_desc->out_type_) &&
            (cfg.traits_desc_.acc_type_ == traits_desc->acc_type_)
""" % locals())
        if ta in "n":
            file.write("""
            && (alignment_a %% cfg.kernel_desc_.Alignment_A == 0)
""" % locals())
        if ta in "t" and mode in "wgrad_indexed":
            file.write("""
            && (alignment_a %% cfg.kernel_desc_.Alignment_A == 0)
""" % locals())
        if tc in "n":
            file.write("""
            && (alignment_c %% cfg.kernel_desc_.Alignment_C == 0)
""" % locals())
        file.write("""
            ) {
            if( kernel_descs ) {
                memcpy(&kernel_descs[count], &cfg.kernel_desc_, sizeof(Convolution_kernel_desc));
            }
            count++;
        }
    }
    if( num_kernels ) {
        *num_kernels = count;
    }
    return 0;
}
""" % locals())

        # Add a line of '/'.
        generate_separation_line(file)
        file.write("\n")

####################################################################################################

def generate_makefile(sample_name, modes, configs):
    with open("Makefile", "w") as file:
        file.write(get_license_header("#"))
        file.write("\n")
        file.write("include Makefile.config\n")
        file.write("\n")

        # Generate the list of files to compile.
        #file.write("OBJECTS  = obj/%s.cpp.o\n" % sample_name)
        file.write("OBJECTS  = %s\n" % "")
        for mode, flags in modes:
            for ta in ["n", "t"]:
                for tc in ["n", "t"]:
                    dispatch_name = get_dispatch_name(sample_name, mode, flags, ta, tc)
                    file.write("OBJECTS += obj/generated/%(dispatch_name)s.cpp.o\n" % locals())
                    for arch, inst, arg, align_a, align_c, m, n, k, stages in configs:
                        if skip_meaningless_cfg(mode, inst, align_a, align_c, ta, tc):
                            continue
                        kernel_name, sm = get_kernel_name(dispatch_name, arch, inst, arg, align_a, align_c, m, n, k, stages)
                        file.write("OBJECTS += obj/generated/%s.sm%d.cu.o\n" % (kernel_name, sm))

        # Generate the rules.
        file.write("""
.PHONY: all
all:
\t+ $(MAKE) dirs
\t+ $(MAKE) bin/%(sample_name)s.exe
\t+ $(MAKE) lib/deconv.a

dirs:
\tif [ ! -d bin ]; then mkdir -p bin; fi
\tif [ ! -d lib ]; then mkdir -p lib; fi
\tif [ ! -d include ]; then mkdir -p include; fi
\tcp deconvXmmaApi.h include/.
\tif [ ! -d obj/generated ]; then mkdir -p obj/generated; fi

clean:
\trm -rf bin obj lib include

bin/%(sample_name)s.exe: obj/%(sample_name)s.cpp.o $(OBJECTS)
\t$(CXX) $(CXX_FLAGS) -o $@ $^ -L$(CUDA)/lib64 -lcudart -L$(CUDNN)/lib64 -lcudnn

lib/deconv.a: obj/deconvXmmaApi.cpp.o $(OBJECTS)
\tar rcs -o $@ $^

obj/generated/%%.sm70.cu.o: generated/%%.sm70.cu
\t$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -arch=compute_70 -code=\"compute_70,sm_70,sm_75\" -c -o $@ $<

obj/generated/%%.sm75.cu.o: generated/%%.sm75.cu
\t$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -arch=compute_75 -code=\"compute_75,sm_75,sm_80\" -c -o $@ $<

obj/generated/%%.sm80.cu.o: generated/%%.sm80.cu
\t$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -arch=compute_80 -code=\"compute_80,sm_80\" -c -o $@ $<

obj/generated/%%.cpp.o: generated/%%.cpp
\t$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c -o $@ $<

obj/%%.cpp.o: %%.cpp
\t$(CXX) $(CXX_FLAGS) $(INCLUDE_FLAGS) -c -o $@ $<

%%.sm70.ptx: generated/%%.sm70.cu
\t$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -arch=compute_70 -code=\"compute_70\" -ptx -o $@ $<

%%.sm75.ptx: generated/%%.sm75.cu
\t$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -arch=compute_75 -code=\"compute_75\" -ptx -o $@ $<

%%.sm80.ptx: generated/%%.sm80.cu
\t$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) -arch=compute_80 -code=\"compute_80\" -ptx -o $@ $<

""" % locals())

####################################################################################################

# Supported architectures/insts.
traits = [
    ("Volta" , "hmma_fp32",   [None], [(16)], [(16)]),
#    ("Volta" , "hmma_fp16",   [None], [(16)], [(16)]),
    ("Turing", "hmma_fp32",   [None], [(16)], [(16)]),
#    ("Turing", "hmma_fp16",   [None], [(16)], [(16)]),
    ("Ampere", "hmma_fp32",   [None], [(16)], [(16)]),
#    ("Ampere", "hmma_bf16",   [("bf16", "bf16"), ("bf16", "fp32")], [(16), (16)], [(16), ( 4)]),
#    ("Ampere", "hmma_tf32",   [("fp32", "fp32"), ("tf32", "tf32")], [( 4), (16)], [( 4), (16)]), # Input/output types Alignment_C.
#    ("Ampere", "hmma_fp16",   [None], [(16)], [(16)]),
]

# The CTA tiles.
tiles = [
    (256, 128,  32),
    (256,  64,  64),
    (256,  64,  32),
    (128, 256,  32),
    (128, 128,  64),
    (128, 128,  32),
    (128,  64,  64),
    ( 64,  64,  64),
    ( 64,  32,  64),
]

# The number of stages.
arch_stages = [
    ("Volta",  1),
    ("Turing", 1),
    ("Turing", 2),
    ("Ampere", 1),
    ("Ampere", 2),
    ("Ampere", 3),
]

# The different modes/kernels.
modes = [
#    ("fprop",     ""),
#    ("fprop_indexed", ""),
    ("dgrad",     ""),
    ("dgrad_indexed", ""),
#    ("wgrad_indexed", ""),
#    ("wgrad_indexed", "simple_1x1x1")
]

####################################################################################################

if __name__ == '__main__':
    # Are we asked to clean up things?
    if len(sys.argv) == 2 and sys.argv[1] == "--clean":
        for path in ["bin", "lib", "generated", "obj"]:
            if os.path.exists(path):
                shutil.rmtree(path)
        if os.path.exists("Makefile"):
            os.remove("Makefile")
    else:
        # Generate the configs.
        configs = []
        for arch, inst, args, aligns_a, aligns_c in traits:
            for arg in args:
                for align_a in aligns_a:
                    for align_c in aligns_c:
                        for m, n, k in tiles:
                            for stages in [s for (a, s) in arch_stages if a == arch]:
                                configs.append((arch, inst, arg, align_a, align_c, m, n, k, stages))

        # Make sure we have a "generated" folder.
        if not os.path.exists("generated"):
            os.mkdir("generated")

        # Create the kernel files.
        for arch, inst, arg, align_a, align_c, m, n, k, stages in configs:
            for mode, flags in modes:
                for ta in ["n", "t"]:
                    for tc in ["n", "t"]:
                        if skip_meaningless_cfg(mode, inst, align_a, align_c, ta, tc):
                            continue
                        generate_kernel_file("implicit_gemm", mode, flags, ta, tc, arch, inst, arg, align_a, align_c, m, n, k, stages)

        # Create the main dispatcher file.
        for mode, flags in modes:
            for ta in ["n", "t"]:
                for tc in ["n", "t"]:
                    generate_dispatch_file("implicit_gemm", mode, flags, ta, tc, configs)

        # Create the Makefile.
        generate_makefile("implicit_gemm", modes, configs)

####################################################################################################

