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

#include <samples.h>
#include <cudnn.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define XMMA_CHECK_CUDNN(call) do { \
  cudnnStatus_t status_ = call; \
  if( status_ != CUDNN_STATUS_SUCCESS ) { \
    fprintf(stderr, "Cudnn error in file \"%s\" at line %d: %s\n", \
        __FILE__, \
        __LINE__, \
        cudnnGetErrorString(status_)); \
    exit(1); \
  } \
} while(0)

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline cudnnDataType_t data_type_to_cudnn(Data_type dtype) {
    switch( dtype ) {
    case DATA_TYPE_TF32:
    case DATA_TYPE_FP32:
        return CUDNN_DATA_FLOAT;
    case DATA_TYPE_FP16:
        return CUDNN_DATA_HALF;
    case DATA_TYPE_INT32:
        return CUDNN_DATA_INT32;
    case DATA_TYPE_INT8:
        return CUDNN_DATA_INT8;
    case DATA_TYPE_INT8x32:
        return CUDNN_DATA_INT8x32;
    // TODO: will change CUDNN_DATA_HALF to CUDNN_DATA_16BF when cudnn support it
    case DATA_TYPE_BF16:
        return CUDNN_DATA_HALF;
    default:
        assert(false);
        return CUDNN_DATA_FLOAT;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

