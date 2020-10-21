/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "NvInfer.h"

#include <string>
#include <vector>

/* Helper function to split a string based on a delimiting character */
inline std::vector<std::string> splitString(const std::string& input, const std::string& delimiter)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t next = 0;
    while(next != std::string::npos)
    {
        next = input.find(delimiter, start);
        result.emplace_back(input, start, next - start);
        start = next + 1;
    }
    return result;
}

// Get element size of a data type.
inline unsigned int getElementSize(nvinfer1::DataType t) {
    switch (t) {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    // Fall through to error
    default: break;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

// Return m rounded up to nearest multiple of n
inline int roundUp(int m, int n)
{
    return ((m + n - 1) / n) * n;
}

// Get the size of a binding dimensions
inline int64_t volume(const nvinfer1::Dims& d, const nvinfer1::TensorFormat& format, const bool hasImplicitBatch = false)
{
    nvinfer1::Dims d_new = d;
    // Get number of scalars per vector.
    int spv{1};
    switch(format)
    {
        case nvinfer1::TensorFormat::kCHW2: spv = 2; break;
        case nvinfer1::TensorFormat::kCHW4: spv = 4; break;
        case nvinfer1::TensorFormat::kHWC8: spv = 8; break;
        case nvinfer1::TensorFormat::kCHW16: spv = 16; break;
        case nvinfer1::TensorFormat::kCHW32: spv = 32; break;
        case nvinfer1::TensorFormat::kLINEAR:
        default: spv = 1; break;
    }
    if (spv > 1)
    {
        assert(d.nbDims >= 3); // Vectorized format only makes sense when nbDims>=3.
        d_new.d[d_new.nbDims - 3] = roundUp(d_new.d[d_new.nbDims - 3], spv);
    }
    // Skip the first dimension, which is batch dim.
    return std::accumulate(d_new.d + (hasImplicitBatch ? 0 : 1), d_new.d + d_new.nbDims, 1, std::multiplies<int64_t>());
}

inline int64_t volume(const nvinfer1::Dims& d, const bool hasImplicitBatch = false)
{
    return volume(d, nvinfer1::TensorFormat::kLINEAR, hasImplicitBatch);
}

// Create a shared pointer of an nvinfer1:: object which will be automatically destroyed when going out of scope.
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
inline std::shared_ptr<T> InferObject(T* obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, InferDeleter());
}

#endif // __UTILS_HPP__
