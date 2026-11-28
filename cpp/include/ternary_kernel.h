#pragma once
// #include <immintrin.h>
#include <arm_neon.h>

#include <cstddef>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>

namespace bitnet{
constexpr int64_t DEFAULT_GRAIN_SIZE = at::internal::GRAIN_SIZE; // 32768
inline int64_t grain_size(int64_t operations) {
    return std::max<int64_t>(1, DEFAULT_GRAIN_SIZE/operations);
}

}
