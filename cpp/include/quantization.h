#pragma once
#include <cstddef>
#include <cstdint>

namespace quant {
float max_abs_matrix_val(const float *matrix, int64_t total_elements);
void clip(const float *input, int8_t *output, float inverse_scale);
float quantize_activations(const float *input, int8_t *output, int bits, int64_t height,
                           int64_t width);
void dequantize_activations(const int8_t *input, float *output, float scale_factor, int64_t height,
                            int64_t width);
} // namespace quant
