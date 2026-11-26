#pragma once
#include <cstddef>
#include <cstdint>

namespace quant {
float max_abs_matrix_val(const float *matrix, size_t total_elements);
float min_matrix_val(const float *matrix, size_t total_elements);
void clip(const float *input, int8_t *output, float inverse_scale);
void clip_non_negative(const float *input, int8_t *output, float inverse_scale,
                       size_t matrix_size, float upper);
float quantize_activations(const float *input, int8_t *output, size_t bits,
                           size_t height, size_t width);
} // namespace quant
