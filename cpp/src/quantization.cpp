#include "quantization.h"
#include "ternary_kernel.h"
#include <cmath>

namespace quant {
float max_abs_matrix_val(const float *matrix, int64_t total_elements) {
    float result = at::parallel_reduce(
        0, total_elements, kernel::grain_size(total_elements), 0.0f,
        [&](int64_t begin, int64_t end, float init) -> float {
            float intermediate = init;
            for (int64_t i = begin; i < end; ++i) {
                result = fmax(result, fabs(matrix[i]));
            }
            return result;
        },
        [](float a, float b) -> float { return fmax(a, b); });
    return result;
}

void clip(const float *input, int8_t *output, float inverse_scale, int64_t matrix_size, float upper,
          float lower) {
    // TODO Determine Grain Size dynammically depending on input and workload
    at::parallel_for(0, matrix_size, kernel::grain_size(matrix_size), [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            float elt = *(input + i);
            float scaled = std::nearbyint(elt * inverse_scale);
            output[i] = static_cast<int8_t>(fmax(lower, fmin(upper, scaled)));
        }
    });
}

// TODO Wonder if quantize into int16_t or int32_t would be better than 8-bit
// integers
float quantize_activations(const float *input, int8_t *output, int bits, int64_t height,
                           int64_t width) {
    // * int8 [-128, 127]
    size_t total_elements = height * width;
    float upper = (1 << (bits - 1)) - 1;
    float lower = -upper - 1;
    float gamma = fmax(max_abs_matrix_val(input, height * width), 0.00001);
    float inverse_scale_factor = upper / gamma;
    clip(input, output, inverse_scale_factor, total_elements, upper, lower);
    return inverse_scale_factor;
}

// TODO consider whatever shifts may be needed to account for relulu
void dequantize_activations(const int8_t *input, float *output, float scale_factor, int64_t height,
                            int64_t width) {
    size_t total_elements = height * width;
    at::parallel_for(0, total_elements, kernel::grain_size(total_elements), [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            output[i] = input[i] * scale_factor;
        }
    });
}
} // namespace quant
