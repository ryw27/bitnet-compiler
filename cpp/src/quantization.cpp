#include "quantization.h"
#include <cmath>

namespace quant {
    float max_abs_matrix_val(const float *matrix, int64_t total_elements) {
        float result = 0.0f;
        for (size_t i = 0; i < total_elements; ++i) {
            result = fmax(result, fabs(matrix[i]));
        }
        return result;
    }

    float min_matrix_val(const float *matrix, int64_t total_elements) {
        float result = 0.0f;
        for (size_t i = 0; i < total_elements; ++i) {
            result = fmin(result, (matrix[i]));
        }
        return result;
    }

    void clip(const float *input, int8_t *output, float inverse_scale,
              int64_t matrix_size, float upper, float lower) {
        for (size_t i = 0; i < matrix_size; ++i) {
            float elt = *(input + i);
            float scaled = std::nearbyint(elt * inverse_scale);
            output[i] = static_cast<int8_t>(fmax(lower, fmin(upper, scaled)));
        }
    }

    void clip_non_negative(const float *input, int8_t *output, float inverse_scale,
                           int64_t matrix_size, float upper) {
        float smallest_matrix_val = min_matrix_val(input, matrix_size);
        for (size_t i = 0; i < matrix_size; ++i) {
            float elt = *(input + i) - smallest_matrix_val;
            float scaled = std::nearbyint(elt * inverse_scale);
            output[i] = static_cast<int8_t>(fmax(0, fmin(upper, scaled)));
        }
    }

    // TODO Wonder if quantize into int16_t or int32_t would be better than 8-bit integers
    float quantize_activations(const float *input, int8_t *output, int bits,
                               int64_t height, int64_t width) {
        // * int8 [-128, 127]
        size_t total_elements = height * width;
        float upper = (1 << (bits - 1)) - 1;
        float lower = -upper - 1;
        float gamma = fmax(max_abs_matrix_val(input, height * width), 0.00001);
        float inverse_scale_factor = upper / gamma;
        clip(input, output, inverse_scale_factor , total_elements, upper, lower);
        return inverse_scale_factor;
    }

    // TODO consider whatever shifts may be needed to account for relulu
    void dequantize_activations(const int8_t *input, float *output, float scale_factor, int64_t height, int64_t width) {
        size_t total_elements = height * width;
        for (size_t i = 0; i < total_elements; ++i) {
            output[i] = input[i] * scale_factor;
        }
    }
}
