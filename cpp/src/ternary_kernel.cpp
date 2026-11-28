#include "ternary_kernel.h"
#include "quantization.h"
#include <cmath>
#include <torch/extension.h>

torch::Tensor kernel(const torch::Tensor &activations, const torch::Tensor &weights) {
    TORCH_CHECK(activations.device().is_cpu(), "Device must be CPU");
    TORCH_CHECK(weights.device().is_cpu(), "Device must be CPU");
    TORCH_CHECK(activations.is_contiguous());
    TORCH_CHECK(weights.is_contiguous());
    TORCH_CHECK(activations.dim() >= 2, "Activations must be 2D or 3D");
    TORCH_CHECK(weights.dim() == 2, "Weights must be 2D");

    const int64_t A_COLS = activations.size(-1);
    const int64_t A_ROWS = activations.numel() / A_COLS;

    const int64_t W_COLS = weights.size(0);
    const int64_t W_ROWS = weights.size(1);
    const int64_t OUT_COLS = W_COLS;

    const float *act_ptr = activations.data_ptr<float>();
    const float *w_ptr = weights.data_ptr<float>();

    at::Tensor qzd_act = torch::empty({A_ROWS, A_COLS});
    int8_t *qzd_act_ptr = qzd_act.data_ptr<int8_t>();

    at::Tensor out = torch::empty({A_ROWS, OUT_COLS});
    float *out_ptr = out.data_ptr<float>();

    float scale_factor = quant::quantize_activations(act_ptr, qzd_act_ptr, 8, A_ROWS, A_COLS);
    quant::dequantize_activations(qzd_act_ptr, out_ptr, scale_factor, A_ROWS, A_COLS);

    std::vector<int64_t> out_shape = activations.sizes().vec();
    out_shape.back() = OUT_COLS;
    return out.view(out_shape);
}
