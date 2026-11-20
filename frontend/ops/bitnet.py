"""bitnet::bitlinear and the swap for Transformers' AutoBitLinear.

Works for any model whose quantized layers are offline-quantized AutoBitLinear.
"""
import types

import torch
import torch.nn.functional as F
from transformers.integrations.bitnet import ActQuant, AutoBitLinear

from ..errors import FrontendError


@torch.library.custom_op("bitnet::bitlinear", mutates_args=())
def bitlinear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Near one-to-one implementation of AutoBitLinear.forward."""
    return F.linear(ActQuant.apply(x), weight) * scale


@bitlinear.register_fake
def _(x, weight, scale):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _bitlinear_forward(self, input):
    return torch.ops.bitnet.bitlinear(input, self.weight, self.weight_scale)


def swap_bitlinear(model) -> int:
    """Route every AutoBitLinear through bitnet::bitlinear. Returns the number swapped.

    Modifies the model in place. All layers are checked before any are changed.
    The checks are preconditions of the swap, not validation: the op only
    supports layers without a bias, rms_norm or online quantization.
    """
    layers = [(name, m) for name, m in model.named_modules() if isinstance(m, AutoBitLinear)]

    for name, m in layers:
        if m.bias is not None:
            raise FrontendError(f"{name} has a bias, which bitnet::bitlinear does not support")
        if m.rms_norm is not None:
            raise FrontendError(f"{name} has an rms_norm, which bitnet::bitlinear does not support")
        if m.online_quant:
            raise FrontendError(f"{name} uses online_quant, which bitnet::bitlinear does not support")

    for _, m in layers:
        m.forward = types.MethodType(_bitlinear_forward, m)
    return len(layers)
