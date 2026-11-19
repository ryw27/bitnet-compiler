"""Transformation steps: load the model, swap layers, export the graph.

Validation lives in validation.py and never changes the model or the graph.
"""
import types

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.bitnet import ActQuant, AutoBitLinear

from errors import FrontendError


@torch.library.custom_op("bitnet::bitlinear", mutates_args=())
def bitlinear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Near one-to-one implementation of AutoBitLinear.forward."""
    return F.linear(ActQuant.apply(x), weight) * scale


@bitlinear.register_fake
def _(x, weight, scale):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _bitlinear_forward(self, input):
    return torch.ops.bitnet.bitlinear(input, self.weight, self.weight_scale)


def load_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    return model

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

# Example input for tracing. Its length (8) must not get baked into the graph;
# the sequence dimension is exported as dynamic.
EXAMPLE_INPUT_IDS = [128000, 9906, 1917, 264, 502, 1217, 2268, 40]

class LogitsOnly(torch.nn.Module):
    """Gives export two tensor inputs and one tensor output: the logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]

def export_model(model) -> torch.export.ExportedProgram:
    """Export the model for batch size 1 and any sequence length.

    Call after swap_bitlinear so each quantized layer is one bitnet::bitlinear node.
    """
    input_ids = torch.tensor([EXAMPLE_INPUT_IDS], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    seq = torch.export.Dim.AUTO
    with torch.no_grad():
        return torch.export.export(
            LogitsOnly(model),
            (input_ids, attention_mask),
            dynamic_shapes={"input_ids": {1: seq}, "attention_mask": {1: seq}},
            strict=False,
        )

if __name__ == "__main__":
    model_id = "microsoft/bitnet-b1.58-2B-4T"
    load_model(model_id)
