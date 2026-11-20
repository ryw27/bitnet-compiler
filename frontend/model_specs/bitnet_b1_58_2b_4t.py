"""microsoft/bitnet-b1.58-2B-4T: 30 decoder layers, offline-quantized AutoBitLinear."""
from ..ops.bitnet import swap_bitlinear
from ..spec import REPO_ROOT, ModelSpec

SPEC = ModelSpec(
    name="bitnet-b1.58-2b-4t",
    model_dir=REPO_ROOT / "models" / "bitnet-b1.58-2B-4T",
    swaps=(swap_bitlinear,),
    expected_swaps=210,  # 30 layers x 7 projections
    expected_ops={
        "bitnet::bitlinear": 210,  # one node per quantized layer
        "aten::linear": 1,         # lm_head, which is not quantized
        "aten::round": 0,          # quantization must not be inlined into the graph
    },
)
