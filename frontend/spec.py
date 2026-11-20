"""Per-model configuration.

Everything model-specific lives in a ModelSpec. The pipeline, export and validation code
is generic and reads what it needs from the spec or from the loaded model's config.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def synthetic_prompt(bos: int, length: int) -> list[int]:
    """BOS followed by consecutive token IDs from 1000. The values don't matter, only the length."""
    return [bos] + list(range(1000, 1000 + length - 1))


@dataclass(frozen=True)
class ModelSpec:
    # Used for --model and the output directory name.
    name: str
    # Default checkpoint directory.
    model_dir: Path
    # Layer swaps to apply before export. Each modifies the model in place and returns
    # the number of layers it swapped.
    swaps: tuple[Callable[[torch.nn.Module], int], ...]
    # Check G1: the total number of layers the swaps must replace.
    expected_swaps: int
    # Check G2: the exact number of nodes of each op in the exported graph,
    # e.g. {"bitnet::bitlinear": 210}.
    expected_ops: dict[str, int]
