"""Generic frontend pipeline: load, capture references, swap, export, validate.

Transformation steps (swap, export) live here. Validation lives in validation.py and
never changes the model or the graph.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM

from . import validation
from .errors import FrontendError
from .spec import ModelSpec, synthetic_prompt

DTYPE = torch.bfloat16
# Length of the example input used for tracing. Must be at least 2, or export may fix the
# sequence length instead of keeping it dynamic.
EXAMPLE_LENGTH = 8


def load_model(model_dir: Path) -> torch.nn.Module:
    """Load a Hugging Face causal LM from a local directory, in eval mode."""
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=DTYPE, device_map="cpu")
    return model.eval()


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


def export_model(model, example_ids: list[int]) -> torch.export.ExportedProgram:
    """Export the model for batch size 1 and any sequence length.

    The example only starts the trace; the sequence dimension is exported as dynamic.
    """
    input_ids = torch.tensor([example_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    seq = torch.export.Dim.AUTO
    with torch.no_grad():
        return torch.export.export(
            LogitsOnly(model),
            (input_ids, attention_mask),
            dynamic_shapes={"input_ids": {1: seq}, "attention_mask": {1: seq}},
            strict=False,
        )


@dataclass
class FrontendResult:
    program: torch.export.ExportedProgram
    prompts: list[list[int]]
    references: list[torch.Tensor]  # logits from the original model, one per prompt


def run(
    spec: ModelSpec,
    model_dir: Path | None = None,
    log: Callable[[str], None] = lambda message: None,
) -> FrontendResult:
    """Produce a validated graph for one model. Raises FrontendError if any check fails."""
    log("loading model")
    model = load_model(model_dir or spec.model_dir)
    bos = model.config.bos_token_id
    if bos is None:
        raise FrontendError(f"{spec.name}: config has no bos_token_id")

    log("capturing reference logits")  # validation; must run before the swaps
    prompts = validation.validation_prompts(bos)
    references = validation.capture_references(model, prompts)

    log("swapping layers")
    swap_count = sum(swap(model) for swap in spec.swaps)

    log("exporting")
    program = export_model(model, synthetic_prompt(bos, EXAMPLE_LENGTH))

    log("validating")
    validation.validate(program, spec, swap_count, prompts, references, model.config.vocab_size)

    return FrontendResult(program, prompts, references)
