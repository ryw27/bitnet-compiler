"""Validation steps. Nothing here changes the model or the graph."""
import torch

from .errors import FrontendError
from .spec import ModelSpec, synthetic_prompt


PROMPT_LENGTHS = (2, 3, 5, 8, 32, 128)


def validation_prompts(bos: int) -> list[list[int]]:
    return [synthetic_prompt(bos, n) for n in PROMPT_LENGTHS]


def capture_references(model, prompts: list[list[int]]) -> list[torch.Tensor]:
    if model.training:
        raise ValueError("capture_references requires a model in eval mode")

    references = []
    with torch.no_grad():
        for prompt in prompts:
            input_ids = torch.tensor([prompt], dtype=torch.long)
            outputs = model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
            references.append(outputs.logits.clone())
    return references


def _op_name(target) -> str:
    """'namespace::op' for registered ops, e.g. 'aten::linear' or 'bitnet::bitlinear'."""
    schema = getattr(target, "_schema", None)
    return schema.name if schema is not None else str(target)


def validate_graph(
    program: torch.export.ExportedProgram,
    swap_count: int,
    expected_swaps: int,
    expected_ops: dict[str, int],
) -> None:
    """Checks G1 and G2."""
    if swap_count != expected_swaps:
        raise FrontendError(f"G1: swaps replaced {swap_count} layers, expected {expected_swaps}")

    names = [_op_name(n.target) for n in program.graph.nodes if n.op == "call_function"]
    for op, expected in expected_ops.items():
        found = names.count(op)
        if found != expected:
            raise FrontendError(f"G2: graph has {found} {op} nodes, expected {expected}")


def validate_logits(
    program: torch.export.ExportedProgram,
    prompts: list[list[int]],
    references: list[torch.Tensor],
    vocab_size: int,
) -> None:
    """Checks N1 and N2: run the exported graph and compare with the original model."""
    if len(prompts) != len(references):
        raise ValueError(f"{len(prompts)} prompts but {len(references)} references")

    module = program.module()
    with torch.no_grad():
        for prompt, reference in zip(prompts, references):
            n = len(prompt)
            input_ids = torch.tensor([prompt], dtype=torch.long)
            logits = module(input_ids, torch.ones_like(input_ids))

            expected = [1, n, vocab_size]
            if list(logits.shape) != expected:
                raise FrontendError(f"N2: length {n}: logits shape {list(logits.shape)}, expected {expected}")
            if logits.shape != reference.shape:
                raise FrontendError(
                    f"N1: length {n}: reference shape {list(reference.shape)} does not match logits {expected}"
                )
            if not torch.equal(logits, reference):
                diff = (logits.float() - reference.float()).abs().max().item()
                raise FrontendError(f"N1: length {n}: logits differ from reference (max abs diff {diff})")


def validate(
    program: torch.export.ExportedProgram,
    spec: ModelSpec,
    swap_count: int,
    prompts: list[list[int]],
    references: list[torch.Tensor],
    vocab_size: int,
) -> None:
    """Run every check. Raises FrontendError on the first failure."""
    validate_graph(program, swap_count, spec.expected_swaps, spec.expected_ops)
    validate_logits(program, prompts, references, vocab_size)
