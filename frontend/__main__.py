"""Command line: uv run python -m frontend [--model NAME] [options]"""
import argparse
import sys
import time
from pathlib import Path

from safetensors.torch import save_file

from .errors import FrontendError
from .model_specs import DEFAULT_MODEL, MODELS
from .pipeline import run
from .spec import REPO_ROOT


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m frontend", description="Export a model as a validated torch.export graph.")
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL, help="model to export")
    parser.add_argument("--model-dir", type=Path, help="checkpoint directory (default: the model's own)")
    parser.add_argument("--out-dir", type=Path, help="output directory (default: build/frontend/<model>)")
    parser.add_argument("--print-graph", action="store_true", help="print the exported graph")
    args = parser.parse_args(argv)

    spec = MODELS[args.model]
    model_dir = args.model_dir or spec.model_dir
    out_dir = args.out_dir or REPO_ROOT / "build" / "frontend" / spec.name

    # Exit 2: problems found before any work starts.
    if not (model_dir / "config.json").is_file():
        _log(f"error: no config.json in {model_dir}")
        return 2

    # Exit 1: a precondition or validation check failed. Nothing is written.
    start = time.time()
    try:
        result = run(spec, model_dir, log=_log)
    except FrontendError as e:
        _log(f"frontend failed: {e}")
        return 1

    if args.print_graph:
        print(result.program.graph_module.print_readable(print_output=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    final_logits = {f"len_{len(p)}": r[0, -1].contiguous() for p, r in zip(result.prompts, result.references)}
    save_file(final_logits, str(out_dir / "reference_logits.safetensors"))

    _log(f"all checks passed in {time.time() - start:.0f}s; outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
