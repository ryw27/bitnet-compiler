"""Frontend: turn a Hugging Face checkpoint into a validated torch.export graph.

Run from the repo root with `uv run python -m frontend`. From Python:

    from frontend.model_specs import MODELS
    from frontend.pipeline import run

    result = run(MODELS["bitnet-b1.58-2b-4t"])
    result.program  # the validated ExportedProgram
"""
