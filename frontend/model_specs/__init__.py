"""Registry of supported models.

To add a model, create a module here that defines SPEC and add it to MODELS below.
"""
from ..spec import ModelSpec
from . import bitnet_b1_58_2b_4t

MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        bitnet_b1_58_2b_4t.SPEC,
    )
}

DEFAULT_MODEL = bitnet_b1_58_2b_4t.SPEC.name
