"""Validation steps. Nothing here changes the model or the graph."""
import torch


def capture_references(model, prompts: list[list[int]]) -> list[torch.Tensor]:
    if model.training:
        raise ValueError("capture_references requires a model in eval mode")

    references = []
    with torch.no_grad():
        for prompt in prompts:
            input_ids = torch.tensor([prompt], dtype=torch.long)
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            )
            references.append(outputs.logits.clone())
    return references
