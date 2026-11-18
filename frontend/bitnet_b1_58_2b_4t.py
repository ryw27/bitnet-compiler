from optimum.fx.utils import symbolic_trace
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def fetch_model():
    model_id = "microsoft/bitnet-b1.58-2B-4T"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

def fetch_graph_module(model):
    traced_model = symbolic_trace(
        model,
        input_names=["input_ids", "attention_mask"],
        disable_check = True
    )

    print(model.graph)

if __name__ == "__main__":
    model = fetch_model()
    fetch_graph_module(model)
