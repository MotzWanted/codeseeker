import torch

from model.label_wise_attention import LabelCrossAttention

def _get_last_layer(model):
    children = list(model.named_children())
    return children[-1][0], children[-1][1] if children else None

def replace_decoder_head_with_label_wise_attention(
    model: torch.nn.Module, num_classes: int, input_size: int
) -> torch.nn.Module:
    """
    Replace the final decoder head of a decoder-only model with a label-wise attention head.

    Args:
        model (torch.nn.Module): The model to modify
        num_classes (int): The number of classes
        input_size (int): The size of the input

    Returns:
        torch.nn.Module: The modified model
    """
    last_layer_name, _ = _get_last_layer(model)
    if last_layer_name is None:
        raise ValueError("Could not identity the last layer of the model")
    setattr(model, last_layer_name, LabelCrossAttention(input_size, num_classes))
    print(f"Replaced last layer '{last_layer_name}' with nn.Identity()")

    return model

def has_label_wise_attention(model: torch.nn.Module) -> bool:
    """
    Check if the model uses label-wise attention.

    Args:
        model (torch.nn.Module): The model to check

    Returns:
        bool: True if the model uses label-wise attention
    """
    _, last_layer = _get_last_layer(model)
    return isinstance(last_layer, LabelCrossAttention)


# Test the function on a dummy model
import transformers

model = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased")
print(model)
model = replace_decoder_head_with_label_wise_attention(model, 10, 768)
print(model)
print(model_uses_label_wise_attention(model))