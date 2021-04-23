import math

import torch
from torch.optim.adamw import AdamW
from transformers import get_scheduler


def create_optimizer_and_scheduler(model, lr,
                                   num_training_steps,
                                   weight_decay=0.0,
                                   warmup_ratio=0.0,
                                   warmup_steps=0,
                                   lr_scheduler_type="linear"):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    warmup_steps = warmup_steps if warmup_steps > 0 else math.ceil(num_training_steps * warmup_ratio)

    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, lr_scheduler





def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result





