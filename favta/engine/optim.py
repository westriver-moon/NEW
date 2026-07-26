from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch


def build_optimizer(config: Mapping[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    train = config["train"]
    groups: Dict[str, List[torch.nn.Parameter]] = {"vision": [], "text": [], "other": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("vision."):
            groups["vision"].append(parameter)
        elif name.startswith("text."):
            groups["text"].append(parameter)
        else:
            groups["other"].append(parameter)
    parameter_groups = []
    for name, lr_key in (("vision", "lr_vision"), ("text", "lr_text"), ("other", "lr_other")):
        if groups[name]:
            parameter_groups.append(
                {
                    "params": groups[name],
                    "lr": float(train[lr_key]),
                    "weight_decay": float(train["weight_decay"]),
                    "group_name": name,
                }
            )
    if not parameter_groups:
        raise ValueError("model has no trainable parameters")
    optimizer = str(train["optimizer"]).lower()
    if optimizer == "adamw":
        return torch.optim.AdamW(parameter_groups)
    if optimizer == "sgd":
        return torch.optim.SGD(parameter_groups, momentum=0.9)
    raise ValueError("optimizer must be adamw or sgd")

