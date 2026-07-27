from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import WeightedFeatureFusion
from .text_encoder import TextTransformerEncoder
from .vision_transformer import OverlappingVisionTransformer


def _external_state(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _component_state(path: str, expected_kind: str):
    state = _external_state(path)
    if state.get("schema_version") != 1 or state.get("kind") != expected_kind:
        raise ValueError("external parameter file is not a %s checkpoint" % expected_kind)
    if not isinstance(state.get("model"), Mapping):
        raise ValueError("external parameter file has no model state")
    return state["model"]


class SharedIdentityClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.norm.bias.requires_grad_(False)
        self.linear = nn.Linear(dim, num_classes, bias=False)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(feature))


class FAVTANetwork(nn.Module):
    view_names = ("RGB", "IR", "Text", "Fusion")

    def __init__(self, vision, text, fusion, classifier):
        super().__init__()
        self.vision = vision
        self.text = text
        self.fusion = fusion
        self.classifier = classifier
        self.vision_frozen = False

    def freeze_vision(self, frozen: bool = True) -> None:
        self.vision_frozen = bool(frozen)
        for parameter in self.vision.parameters():
            parameter.requires_grad_(not self.vision_frozen)
        self.vision.train(self.training and not self.vision_frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision_frozen:
            self.vision.eval()
        return self

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.vision(image), dim=1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text(text), dim=1)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, text: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        features = {
            "RGB": self.encode_image(rgb),
            "IR": self.encode_image(ir),
            "Text": self.encode_text(text),
        }
        features["Fusion"] = F.normalize(self.fusion(features["IR"], features["Text"]), dim=1)
        if set(features) != set(self.view_names):
            raise RuntimeError("model must produce exactly RGB, IR, Text, and Fusion views")
        logits = {name: self.classifier(feature) for name, feature in features.items()}
        return {"features": features, "logits": logits}


def build_model(config: Mapping[str, Any], num_classes: int) -> FAVTANetwork:
    model_config = config["model"]
    embed_dim = int(model_config["embed_dim"])
    vision = OverlappingVisionTransformer(
        model_config["image_size"],
        embed_dim,
        int(model_config["depth"]),
        int(model_config["num_heads"]),
        float(model_config["mlp_ratio"]),
        float(model_config["dropout"]),
        float(model_config["drop_path"]),
        model_config["tokenizer"]["branches"],
    )
    text_depth = max(1, int(model_config["depth"]) // 2)
    text = TextTransformerEncoder(
        int(model_config["text_vocab_size"]),
        int(model_config["text_length"]),
        embed_dim,
        text_depth,
        int(model_config["num_heads"]),
        float(model_config["dropout"]),
    )
    model = FAVTANetwork(
        vision,
        text,
        WeightedFeatureFusion(float(model_config["fusion_weight"])),
        SharedIdentityClassifier(embed_dim, int(num_classes)),
    )
    if model_config.get("vision_pretrained"):
        model.vision.load_state_dict(
            _component_state(model_config["vision_pretrained"], "visual_encoder"),
            strict=True,
        )
    if model_config.get("text_pretrained"):
        model.text.load_state_dict(
            _component_state(model_config["text_pretrained"], "text_encoder"),
            strict=True,
        )
    model.freeze_vision(bool(model_config.get("freeze_vision", True)))
    return model
