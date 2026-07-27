from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .favta_network import SharedIdentityClassifier
from .vision_transformer import OverlappingVisionTransformer


class VisualPretrainNetwork(nn.Module):
    def __init__(self, vision: OverlappingVisionTransformer, embed_dim: int, num_classes: int):
        super().__init__()
        self.vision = vision
        self.classifier = SharedIdentityClassifier(embed_dim, num_classes)

    def encode_image(self, image):
        return F.normalize(self.vision(image), dim=1)

    def forward(self, visible, ir):
        visible_feature = self.encode_image(visible)
        ir_feature = self.encode_image(ir)
        return {
            "features": {"Visible": visible_feature, "IR": ir_feature},
            "logits": {
                "Visible": self.classifier(visible_feature),
                "IR": self.classifier(ir_feature),
            },
        }


def build_visual_pretrain_model(config, num_classes):
    model = config["model"]
    vision = OverlappingVisionTransformer(
        model["image_size"],
        int(model["embed_dim"]),
        int(model["depth"]),
        int(model["num_heads"]),
        float(model["mlp_ratio"]),
        float(model["dropout"]),
        float(model["drop_path"]),
        model["tokenizer"]["branches"],
    )
    return VisualPretrainNetwork(vision, int(model["embed_dim"]), int(num_classes))
