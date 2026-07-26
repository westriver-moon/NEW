from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFeatureFusion(nn.Module):
    def __init__(self, visual_weight: float = 0.5):
        super().__init__()
        if not 0.0 <= visual_weight <= 1.0:
            raise ValueError("fusion weight must be in [0, 1]")
        self.visual_weight = float(visual_weight)

    def forward(self, ir: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if ir.shape != text.shape:
            raise ValueError("IR and Text feature shapes must match")
        ir = F.normalize(ir, dim=1)
        text = F.normalize(text, dim=1)
        return self.visual_weight * ir + (1.0 - self.visual_weight) * text

