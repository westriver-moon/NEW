from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    squared = left.pow(2).sum(1, keepdim=True) + right.pow(2).sum(1, keepdim=True).t()
    return squared.addmm(left, right.t(), beta=1.0, alpha=-2.0).clamp_min(1.0e-12).sqrt()


class WeightedRegularizedTripletLoss(nn.Module):
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        distances = pairwise_distance(features, features)
        same = labels[:, None].eq(labels[None, :])
        same.fill_diagonal_(False)
        different = ~labels[:, None].eq(labels[None, :])
        positive_logits = distances.masked_fill(~same, float("-inf"))
        negative_logits = (-distances).masked_fill(~different, float("-inf"))
        if not same.any(dim=1).all() or not different.any(dim=1).all():
            raise ValueError("WRT requires a positive and negative for every anchor")
        positive_weights = torch.softmax(positive_logits, dim=1)
        negative_weights = torch.softmax(negative_logits, dim=1)
        positive = (distances * positive_weights).sum(dim=1)
        negative = (distances * negative_weights).sum(dim=1)
        return F.softplus(positive - negative).mean()

