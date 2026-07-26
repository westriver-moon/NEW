from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplet import WeightedRegularizedTripletLoss, pairwise_distance


PAIR_ORDER: Tuple[Tuple[str, str], ...] = (
    ("RGB", "IR"),
    ("RGB", "Fusion"),
    ("RGB", "Text"),
    ("IR", "Fusion"),
    ("IR", "Text"),
    ("Fusion", "Text"),
)


def _directional_hard_triplet(anchor: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    distances = pairwise_distance(anchor, target)
    same = labels[:, None].eq(labels[None, :])
    different = ~same
    if not same.any(dim=1).all() or not different.any(dim=1).all():
        raise ValueError("directional hard triplet requires positive and negative targets")
    hardest_positive = distances.masked_fill(~same, float("-inf")).max(dim=1).values
    hardest_negative = distances.masked_fill(~different, float("inf")).min(dim=1).values
    return F.softplus(hardest_positive - hardest_negative).mean()


class FourViewBidirectionalHardTripletLoss(nn.Module):
    def __init__(self, pair_weights: Mapping[str, float], global_weight: float = 1.0):
        super().__init__()
        expected = {"%s-%s" % pair for pair in PAIR_ORDER}
        if set(pair_weights) != expected:
            raise ValueError("pair_weights must contain exactly the six FAVTA pairs")
        self.pair_weights = {key: float(value) for key, value in pair_weights.items()}
        self.global_weight = float(global_weight)
        self.last_components: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def forward(self, views: Mapping[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        expected_views = {"RGB", "IR", "Text", "Fusion"}
        if set(views) != expected_views:
            missing = expected_views - set(views)
            extra = set(views) - expected_views
            raise ValueError("invalid FAVTA views; missing=%s extra=%s" % (sorted(missing), sorted(extra)))
        components: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        total = labels.new_zeros((), dtype=views["RGB"].dtype)
        normalizer = 0.0
        for left, right in PAIR_ORDER:
            key = "%s-%s" % (left, right)
            weight = self.pair_weights[key]
            forward = _directional_hard_triplet(views[left], views[right], labels)
            backward = _directional_hard_triplet(views[right], views[left], labels)
            components["%s->%s" % (left, right)] = forward
            components["%s->%s" % (right, left)] = backward
            total = total + weight * 0.5 * (forward + backward)
            normalizer += weight
        if normalizer <= 0.0:
            raise ValueError("sum of pair weights must be positive")
        self.last_components = components
        return self.global_weight * total / normalizer


class FAVTALoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        loss = config["loss"]
        self.id_weight = float(loss["id_weight"])
        self.wrt_weight = float(loss["wrt_weight"])
        self.favta_enabled = bool(loss["favta_enabled"])
        self.wrt = WeightedRegularizedTripletLoss()
        self.alignment = FourViewBidirectionalHardTripletLoss(
            loss["pair_weights"], float(loss["favta_weight"])
        )

    def forward(self, output: Mapping[str, Mapping[str, torch.Tensor]], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = output["features"]
        logits = output["logits"]
        id_loss = torch.stack([F.cross_entropy(logit, labels) for logit in logits.values()]).mean()
        if self.favta_enabled:
            metric = self.alignment(features, labels)
            metric_name = "favta"
            total = self.id_weight * id_loss + metric
        else:
            metric = self.wrt(features["Fusion"], labels)
            metric_name = "wrt"
            total = self.id_weight * id_loss + self.wrt_weight * metric
        return {"total": total, "id": id_loss, metric_name: metric}
