from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplet import pairwise_distance


@dataclass(frozen=True)
class StageATransition:
    rho: float
    rgb_identity_ratio: torch.Tensor


def cosine_transition(
    epoch: int,
    iteration: int,
    iterations_per_epoch: int,
    gray_epochs: int,
    transition_epochs: int,
) -> float:
    steps = max(int(iterations_per_epoch), 1)
    position = float(epoch) + float(iteration) / steps
    if position < gray_epochs:
        return 0.0
    if position >= gray_epochs + transition_epochs:
        return 1.0
    progress = (position - gray_epochs) / transition_epochs
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def _identity_labels(labels: torch.Tensor, instances: int) -> torch.Tensor:
    labels = labels.view(-1)
    if labels.numel() % instances:
        raise ValueError("Stage A batch size must be divisible by instances_per_identity")
    chunks = labels.view(-1, instances)
    if not torch.all(chunks.eq(chunks[:, :1])):
        raise ValueError("Stage A requires contiguous identity-balanced batches")
    if chunks.size(0) < 2:
        raise ValueError("Stage A requires at least two identities per batch")
    return chunks[:, 0]


def _batch_hard_triplet(
    features: torch.Tensor, labels: torch.Tensor, margin: float
) -> torch.Tensor:
    distances = pairwise_distance(features, features)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = labels[:, None].ne(labels[None, :])
    if not same.any(dim=1).all() or not different.any(dim=1).all():
        raise ValueError("batch-hard triplet requires a positive and negative for every anchor")
    positive = distances.masked_fill(~same, float("-inf")).max(dim=1).values
    negative = distances.masked_fill(~different, float("inf")).min(dim=1).values
    return F.relu(float(margin) + positive - negative).mean()


def _cross_modal_hard_triplet(
    visible: torch.Tensor,
    infrared: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    distances = pairwise_distance(visible, infrared)
    same = labels[:, None].eq(labels[None, :])
    different = ~same

    def directional(matrix: torch.Tensor) -> torch.Tensor:
        positive = matrix.masked_fill(~same, float("-inf")).max(dim=1).values
        negative = matrix.masked_fill(~different, float("inf")).min(dim=1).values
        return F.relu(float(margin) + positive - negative).mean()

    return 0.5 * (directional(distances) + directional(distances.t()))


def _mean_sample_alignment(
    visible: torch.Tensor,
    infrared: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    same = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    intra_mask = same & ~diagonal
    if not intra_mask.any(dim=1).all():
        raise ValueError("sample alignment requires two samples per identity")
    intra_visible = pairwise_distance(visible, visible)
    intra_infrared = pairwise_distance(infrared, infrared)
    cross = pairwise_distance(visible, infrared)
    intra_visible_mean = (intra_visible * intra_mask).sum(dim=1) / intra_mask.sum(dim=1)
    intra_infrared_mean = (intra_infrared * intra_mask).sum(dim=1) / intra_mask.sum(dim=1)
    cross_visible_mean = (cross * same).sum(dim=1) / same.sum(dim=1)
    cross_infrared_mean = (cross.t() * same).sum(dim=1) / same.sum(dim=1)
    return 0.5 * (
        (cross_visible_mean - intra_visible_mean).pow(2).mean()
        + (cross_infrared_mean - intra_infrared_mean).pow(2).mean()
    )


class ModalitySeparatedCenterTripletLoss(nn.Module):
    def __init__(self, instances_per_identity: int, margin: float):
        super().__init__()
        self.instances = int(instances_per_identity)
        self.margin = float(margin)

    def forward(
        self,
        visible: torch.Tensor,
        infrared: torch.Tensor,
        labels: torch.Tensor,
        return_stats: bool = False,
    ):
        identities = _identity_labels(labels, self.instances)
        if visible.shape != infrared.shape or visible.size(0) != labels.numel():
            raise ValueError("center triplet requires aligned visible and infrared features")
        with torch.cuda.amp.autocast(enabled=False):
            visible = F.normalize(visible.float(), dim=1)
            infrared = F.normalize(infrared.float(), dim=1)
            count = identities.numel()
            visible_centers = F.normalize(
                visible.view(count, self.instances, -1).mean(dim=1), dim=1
            )
            infrared_centers = F.normalize(
                infrared.view(count, self.instances, -1).mean(dim=1), dim=1
            )
            all_centers = torch.cat((visible_centers, infrared_centers), dim=0)
            all_identities = torch.cat((identities, identities), dim=0)

            def directional(anchor: torch.Tensor, positive: torch.Tensor):
                positive_distance = torch.norm(anchor - positive, dim=1)
                distances = torch.cdist(anchor, all_centers)
                negative_mask = identities[:, None].ne(all_identities[None, :])
                negative_distance = distances.masked_fill(
                    ~negative_mask, float("inf")
                ).min(dim=1).values
                hinge = F.relu(self.margin + positive_distance - negative_distance)
                return hinge.mean(), positive_distance, negative_distance, hinge

            visible_loss, visible_positive, visible_negative, visible_hinge = directional(
                visible_centers, infrared_centers
            )
            infrared_loss, infrared_positive, infrared_negative, infrared_hinge = directional(
                infrared_centers, visible_centers
            )
            loss = 0.5 * (visible_loss + infrared_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("center triplet produced a non-finite value")
            if not return_stats:
                return loss
            return loss, {
                "msct_active_identity_ratio": 0.5
                * (
                    (visible_hinge > 0).float().mean()
                    + (infrared_hinge > 0).float().mean()
                ),
                "msct_positive_center_distance": 0.5
                * (visible_positive.mean() + infrared_positive.mean()),
                "msct_negative_center_distance": 0.5
                * (visible_negative.mean() + infrared_negative.mean()),
            }


class StageALoss(nn.Module):
    def __init__(self, config: Mapping):
        super().__init__()
        stage = config["stage_a"]
        self.seed = int(config["experiment"]["seed"])
        self.instances = int(config["train"]["instances_per_identity"])
        self.gray_epochs = int(stage["gray_epochs"])
        self.transition_epochs = int(stage["transition_epochs"])
        self.id_weight = float(stage["id_weight"])
        self.triplet_margin = float(stage["triplet_margin"])
        self.cross_weight = float(stage["cross_modal_triplet_weight"])
        self.msel_weight = float(stage["msel_weight"])
        self.msct_weight = float(stage["msct_weight"])
        self.msct = ModalitySeparatedCenterTripletLoss(
            self.instances, float(stage["msct_margin"])
        )

    def select_visible(
        self,
        rgb: torch.Tensor,
        gray: torch.Tensor,
        labels: torch.Tensor,
        *,
        epoch: int,
        iteration: int,
        iterations_per_epoch: int,
    ):
        identities = _identity_labels(labels, self.instances)
        rho = cosine_transition(
            epoch,
            iteration,
            iterations_per_epoch,
            self.gray_epochs,
            self.transition_epochs,
        )
        if rho <= 0.0:
            return gray, StageATransition(0.0, rgb.new_zeros(()))
        if rho >= 1.0:
            return rgb, StageATransition(1.0, rgb.new_ones(()))
        generator = torch.Generator(device="cpu")
        mixed_seed = (
            self.seed * 6364136223846793005
            + (int(epoch) + 1) * 1442695040888963407
            + (int(iteration) + 1) * 22695477
        ) % (2**63 - 1)
        generator.manual_seed(mixed_seed)
        identity_uses_rgb = torch.rand(identities.numel(), generator=generator) < rho
        sample_uses_rgb = identity_uses_rgb.repeat_interleave(self.instances).to(rgb.device)
        visible = torch.where(sample_uses_rgb.view(-1, 1, 1, 1), rgb, gray)
        return visible, StageATransition(
            rho,
            identity_uses_rgb.float().mean().to(rgb.device),
        )

    def forward(
        self,
        output: Mapping[str, Mapping[str, torch.Tensor]],
        labels: torch.Tensor,
        transition: StageATransition,
    ) -> Dict[str, torch.Tensor]:
        visible = output["features"]["Visible"]
        infrared = output["features"]["IR"]
        logits = output["logits"]
        identity = F.cross_entropy(logits["Visible"], labels) + F.cross_entropy(
            logits["IR"], labels
        )
        zero = visible.new_zeros(())
        rho = float(transition.rho)
        intra_raw = zero
        cross_raw = zero
        msel_raw = zero
        msct_raw = zero
        stats = {
            "msct_active_identity_ratio": zero,
            "msct_positive_center_distance": zero,
            "msct_negative_center_distance": zero,
        }
        with torch.cuda.amp.autocast(enabled=False):
            metric_visible = visible.float()
            metric_infrared = infrared.float()
            if rho < 1.0:
                intra_raw = _batch_hard_triplet(
                    metric_visible, labels, self.triplet_margin
                )
                intra_raw = intra_raw + _batch_hard_triplet(
                    metric_infrared, labels, self.triplet_margin
                )
            if rho > 0.0:
                cross_raw = _cross_modal_hard_triplet(
                    metric_visible, metric_infrared, labels, self.triplet_margin
                )
                msel_raw = _mean_sample_alignment(
                    metric_visible, metric_infrared, labels
                )
                msct_raw, stats = self.msct(
                    metric_visible,
                    metric_infrared,
                    labels,
                    return_stats=True,
                )
        triplet = (1.0 - rho) * intra_raw + rho * self.cross_weight * cross_raw
        msel = rho * self.msel_weight * msel_raw
        msct = rho * self.msct_weight * msct_raw
        total = self.id_weight * identity + triplet + msel + msct
        return {
            "total": total,
            "identity": identity.detach(),
            "triplet": triplet.detach(),
            "msel": msel.detach(),
            "msct": msct.detach(),
            "transition_rho": visible.new_tensor(rho),
            "rgb_identity_ratio": transition.rgb_identity_ratio.detach(),
            "intra_triplet_raw": intra_raw.detach(),
            "cross_triplet_raw": cross_raw.detach(),
            "msel_raw": msel_raw.detach(),
            "msct_raw": msct_raw.detach(),
            **{name: value.detach() for name, value in stats.items()},
        }
