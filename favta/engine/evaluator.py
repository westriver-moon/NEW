from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import evaluate_rank


@torch.no_grad()
def extract_image_features(model, loader, device: torch.device) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    model.eval()
    features = []
    pids = []
    cameras = []
    for batch in loader:
        image = batch["image"].to(device)
        image_feature = model.encode_image(image)
        modalities = batch.get("modality")
        if "text" in batch and modalities and all(value == "ir" for value in modalities):
            text_feature = model.encode_text(batch["text"].to(device))
            image_feature = F.normalize(model.fusion(image_feature, text_feature), dim=1)
        features.append(image_feature.cpu())
        pids.extend(batch["pid"].numpy().tolist())
        cameras.extend(batch["camera"].numpy().tolist())
    return torch.cat(features, dim=0), np.asarray(pids), np.asarray(cameras)


def evaluate_features(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_cameras: np.ndarray,
    gallery_cameras: np.ndarray,
    dataset: str,
) -> Dict[str, np.ndarray]:
    query_features = F.normalize(query_features, dim=1)
    gallery_features = F.normalize(gallery_features, dim=1)
    distances = (-query_features.mm(gallery_features.t())).cpu().numpy()
    return evaluate_rank(
        distances,
        query_pids,
        gallery_pids,
        query_cameras,
        gallery_cameras,
        dataset=dataset,
    )
