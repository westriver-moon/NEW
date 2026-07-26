from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _average_precision(matches: np.ndarray) -> float:
    relevant = int(matches.sum())
    if relevant == 0:
        return 0.0
    precision = matches.cumsum() / (np.arange(len(matches)) + 1.0)
    return float((precision * matches).sum() / relevant)


def evaluate_rank(
    distances: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_cameras: Optional[np.ndarray] = None,
    gallery_cameras: Optional[np.ndarray] = None,
    dataset: str = "regdb",
    max_rank: int = 20,
) -> Dict[str, np.ndarray]:
    if distances.shape != (len(query_pids), len(gallery_pids)):
        raise ValueError("distance matrix shape does not match pid arrays")
    max_rank = min(int(max_rank), len(gallery_pids))
    cmcs = []
    aps = []
    inps = []
    order = np.argsort(distances, axis=1)
    for index, ranking in enumerate(order):
        ranked_pids = gallery_pids[ranking]
        keep = np.ones(len(ranking), dtype=bool)
        if dataset == "sysu":
            if query_cameras is None or gallery_cameras is None:
                raise ValueError("SYSU evaluation requires camera ids")
            ranked_cameras = gallery_cameras[ranking]
            keep &= ~((query_pids[index] == ranked_pids) & (query_cameras[index] == ranked_cameras))
            keep &= ~((query_cameras[index] == 3) & (ranked_cameras == 2))
        ranking = ranking[keep]
        ranked_pids = gallery_pids[ranking]
        raw_matches = ranked_pids == query_pids[index]
        if not raw_matches.any():
            continue
        aps.append(_average_precision(raw_matches.astype(np.float64)))
        positive_positions = np.flatnonzero(raw_matches)
        inps.append(float(raw_matches.cumsum()[positive_positions[-1]] / (positive_positions[-1] + 1.0)))
        if dataset == "sysu":
            _, first = np.unique(ranked_pids, return_index=True)
            unique_matches = raw_matches[np.sort(first)]
        else:
            unique_matches = raw_matches
        cmc = unique_matches.cumsum()
        cmc[cmc > 1] = 1
        padded = np.zeros(max_rank, dtype=np.float64)
        padded[: min(max_rank, len(cmc))] = cmc[:max_rank]
        if len(cmc) and len(cmc) < max_rank:
            padded[len(cmc) :] = cmc[-1]
        cmcs.append(padded)
    if not cmcs:
        raise RuntimeError("no valid query has a gallery match")
    return {
        "cmc": np.mean(cmcs, axis=0),
        "mAP": float(np.mean(aps)),
        "mINP": float(np.mean(inps)),
    }

