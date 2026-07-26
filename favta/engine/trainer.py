from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, gradient_clip_norm: float = 0.0) -> Dict[str, float]:
    model.train()
    totals = defaultdict(float)
    batches = 0
    amp_enabled = scaler is not None and device.type == "cuda"
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        ir = batch["ir"].to(device, non_blocking=True)
        text = batch["text"].to(device, non_blocking=True)
        labels = batch["pid"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(rgb, ir, text)
            losses = criterion(output, labels)
        if scaler is not None:
            scaler.scale(losses["total"]).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        for name, value in losses.items():
            totals[name] += float(value.detach())
        batches += 1
    if batches == 0:
        raise RuntimeError("training loader produced no batches")
    return {name: value / batches for name, value in totals.items()}
