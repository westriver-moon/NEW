from __future__ import annotations

import math


class WarmupCosineScheduler:
    def __init__(self, optimizer, epochs: int, warmup_epochs: int, warmup_factor: float, min_lr_factor: float):
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_factor = float(warmup_factor)
        self.min_lr_factor = float(min_lr_factor)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = -1

    def factor_at(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            progress = float(epoch + 1) / float(self.warmup_epochs)
            return self.warmup_factor + (1.0 - self.warmup_factor) * progress
        cosine_epochs = max(1, self.epochs - self.warmup_epochs)
        if cosine_epochs == 1:
            progress = 1.0
        else:
            progress = min(
                1.0,
                max(
                    0.0,
                    float(epoch - self.warmup_epochs) / float(cosine_epochs - 1),
                ),
            )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_factor + (1.0 - self.min_lr_factor) * cosine

    def step(self, epoch: int = None) -> None:
        self.last_epoch = self.last_epoch + 1 if epoch is None else int(epoch)
        factor = self.factor_at(self.last_epoch)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def state_dict(self):
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.last_epoch = int(state["last_epoch"])
        self.base_lrs = list(state["base_lrs"])

