import copy
import random

import numpy as np
import torch

from favta.config.loader import DEFAULTS
from favta.engine import WarmupCosineScheduler, build_optimizer, load_checkpoint, save_checkpoint


class TinyGroupedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = torch.nn.Linear(4, 4)
        self.text = torch.nn.Linear(4, 4)
        self.classifier = torch.nn.Linear(4, 2)

    def forward(self, value):
        return self.classifier(self.text(self.vision(value)))


def test_optimizer_groups_and_cosine_is_monotonic_after_warmup():
    config = copy.deepcopy(DEFAULTS)
    config["train"].update({"lr_vision": 1e-4, "lr_text": 2e-4, "lr_other": 3e-4})
    model = TinyGroupedModel()
    optimizer = build_optimizer(config, model)
    named = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert named == {"vision": 1e-4, "text": 2e-4, "other": 3e-4}
    scheduler = WarmupCosineScheduler(optimizer, epochs=10, warmup_epochs=2, warmup_factor=0.1, min_lr_factor=0.01)
    values = [scheduler.factor_at(epoch) for epoch in range(2, 11)]
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_checkpoint_strictly_restores_all_training_state(tmp_path):
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    config = copy.deepcopy(DEFAULTS)
    model = TinyGroupedModel()
    optimizer = build_optimizer(config, model)
    scheduler = WarmupCosineScheduler(optimizer, 10, 2, 0.1, 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step(4)
    expected_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "complete.pth"
    save_checkpoint(str(path), 4, model, optimizer, scheduler, scaler, {"marker": "fresh"})
    expected_rng = (random.random(), float(np.random.rand()), torch.rand(1))
    for parameter in model.parameters():
        parameter.data.zero_()
    scheduler.step(8)
    payload = load_checkpoint(str(path), model, optimizer, scheduler, scaler)
    actual_rng = (random.random(), float(np.random.rand()), torch.rand(1))
    assert payload["epoch"] == 4 and payload["extra"]["marker"] == "fresh"
    assert scheduler.last_epoch == 4
    assert actual_rng[0] == expected_rng[0]
    assert actual_rng[1] == expected_rng[1]
    assert torch.equal(actual_rng[2], expected_rng[2])
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected_parameters[name])
