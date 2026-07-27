import copy
import random

import numpy as np
import pytest
import torch

from favta.config.loader import DEFAULTS
from favta.engine import (
    WarmupCosineScheduler,
    build_checkpoint_provenance,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_provenance,
)


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
    values = [scheduler.factor_at(epoch) for epoch in range(2, 10)]
    assert all(left >= right for left, right in zip(values, values[1:]))
    assert values[0] == 1.0
    assert values[-1] == 0.01


def test_stage_specific_learning_rate_override_only_changes_requested_group():
    config = copy.deepcopy(DEFAULTS)
    model = TinyGroupedModel()
    optimizer = build_optimizer(config, model, {"vision": 9e-4})
    named = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert named == {
        "vision": 9e-4,
        "text": config["train"]["lr_text"],
        "other": config["train"]["lr_other"],
    }


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


def test_checkpoint_provenance_validates_trial_direction_and_model_config():
    config = copy.deepcopy(DEFAULTS)
    config["dataset"].update(
        {"name": "regdb", "regdb_trial": 3, "regdb_direction": "visible_to_thermal"}
    )
    payload = {"extra": {"provenance": build_checkpoint_provenance(config)}}
    validate_checkpoint_provenance(payload, config, "trial-3.pth")

    wrong_trial = copy.deepcopy(config)
    wrong_trial["dataset"]["regdb_trial"] = 4
    with pytest.raises(ValueError, match="regdb_trial"):
        validate_checkpoint_provenance(payload, wrong_trial, "trial-3.pth")

    wrong_fusion = copy.deepcopy(config)
    wrong_fusion["model"]["fusion_weight"] = 0.75
    with pytest.raises(ValueError, match="model configuration"):
        validate_checkpoint_provenance(payload, wrong_fusion, "trial-3.pth")

    with pytest.raises(ValueError, match="no experiment provenance"):
        validate_checkpoint_provenance({"extra": {}}, config, "legacy.pth")
