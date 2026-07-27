import copy
from pathlib import Path

import pytest
import torch

from favta.config.loader import DEFAULTS, validate_config
from favta.losses import (
    ModalitySeparatedCenterTripletLoss,
    StageALoss,
    cosine_transition,
)
from favta.models.visual_pretrain import build_visual_pretrain_model
from favta.losses.stage_a import StageATransition, _batch_hard_triplet


ROOT = Path(__file__).resolve().parents[1]


def stage_config():
    config = copy.deepcopy(DEFAULTS)
    config["model"].update(
        {
            "image_size": [32, 16],
            "embed_dim": 32,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "drop_path": 0.0,
        }
    )
    config["model"]["tokenizer"]["branches"] = [
        {"patch_size": [8, 8], "stride": [4, 4]}
    ]
    config["train"].update({"batch_size": 4, "instances_per_identity": 2})
    validate_config(config)
    return config


def test_transition_boundaries_are_zero_based_continuous_and_monotonic():
    assert cosine_transition(5, 99, 100, 6, 4) == 0.0
    assert cosine_transition(6, 0, 100, 6, 4) == 0.0
    assert cosine_transition(8, 0, 100, 6, 4) == pytest.approx(0.5)
    assert cosine_transition(10, 0, 100, 6, 4) == 1.0
    values = [
        cosine_transition(6 + step // 100, step % 100, 100, 6, 4)
        for step in range(401)
    ]
    assert all(0.0 <= value <= 1.0 for value in values)
    assert all(left <= right for left, right in zip(values, values[1:]))


def test_identity_input_selection_is_grouped_and_reproducible():
    criterion = StageALoss(stage_config())
    labels = torch.tensor([0, 0, 1, 1])
    rgb = torch.ones(4, 3, 2, 2)
    gray = torch.zeros_like(rgb)
    first, first_transition = criterion.select_visible(
        rgb, gray, labels, epoch=8, iteration=13, iterations_per_epoch=100
    )
    second, second_transition = criterion.select_visible(
        rgb, gray, labels, epoch=8, iteration=13, iterations_per_epoch=100
    )
    assert torch.equal(first, second)
    assert torch.equal(first.view(2, 2, 3, 2, 2)[:, 0], first.view(2, 2, 3, 2, 2)[:, 1])
    assert first_transition.rho == second_transition.rho
    assert torch.equal(
        first_transition.rgb_identity_ratio, second_transition.rgb_identity_ratio
    )
    all_gray, gray_transition = criterion.select_visible(
        rgb, gray, labels, epoch=5, iteration=99, iterations_per_epoch=100
    )
    all_rgb, rgb_transition = criterion.select_visible(
        rgb, gray, labels, epoch=10, iteration=0, iterations_per_epoch=100
    )
    assert torch.equal(all_gray, gray) and gray_transition.rho == 0.0
    assert torch.equal(all_rgb, rgb) and rgb_transition.rho == 1.0


def test_center_triplet_geometry_validation_and_gradients():
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        requires_grad=True,
    )
    criterion = ModalitySeparatedCenterTripletLoss(2, 0.1)
    loss, stats = criterion(
        separated, separated.clone(), labels, return_stats=True
    )
    assert loss.item() == pytest.approx(0.0)
    assert stats["msct_positive_center_distance"].item() == pytest.approx(0.0)
    close = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.9999, 0.001], [0.9999, 0.001]],
        requires_grad=True,
    )
    close_loss = criterion(close, close.clone(), labels)
    assert close_loss > loss
    close_loss.backward()
    assert torch.isfinite(close.grad).all()
    with pytest.raises(ValueError, match="contiguous"):
        criterion(separated, separated, torch.tensor([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="at least two identities"):
        criterion(separated[:2], separated[:2], torch.tensor([0, 0]))


def test_stage_a_is_the_only_visual_pretraining_objective_and_backpropagates():
    torch.manual_seed(29)
    config = stage_config()
    model = build_visual_pretrain_model(config, 2)
    criterion = StageALoss(config)
    labels = torch.tensor([0, 0, 1, 1])
    rgb = torch.randn(4, 3, 32, 16)
    gray = torch.randn(4, 3, 32, 16)
    infrared = torch.randn(4, 3, 32, 16)
    visible, transition = criterion.select_visible(
        rgb, gray, labels, epoch=8, iteration=0, iterations_per_epoch=100
    )
    output = model(visible, infrared)
    assert set(output["features"]) == {"Visible", "IR"}
    losses = criterion(output, labels, transition)
    assert set(losses) == {
        "total",
        "identity",
        "triplet",
        "msel",
        "msct",
        "transition_rho",
        "rgb_identity_ratio",
        "intra_triplet_raw",
        "cross_triplet_raw",
        "msel_raw",
        "msct_raw",
        "msct_active_identity_ratio",
        "msct_positive_center_distance",
        "msct_negative_center_distance",
    }
    losses["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert torch.isfinite(losses["total"])
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)
    source = (ROOT / "favta" / "cli" / "pretrain_visual.py").read_text(
        encoding="utf-8"
    )
    assert "WeightedRegularizedTripletLoss" not in source
    assert "wrt" not in source.lower()


def test_stage_a_intra_triplet_averages_the_two_modalities():
    criterion = StageALoss(stage_config())
    labels = torch.tensor([0, 0, 1, 1])
    visible = torch.tensor([[0.0, 0.0], [0.2, 0.0], [1.0, 0.0], [1.2, 0.0]])
    infrared = torch.tensor([[0.0, 0.0], [0.4, 0.0], [1.0, 0.0], [1.4, 0.0]])
    logits = {"Visible": torch.randn(4, 2), "IR": torch.randn(4, 2)}
    output = {"features": {"Visible": visible, "IR": infrared}, "logits": logits}
    transition = StageATransition(0.0, torch.tensor(0.0))
    losses = criterion(output, labels, transition)
    expected = 0.5 * (
        _batch_hard_triplet(visible, labels, criterion.triplet_margin)
        + _batch_hard_triplet(infrared, labels, criterion.triplet_margin)
    )
    assert losses["intra_triplet_raw"] == pytest.approx(float(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_stage_a_cuda_amp_step_is_finite():
    config = stage_config()
    device = torch.device("cuda")
    model = build_visual_pretrain_model(config, 2).to(device)
    criterion = StageALoss(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    scaler = torch.cuda.amp.GradScaler(
        enabled=True, init_scale=float(config["stage_a"]["amp_init_scale"])
    )
    labels = torch.tensor([0, 0, 1, 1], device=device)
    visible, transition = criterion.select_visible(
        torch.randn(4, 3, 32, 16, device=device),
        torch.randn(4, 3, 32, 16, device=device),
        labels,
        epoch=8,
        iteration=0,
        iterations_per_epoch=100,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True):
        output = model(visible, torch.randn(4, 3, 32, 16, device=device))
        loss = criterion(output, labels, transition)["total"]
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert torch.isfinite(loss)
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    scaler.step(optimizer)
    scaler.update()
