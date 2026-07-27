import copy

import pytest
import torch

from favta.config.loader import DEFAULTS, validate_config
from favta.losses import FAVTALoss, FourViewBidirectionalHardTripletLoss
from favta.models import build_model
from favta.models.visual_pretrain import build_visual_pretrain_model


def tiny_config(variant="full"):
    config = copy.deepcopy(DEFAULTS)
    enhanced = variant in {"visual", "full"}
    aligned = variant in {"favta", "full"}
    config["experiment"]["variant"] = variant
    config["dataset"]["sr_root"] = "/external/sr" if enhanced else None
    config["visual_enhancement"] = {"enabled": enhanced, "modalities": ["rgb", "ir"] if enhanced else [], "exact_size": True}
    config["loss"]["favta_enabled"] = aligned
    config["loss"]["favta_weight"] = 1.25 if variant == "full" else 1.0
    config["model"].update(
        {
            "image_size": [32, 16],
            "embed_dim": 32,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "drop_path": 0.0,
            "text_vocab_size": 32,
            "text_length": 8,
            "freeze_vision": False,
        }
    )
    branches = [{"patch_size": [8, 8], "stride": [4, 4]}]
    if enhanced:
        branches.append({"patch_size": [8, 4], "stride": [4, 2]})
    config["model"]["tokenizer"]["branches"] = branches
    config["train"]["batch_size"] = 4
    config["train"]["instances_per_identity"] = 2
    validate_config(config)
    return config


def test_baseline_and_visual_tokenizer_cardinality():
    baseline = build_model(tiny_config("baseline"), 2)
    visual = build_model(tiny_config("visual"), 2)
    assert baseline.vision.tokenizer.branch_count == 1
    assert visual.vision.tokenizer.branch_count == 2


def test_visual_pretraining_checkpoint_must_match_stage_b_architecture():
    baseline = tiny_config("baseline")
    full = tiny_config("full")
    full["model"]["image_size"] = [64, 32]
    pretrained = build_visual_pretrain_model(baseline, 2)
    model = build_model(full, 2)
    with pytest.raises(RuntimeError) as caught:
        model.vision.load_state_dict(pretrained.vision.state_dict(), strict=True)
    assert "tokenizer.projections.1.weight" in str(caught.value)
    assert "size mismatch for position" in str(caught.value)


def test_frozen_vision_remains_deterministic_when_parent_enters_train_mode():
    config = tiny_config("full")
    config["model"]["dropout"] = 0.5
    config["model"]["drop_path"] = 0.5
    config["model"]["freeze_vision"] = True
    model = build_model(config, 2)
    model.train()
    assert model.training
    assert not model.vision.training
    assert not any(module.training for module in model.vision.modules())
    image = torch.randn(4, 3, 32, 16)
    with torch.no_grad():
        first = model.encode_image(image)
        second = model.encode_image(image)
    assert torch.equal(first, second)


def test_full_model_has_four_views_twelve_directional_losses_and_gradients():
    torch.manual_seed(7)
    config = tiny_config("full")
    model = build_model(config, 2)
    rgb = torch.randn(4, 3, 32, 16)
    ir = torch.randn(4, 3, 32, 16)
    text = torch.tensor([[1, 4, 2, 0, 0, 0, 0, 0], [1, 5, 2, 0, 0, 0, 0, 0], [1, 6, 2, 0, 0, 0, 0, 0], [1, 7, 2, 0, 0, 0, 0, 0]])
    labels = torch.tensor([0, 0, 1, 1])
    output = model(rgb, ir, text)
    assert set(output["features"]) == {"RGB", "IR", "Text", "Fusion"}
    criterion = FAVTALoss(config)
    losses = criterion(output, labels)
    losses["total"].backward()
    assert len(criterion.alignment.last_components) == 12
    for value in criterion.alignment.last_components.values():
        assert torch.isfinite(value)
    for feature in output["features"].values():
        assert feature.grad_fn is not None
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_missing_or_unknown_view_never_falls_back():
    loss = FourViewBidirectionalHardTripletLoss(DEFAULTS["loss"]["pair_weights"])
    labels = torch.tensor([0, 0, 1, 1])
    views = {name: torch.randn(4, 8) for name in ("RGB", "IR", "Text")}
    with pytest.raises(ValueError):
        loss(views, labels)
