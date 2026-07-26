import copy

import pytest
import torch

from favta.config.loader import DEFAULTS, validate_config
from favta.engine import build_optimizer
from favta.losses import FAVTALoss
from favta.models import build_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_amp_forward_backward_and_step():
    config = copy.deepcopy(DEFAULTS)
    config["experiment"]["variant"] = "full"
    config["dataset"]["sr_root"] = "/external/sr"
    config["visual_enhancement"] = {"enabled": True, "modalities": ["rgb", "ir"], "exact_size": True}
    config["loss"]["favta_enabled"] = True
    config["loss"]["favta_weight"] = 1.25
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
    config["model"]["tokenizer"]["branches"] = [
        {"patch_size": [8, 8], "stride": [4, 4]},
        {"patch_size": [8, 4], "stride": [4, 2]},
    ]
    config["train"].update({"batch_size": 4, "instances_per_identity": 2, "amp": True})
    validate_config(config)
    device = torch.device("cuda")
    model = build_model(config, 2).to(device)
    criterion = FAVTALoss(config).to(device)
    optimizer = build_optimizer(config, model)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    labels = torch.tensor([0, 0, 1, 1], device=device)
    text = torch.tensor(
        [[1, 4, 2, 0, 0, 0, 0, 0], [1, 5, 2, 0, 0, 0, 0, 0], [1, 6, 2, 0, 0, 0, 0, 0], [1, 7, 2, 0, 0, 0, 0, 0]],
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=True):
        output = model(
            torch.randn(4, 3, 32, 16, device=device),
            torch.randn(4, 3, 32, 16, device=device),
            text,
        )
        loss = criterion(output, labels)["total"]
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert torch.isfinite(loss)
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)
    scaler.step(optimizer)
    scaler.update()
