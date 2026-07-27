from pathlib import Path

import pytest
import torch
from PIL import Image

from favta.data.datasets import VisualPairDataset
from favta.data.records import ImageRecord
from favta.data.sampler import AutoReplaceIdentityBatchSampler
from favta.data.transforms import ImageTransform
from scripts.check_repository import violations


def test_sampler_replaces_sparse_identity_samples():
    sampler = AutoReplaceIdentityBatchSampler({0: [0], 1: [1]}, batch_size=4, instances_per_identity=2, seed=2)
    batch = next(iter(sampler))
    assert len(batch) == 4
    assert batch.count(0) == 2 and batch.count(1) == 2


def test_missing_sr_mirror_is_explicit(tmp_path):
    rgb = ImageRecord(tmp_path / "rgb.jpg", 7, 1, "rgb", Path("cam1/0007/rgb.jpg"))
    ir = ImageRecord(tmp_path / "ir.jpg", 7, 3, "ir", Path("cam3/0007/ir.jpg"))
    transform = ImageTransform([16, 8], training=False)
    dataset = VisualPairDataset(
        [rgb],
        [ir],
        transform,
        ImageTransform([16, 8], training=False, grayscale=True),
        transform,
        sr_root=str(tmp_path / "sr"),
        use_sr=True,
    )
    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_sr_size_mismatch_is_explicit(tmp_path):
    rgb = ImageRecord(tmp_path / "rgb.jpg", 7, 1, "rgb", Path("cam1/0007/rgb.jpg"))
    ir = ImageRecord(tmp_path / "ir.jpg", 7, 3, "ir", Path("cam3/0007/ir.jpg"))
    for relative in (rgb.relative_path, ir.relative_path):
        target = tmp_path / "sr" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (7, 7)).save(target)
    transform = ImageTransform([16, 8], training=False, require_input_size=True)
    dataset = VisualPairDataset(
        [rgb],
        [ir],
        transform,
        ImageTransform(
            [16, 8], training=False, require_input_size=True, grayscale=True
        ),
        transform,
        sr_root=str(tmp_path / "sr"),
        use_sr=True,
    )
    with pytest.raises(ValueError, match="input image size"):
        dataset[0]


def test_visual_pair_produces_distinct_rgb_gray_and_ir_inputs(tmp_path):
    rgb_path = tmp_path / "rgb.jpg"
    ir_path = tmp_path / "ir.jpg"
    Image.new("RGB", (8, 16), color=(220, 40, 10)).save(rgb_path)
    Image.new("RGB", (8, 16), color=(30, 30, 30)).save(ir_path)
    rgb = ImageRecord(rgb_path, 7, 1, "rgb", Path("rgb.jpg"))
    ir = ImageRecord(ir_path, 7, 3, "ir", Path("ir.jpg"))
    dataset = VisualPairDataset(
        [rgb],
        [ir],
        ImageTransform([16, 8], training=False),
        ImageTransform([16, 8], training=False, grayscale=True),
        ImageTransform([16, 8], training=False),
    )
    item = dataset[0]
    assert set(item) == {"rgb", "gray", "ir", "pid"}
    assert not item["rgb"].equal(item["gray"])
    mean = item["gray"].new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = item["gray"].new_tensor([0.229, 0.224, 0.225])[:, None, None]
    reconstructed = item["gray"] * std + mean
    assert torch.allclose(reconstructed[0], reconstructed[1])
    assert torch.allclose(reconstructed[1], reconstructed[2])


def test_repository_contains_only_allowed_sources():
    assert violations() == []
