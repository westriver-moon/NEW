from __future__ import annotations

from typing import Any, Dict, Tuple

from favta.plugins import build_caption_augmentation_plugin

from .datasets import CrossModalDataset, EvaluationImageDataset, VisualPairDataset
from .protocols import (
    discover_regdb_evaluation,
    discover_regdb_training,
    discover_sysu_gallery,
    discover_sysu_query,
    discover_sysu_training,
)
from .text import CaptionIndex, CaptionTokenizer
from .transforms import ImageTransform


def build_training_set(config: Dict[str, Any]) -> CrossModalDataset:
    dataset = config["dataset"]
    if not dataset.get("root"):
        raise ValueError("dataset.root is required")
    if not dataset.get("caption_index"):
        raise ValueError("dataset.caption_index is required for four-view training")
    if not dataset.get("vocab_path"):
        raise ValueError("dataset.vocab_path is required for four-view training")
    if dataset["name"] == "sysu":
        rgb, ir = discover_sysu_training(dataset["root"])
    else:
        rgb, ir = discover_regdb_training(dataset["root"], int(dataset["regdb_trial"]))
    caption_index = CaptionIndex(dataset["caption_index"])
    tokenizer = CaptionTokenizer(
        dataset.get("vocab_path"),
        config["model"]["text_length"],
        config["model"]["text_vocab_size"],
    )
    rgb_captions = [
        caption_index.caption_for(record.relative_path or record.path.name)
        for record in rgb
    ]
    coverage = tokenizer.coverage(rgb_captions)
    minimum_coverage = float(dataset.get("min_vocab_coverage", 0.5))
    if coverage < minimum_coverage:
        raise ValueError(
            "caption vocabulary coverage %.3f is below dataset.min_vocab_coverage %.3f"
            % (coverage, minimum_coverage)
        )
    require_sr_size = bool(config["visual_enhancement"]["enabled"] and config["visual_enhancement"]["exact_size"])
    augmentation_config = dict(config.get("text_augmentation", {}))
    augmentation_config.setdefault("seed", int(config["experiment"]["seed"]))
    caption_augmentation = build_caption_augmentation_plugin(augmentation_config)
    return CrossModalDataset(
        rgb,
        ir,
        ImageTransform(config["model"]["image_size"], training=True, require_input_size=require_sr_size),
        caption_index,
        tokenizer,
        caption_augmentation=caption_augmentation,
        sr_root=dataset.get("sr_root"),
        use_sr=config["visual_enhancement"]["enabled"],
    )


def build_visual_training_set(config: Dict[str, Any]) -> VisualPairDataset:
    dataset = config["dataset"]
    if not dataset.get("root"):
        raise ValueError("dataset.root is required")
    if dataset["name"] == "sysu":
        rgb, ir = discover_sysu_training(dataset["root"])
    else:
        rgb, ir = discover_regdb_training(dataset["root"], int(dataset["regdb_trial"]))
    require_sr_size = bool(config["visual_enhancement"]["enabled"] and config["visual_enhancement"]["exact_size"])
    return VisualPairDataset(
        rgb,
        ir,
        ImageTransform(
            config["model"]["image_size"],
            training=True,
            require_input_size=require_sr_size,
        ),
        ImageTransform(
            config["model"]["image_size"],
            training=True,
            require_input_size=require_sr_size,
            grayscale=True,
        ),
        ImageTransform(
            config["model"]["image_size"],
            training=True,
            require_input_size=require_sr_size,
        ),
        sr_root=dataset.get("sr_root"),
        use_sr=config["visual_enhancement"]["enabled"],
    )


def build_evaluation_sets(config: Dict[str, Any], trial: int = 0) -> Tuple[EvaluationImageDataset, EvaluationImageDataset]:
    dataset = config["dataset"]
    if not dataset.get("root"):
        raise ValueError("dataset.root is required")
    if dataset["name"] == "sysu":
        protocol = config["evaluation"]["protocol"]
        query = discover_sysu_query(dataset["root"], protocol)
        gallery = discover_sysu_gallery(dataset["root"], protocol, trial, config["evaluation"]["gallery_mode"])
    else:
        query, gallery = discover_regdb_evaluation(
            dataset["root"], int(dataset["regdb_trial"]), dataset["regdb_direction"]
        )
    require_sr_size = bool(config["visual_enhancement"]["enabled"] and config["visual_enhancement"]["exact_size"])
    transform = ImageTransform(config["model"]["image_size"], training=False, require_input_size=require_sr_size)
    use_text_fusion = bool(config["evaluation"].get("use_text_fusion", False))
    caption_index = None
    tokenizer = None
    if use_text_fusion:
        if not dataset.get("caption_index") or not dataset.get("vocab_path"):
            raise ValueError(
                "text-fusion evaluation requires dataset.caption_index and dataset.vocab_path"
            )
        caption_index = CaptionIndex(dataset["caption_index"])
        tokenizer = CaptionTokenizer(
            dataset["vocab_path"],
            config["model"]["text_length"],
            config["model"]["text_vocab_size"],
        )
        ir_captions = [
            caption_index.caption_for(record.relative_path or record.path.name)
            for record in list(query) + list(gallery)
            if record.modality == "ir"
        ]
        coverage = tokenizer.coverage(ir_captions)
        minimum_coverage = float(dataset.get("min_vocab_coverage", 0.5))
        if coverage < minimum_coverage:
            raise ValueError(
                "evaluation caption vocabulary coverage %.3f is below dataset.min_vocab_coverage %.3f"
                % (coverage, minimum_coverage)
            )
    kwargs = {
        "sr_root": dataset.get("sr_root"),
        "use_sr": config["visual_enhancement"]["enabled"],
        "caption_index": caption_index,
        "tokenizer": tokenizer,
    }
    return EvaluationImageDataset(query, transform, **kwargs), EvaluationImageDataset(gallery, transform, **kwargs)
