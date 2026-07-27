from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml


class ConfigError(ValueError):
    pass


DEFAULTS: Dict[str, Any] = {
    "experiment": {"variant": "baseline", "seed": 0},
    "dataset": {
        "name": "sysu",
        "root": None,
        "sr_root": None,
        "caption_index": None,
        "vocab_path": None,
        "regdb_trial": 1,
        "regdb_direction": "visible_to_thermal",
    },
    "model": {
        "image_size": [288, 144],
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout": 0.03,
        "drop_path": 0.1,
        "text_vocab_size": 49408,
        "text_length": 77,
        "fusion_weight": 0.5,
        "freeze_vision": True,
        "vision_pretrained": None,
        "text_pretrained": None,
        "tokenizer": {
            "branches": [{"patch_size": [16, 16], "stride": [12, 12]}],
        },
    },
    "visual_enhancement": {
        "enabled": False,
        "modalities": [],
        "exact_size": True,
    },
    "text_augmentation": {
        "enabled": False,
        "plugin": "qwen_paraphrases",
        "index": None,
        "probability": 1.0,
        "strategy": "balanced_cycle",
        "strict": True,
        "validate_source_caption": True,
        "paraphrases_per_caption": 4,
        "strip_prefixes": ["datasets/sysu", "datasets/regdb"],
    },
    "loss": {
        "id_weight": 1.0,
        "wrt_weight": 1.0,
        "favta_enabled": False,
        "favta_weight": 1.0,
        "pair_weights": {
            "RGB-IR": 1.0,
            "RGB-Fusion": 1.0,
            "RGB-Text": 1.0,
            "IR-Fusion": 1.0,
            "IR-Text": 1.0,
            "Fusion-Text": 1.0,
        },
    },
    "train": {
        "epochs": 33,
        "batch_size": 32,
        "instances_per_identity": 4,
        "num_workers": 8,
        "optimizer": "adamw",
        "lr_text": 1.0e-5,
        "lr_vision": 3.0e-4,
        "lr_other": 1.0e-5,
        "weight_decay": 1.0e-4,
        "warmup_epochs": 3,
        "warmup_factor": 0.01,
        "min_lr_factor": 0.01,
        "gradient_clip_norm": 0.0,
        "amp": True,
        "resume": None,
        "output_dir": None,
    },
    "evaluation": {
        "protocol": "all",
        "gallery_trials": 10,
        "gallery_mode": "single",
        "batch_size": 64,
        "num_workers": 4,
    },
}


def _deep_merge(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _parse_scalar(raw: str) -> Any:
    return yaml.safe_load(raw)


def _apply_override(config: MutableMapping[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ConfigError("override must use dotted.path=value syntax")
    dotted, raw = expression.split("=", 1)
    cursor: MutableMapping[str, Any] = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], MutableMapping):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = _parse_scalar(raw)


def _set_dotted(config: MutableMapping[str, Any], dotted: str, value: Any) -> None:
    cursor: MutableMapping[str, Any] = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], MutableMapping):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = copy.deepcopy(value)


def load_config(
    path: str,
    overrides: Optional[Iterable[str]] = None,
    explicit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULTS)
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ConfigError("configuration root must be a mapping")
    _deep_merge(config, loaded)
    for expression in overrides or ():
        _apply_override(config, expression)
    for dotted, value in (explicit or {}).items():
        if value is not None:
            _set_dotted(config, dotted, value)
    config["dataset"]["root"] = _expand(config["dataset"].get("root"))
    config["dataset"]["sr_root"] = _expand(config["dataset"].get("sr_root"))
    config["dataset"]["caption_index"] = _expand(config["dataset"].get("caption_index"))
    config["dataset"]["vocab_path"] = _expand(config["dataset"].get("vocab_path"))
    config["text_augmentation"]["index"] = _expand(config["text_augmentation"].get("index"))
    config["train"]["output_dir"] = _expand(config["train"].get("output_dir"))
    config["train"]["resume"] = _expand(config["train"].get("resume"))
    config["model"]["vision_pretrained"] = _expand(config["model"].get("vision_pretrained"))
    config["model"]["text_pretrained"] = _expand(config["model"].get("text_pretrained"))
    validate_config(config)
    return config


def _expand(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ConfigError("an environment variable in a configured path is not set")
    return os.path.abspath(expanded)


def validate_config(config: Mapping[str, Any]) -> None:
    variant = config["experiment"]["variant"]
    if variant not in {"baseline", "favta", "visual", "full"}:
        raise ConfigError("experiment.variant must be baseline, favta, visual, or full")
    dataset = config["dataset"]["name"]
    if dataset not in {"sysu", "regdb"}:
        raise ConfigError("dataset.name must be sysu or regdb")
    enhanced = bool(config["visual_enhancement"]["enabled"])
    favta = bool(config["loss"]["favta_enabled"])
    expected = {
        "baseline": (False, False),
        "favta": (False, True),
        "visual": (True, False),
        "full": (True, True),
    }[variant]
    if (enhanced, favta) != expected:
        raise ConfigError("variant does not match visual_enhancement/favta switches")
    branches = config["model"]["tokenizer"]["branches"]
    if enhanced:
        if not config["dataset"].get("sr_root"):
            raise ConfigError("visual enhancement requires dataset.sr_root")
        if set(config["visual_enhancement"].get("modalities", [])) != {"rgb", "ir"}:
            raise ConfigError("visual enhancement requires both rgb and ir SR modalities")
        if len(branches) != 2:
            raise ConfigError("visual enhancement requires exactly two tokenizer branches")
    elif len(branches) != 1:
        raise ConfigError("baseline vision requires exactly one tokenizer branch")
    pair_weights = config["loss"].get("pair_weights", {})
    required = {"RGB-IR", "RGB-Fusion", "RGB-Text", "IR-Fusion", "IR-Text", "Fusion-Text"}
    if favta and set(pair_weights) != required:
        raise ConfigError("FAVTA requires exactly the six declared view pairs")
    if dataset == "regdb" and config["dataset"]["regdb_direction"] not in {
        "visible_to_thermal",
        "thermal_to_visible",
    }:
        raise ConfigError("unsupported RegDB direction")
    if dataset == "sysu" and config["evaluation"]["protocol"] not in {"all", "indoor"}:
        raise ConfigError("SYSU protocol must be all or indoor")
    if config["evaluation"].get("gallery_mode") not in {"single", "multi"}:
        raise ConfigError("gallery_mode must be single or multi")
    if int(config["evaluation"]["gallery_trials"]) < 1:
        raise ConfigError("gallery_trials must be positive")
    if int(config["train"]["batch_size"]) % int(config["train"]["instances_per_identity"]):
        raise ConfigError("batch_size must be divisible by instances_per_identity")
    text_augmentation = config.get("text_augmentation", {})
    if text_augmentation.get("enabled"):
        if not isinstance(text_augmentation.get("plugin"), str) or not text_augmentation["plugin"].strip():
            raise ConfigError("text augmentation requires a plugin name or module:object reference")
        if not text_augmentation.get("index"):
            raise ConfigError("text augmentation requires text_augmentation.index")
        probability = float(text_augmentation.get("probability", 1.0))
        if not 0.0 <= probability <= 1.0:
            raise ConfigError("text_augmentation.probability must be in [0, 1]")
        if int(text_augmentation.get("paraphrases_per_caption", 4)) != 4:
            raise ConfigError("text_augmentation.paraphrases_per_caption must be exactly 4")
        if text_augmentation.get("strategy", "balanced_cycle") not in {"balanced_cycle", "iid_uniform"}:
            raise ConfigError("text_augmentation.strategy must be balanced_cycle or iid_uniform")
