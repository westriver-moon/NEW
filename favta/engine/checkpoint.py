from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1


def _load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset(value: Optional[str]) -> Optional[Dict[str, str]]:
    if not value:
        return None
    path = Path(value)
    return {
        "path": str(path),
        "sha256": sha256_file(str(path)) if path.is_file() else "missing",
    }


def build_checkpoint_provenance(config: Mapping[str, Any]) -> Dict[str, Any]:
    dataset = config["dataset"]
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "variant": config["experiment"]["variant"],
        "dataset": {
            "name": dataset["name"],
            "regdb_trial": int(dataset["regdb_trial"]),
            "regdb_direction": dataset["regdb_direction"],
        },
        "model": copy.deepcopy(config["model"]),
        "visual_enhancement": copy.deepcopy(config["visual_enhancement"]),
        "loss": copy.deepcopy(config["loss"]),
        "evaluation": copy.deepcopy(config["evaluation"]),
        "text_augmentation": copy.deepcopy(config["text_augmentation"]),
        "assets": {
            "caption_index": _asset(dataset.get("caption_index")),
            "vocab": _asset(dataset.get("vocab_path")),
            "caption_augmentation_index": _asset(config["text_augmentation"].get("index")),
        },
    }


def _compatible_model_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(value))
    # These files initialize training only; a complete model state is already in
    # the checkpoint and evaluation should not need the source files again.
    result.pop("vision_pretrained", None)
    result.pop("text_pretrained", None)
    return result


def validate_checkpoint_provenance(
    payload: Mapping[str, Any], config: Mapping[str, Any], checkpoint: str
) -> None:
    provenance = payload.get("extra", {}).get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint has no experiment provenance: %s" % checkpoint)
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint provenance schema: %s" % checkpoint)
    expected_dataset = config["dataset"]
    expected = {
        "variant": config["experiment"]["variant"],
        "dataset.name": expected_dataset["name"],
        "dataset.regdb_trial": int(expected_dataset["regdb_trial"]),
        "dataset.regdb_direction": expected_dataset["regdb_direction"],
    }
    actual_dataset = provenance.get("dataset", {})
    actual = {
        "variant": provenance.get("variant"),
        "dataset.name": actual_dataset.get("name"),
        "dataset.regdb_trial": actual_dataset.get("regdb_trial"),
        "dataset.regdb_direction": actual_dataset.get("regdb_direction"),
    }
    mismatches = [name for name in expected if actual[name] != expected[name]]
    if mismatches:
        name = mismatches[0]
        raise ValueError(
            "checkpoint provenance mismatch for %s: stored=%r requested=%r (%s)"
            % (name, actual[name], expected[name], checkpoint)
        )
    if _compatible_model_config(provenance.get("model", {})) != _compatible_model_config(
        config["model"]
    ):
        raise ValueError("checkpoint model configuration does not match evaluation config: %s" % checkpoint)
    for section in ("visual_enhancement", "loss"):
        if provenance.get(section) != config[section]:
            raise ValueError("checkpoint %s configuration does not match: %s" % (section, checkpoint))
    for name, configured_path in (
        ("caption_index", expected_dataset.get("caption_index")),
        ("vocab", expected_dataset.get("vocab_path")),
    ):
        if not configured_path:
            continue
        stored_asset = provenance.get("assets", {}).get(name)
        current_hash = sha256_file(configured_path)
        if not isinstance(stored_asset, Mapping) or stored_asset.get("sha256") != current_hash:
            raise ValueError("checkpoint %s fingerprint does not match: %s" % (name, checkpoint))


def _rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str, epoch: int, model, optimizer=None, scheduler=None, scaler=None, extra=None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": _rng_state(),
        "extra": extra or {},
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def save_model_state(path: str, model, kind: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "kind": str(kind), "model": model.state_dict()}
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None, restore_rng: bool = True) -> Dict[str, Any]:
    payload = _load(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        _restore_rng(payload["rng"])
    return payload
