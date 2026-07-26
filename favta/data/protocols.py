from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .records import ImageRecord


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _read_ids(path: Path) -> List[int]:
    text = path.read_text(encoding="utf-8").replace("\n", ",")
    return [int(value) for value in text.split(",") if value.strip()]


def _images(directory: Path) -> List[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def discover_sysu_training(root: str) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    base = Path(root)
    exp = base / "exp"
    train_ids = set(_read_ids(exp / "train_id.txt") + _read_ids(exp / "val_id.txt"))
    rgb: List[ImageRecord] = []
    ir: List[ImageRecord] = []
    for camera in range(1, 7):
        modality = "ir" if camera in {3, 6} else "rgb"
        for pid in sorted(train_ids):
            directory = base / ("cam%d" % camera) / ("%04d" % pid)
            for path in _images(directory) if directory.is_dir() else []:
                record = ImageRecord(path, pid, camera, modality, path.relative_to(base))
                (ir if modality == "ir" else rgb).append(record)
    if not rgb or not ir:
        raise FileNotFoundError("SYSU training images were not discovered")
    return rgb, ir


def discover_sysu_query(root: str, protocol: str) -> List[ImageRecord]:
    base = Path(root)
    test_ids = _read_ids(base / "exp" / "test_id.txt")
    records: List[ImageRecord] = []
    for camera in (3, 6):
        for pid in test_ids:
            directory = base / ("cam%d" % camera) / ("%04d" % pid)
            for path in _images(directory) if directory.is_dir() else []:
                records.append(ImageRecord(path, pid, camera, "ir", path.relative_to(base)))
    return records


def discover_sysu_gallery(root: str, protocol: str, trial: int, mode: str = "single") -> List[ImageRecord]:
    base = Path(root)
    test_ids = _read_ids(base / "exp" / "test_id.txt")
    cameras = (1, 2) if protocol == "indoor" else (1, 2, 4, 5)
    rng = random.Random(int(trial))
    records: List[ImageRecord] = []
    for camera in cameras:
        for pid in test_ids:
            directory = base / ("cam%d" % camera) / ("%04d" % pid)
            candidates = _images(directory) if directory.is_dir() else []
            selected = candidates if mode == "multi" else ([rng.choice(candidates)] if candidates else [])
            for path in selected:
                records.append(ImageRecord(path, pid, camera, "rgb", path.relative_to(base)))
    return records


def _read_regdb_index(root: Path, split: str, modality: str, trial: int) -> List[ImageRecord]:
    index = root / "idx" / ("%s_%s_%d.txt" % (split, modality, trial))
    records: List[ImageRecord] = []
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        relative, raw_pid = line.rsplit(maxsplit=1)
        path = root / relative
        records.append(ImageRecord(path, int(raw_pid), 1 if modality == "visible" else 2, "rgb" if modality == "visible" else "ir", Path(relative)))
    return records


def discover_regdb_training(root: str, trial: int) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    base = Path(root)
    return _read_regdb_index(base, "train", "visible", trial), _read_regdb_index(base, "train", "thermal", trial)


def discover_regdb_evaluation(root: str, trial: int, direction: str) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    base = Path(root)
    visible = _read_regdb_index(base, "test", "visible", trial)
    thermal = _read_regdb_index(base, "test", "thermal", trial)
    return (visible, thermal) if direction == "visible_to_thermal" else (thermal, visible)

