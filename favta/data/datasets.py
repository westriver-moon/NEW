from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .records import ImageRecord
from .text import CaptionIndex, CaptionTokenizer


class CrossModalDataset(Dataset):
    def __init__(
        self,
        rgb_records: Sequence[ImageRecord],
        ir_records: Sequence[ImageRecord],
        transform,
        caption_index: CaptionIndex,
        tokenizer: CaptionTokenizer,
        caption_augmentation=None,
        sr_root: Optional[str] = None,
        use_sr: bool = False,
    ):
        self.transform = transform
        self.caption_index = caption_index
        self.tokenizer = tokenizer
        self.caption_augmentation = caption_augmentation
        self.sr_root = Path(sr_root) if sr_root else None
        self.use_sr = bool(use_sr)
        rgb_by_pid: Dict[int, List[ImageRecord]] = defaultdict(list)
        ir_by_pid: Dict[int, List[ImageRecord]] = defaultdict(list)
        for record in rgb_records:
            rgb_by_pid[record.pid].append(record)
        for record in ir_records:
            ir_by_pid[record.pid].append(record)
        if set(rgb_by_pid) != set(ir_by_pid):
            raise ValueError("RGB and IR training identities must match")
        original_pids = sorted(rgb_by_pid)
        self.pid_map = {pid: index for index, pid in enumerate(original_pids)}
        self.records = []
        self.pid_to_indices: Dict[int, List[int]] = defaultdict(list)
        for original_pid in original_pids:
            rgb_items = sorted(rgb_by_pid[original_pid], key=lambda item: str(item.path))
            ir_items = sorted(ir_by_pid[original_pid], key=lambda item: str(item.path))
            count = max(len(rgb_items), len(ir_items))
            for offset in range(count):
                index = len(self.records)
                pid = self.pid_map[original_pid]
                self.records.append((rgb_items[offset % len(rgb_items)], ir_items[offset % len(ir_items)], pid))
                self.pid_to_indices[pid].append(index)
        if self.caption_augmentation is not None:
            relative_paths = {
                record.relative_path or Path(record.path.name)
                for record in rgb_records
            }
            self.caption_augmentation.validate_keys(relative_paths)

    @property
    def num_classes(self) -> int:
        return len(self.pid_to_indices)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if self.caption_augmentation is not None:
            self.caption_augmentation.set_epoch(epoch)

    def _resolved(self, record: ImageRecord) -> Path:
        if not self.use_sr:
            return record.path
        if self.sr_root is None or record.relative_path is None:
            raise ValueError("SR input requires a mirror root and relative paths")
        path = self.sr_root / record.relative_path
        if not path.is_file():
            raise FileNotFoundError("missing SR image: %s" % path)
        return path

    def __getitem__(self, index: int):
        rgb_record, ir_record, pid = self.records[index]
        with Image.open(self._resolved(rgb_record)) as image:
            rgb = self.transform(image)
        with Image.open(self._resolved(ir_record)) as image:
            ir = self.transform(image)
        relative = rgb_record.relative_path or Path(rgb_record.path.name)
        caption = self.caption_index.caption_for(relative)
        if self.caption_augmentation is not None:
            caption = self.caption_augmentation.select_caption(relative, caption, sample_index=index)
        text = self.tokenizer(caption)
        return {"rgb": rgb, "ir": ir, "text": text, "pid": torch.tensor(pid, dtype=torch.long)}


class VisualPairDataset(Dataset):
    def __init__(self, rgb_records, ir_records, transform, sr_root=None, use_sr=False):
        from collections import defaultdict

        self.transform = transform
        self.sr_root = Path(sr_root) if sr_root else None
        self.use_sr = bool(use_sr)
        grouped_rgb = defaultdict(list)
        grouped_ir = defaultdict(list)
        for record in rgb_records:
            grouped_rgb[record.pid].append(record)
        for record in ir_records:
            grouped_ir[record.pid].append(record)
        if set(grouped_rgb) != set(grouped_ir):
            raise ValueError("RGB and IR training identities must match")
        self.records = []
        self.pid_to_indices = defaultdict(list)
        self.pid_map = {pid: index for index, pid in enumerate(sorted(grouped_rgb))}
        for source_pid in sorted(grouped_rgb):
            rgb_items = sorted(grouped_rgb[source_pid], key=lambda item: str(item.path))
            ir_items = sorted(grouped_ir[source_pid], key=lambda item: str(item.path))
            for offset in range(max(len(rgb_items), len(ir_items))):
                pid = self.pid_map[source_pid]
                index = len(self.records)
                self.records.append((rgb_items[offset % len(rgb_items)], ir_items[offset % len(ir_items)], pid))
                self.pid_to_indices[pid].append(index)

    @property
    def num_classes(self):
        return len(self.pid_to_indices)

    def __len__(self):
        return len(self.records)

    def _path(self, record):
        if not self.use_sr:
            return record.path
        if self.sr_root is None or record.relative_path is None:
            raise ValueError("SR input requires relative paths")
        path = self.sr_root / record.relative_path
        if not path.is_file():
            raise FileNotFoundError("missing SR image: %s" % path)
        return path

    def __getitem__(self, index):
        rgb_record, ir_record, pid = self.records[index]
        with Image.open(self._path(rgb_record)) as image:
            rgb = self.transform(image)
        with Image.open(self._path(ir_record)) as image:
            ir = self.transform(image)
        return {"rgb": rgb, "ir": ir, "pid": torch.tensor(pid, dtype=torch.long)}


class EvaluationImageDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform,
        sr_root: Optional[str] = None,
        use_sr: bool = False,
        caption_index: Optional[CaptionIndex] = None,
        tokenizer: Optional[CaptionTokenizer] = None,
    ):
        self.records = list(records)
        self.transform = transform
        self.sr_root = Path(sr_root) if sr_root else None
        self.use_sr = bool(use_sr)
        self.caption_index = caption_index
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = record.path
        if self.use_sr:
            if self.sr_root is None or record.relative_path is None:
                raise ValueError("SR evaluation requires relative paths")
            path = self.sr_root / record.relative_path
            if not path.is_file():
                raise FileNotFoundError("missing SR image: %s" % path)
        with Image.open(path) as image:
            tensor = self.transform(image)
        item = {
            "image": tensor,
            "pid": torch.tensor(record.pid, dtype=torch.long),
            "camera": torch.tensor(record.camera, dtype=torch.long),
            "modality": record.modality,
            "path": str(path),
        }
        if self.caption_index is not None and self.tokenizer is not None:
            relative = record.relative_path or Path(record.path.name)
            item["text"] = self.tokenizer(self.caption_index.caption_for(relative))
        return item
