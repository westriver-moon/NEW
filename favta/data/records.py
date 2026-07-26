from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    pid: int
    camera: int
    modality: str
    relative_path: Optional[Path] = None

    def with_root(self, root: Path) -> "ImageRecord":
        relative = self.relative_path or self.path
        return ImageRecord(root / relative, self.pid, self.camera, self.modality, relative)

