from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


class CaptionTokenizer:
    def __init__(self, vocab_path: Optional[str], length: int = 77, vocab_size: int = 49408):
        self.length = int(length)
        self.vocab_size = int(vocab_size)
        self.vocab: Dict[str, int] = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        if vocab_path:
            with Path(vocab_path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    token = line.rstrip("\n")
                    if token and token not in self.vocab and len(self.vocab) < self.vocab_size:
                        self.vocab[token] = len(self.vocab)

    def __call__(self, caption: str) -> torch.Tensor:
        ids = [self.vocab["<start>"]]
        ids.extend(self.vocab.get(token.lower(), self.vocab["<unk>"]) for token in caption.split())
        ids.append(self.vocab["<end>"])
        ids = ids[: self.length]
        ids.extend([0] * (self.length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)


class CaptionIndex:
    def __init__(self, path: Optional[str]):
        self.entries: Dict[str, str] = {}
        if path:
            with Path(path).open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
                raise ValueError("caption index must be a JSON object mapping relative image paths to strings")
            self.entries = {Path(key).as_posix(): value for key, value in raw.items()}

    def caption_for(self, relative_path: Path) -> str:
        key = relative_path.as_posix()
        if key not in self.entries:
            raise KeyError("caption missing for %s" % key)
        return self.entries[key]

