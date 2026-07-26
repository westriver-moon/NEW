from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


DEFAULT_CAPTION_PREFIXES = (
    "datasets/sysu",
    "datasets/regdb",
    "sysu",
    "regdb",
    "SYSU-MM01",
    "RegDB",
)


def path_aliases(value, strip_prefixes=DEFAULT_CAPTION_PREFIXES) -> Tuple[str, ...]:
    key = str(value).replace("\\", "/").lstrip("./")
    aliases = [key]
    folded = key.casefold()
    for raw_prefix in strip_prefixes or ():
        prefix = str(raw_prefix).replace("\\", "/").strip("/") + "/"
        if folded.startswith(prefix.casefold()):
            aliases.append(key[len(prefix) :])
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


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
            if not isinstance(raw, dict):
                raise ValueError("caption index must be a JSON object")
            for key, value in raw.items():
                if not isinstance(key, str):
                    raise ValueError("caption index keys must be strings")
                if isinstance(value, dict):
                    value = value.get("description")
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("caption values must be strings or objects with a description")
                for alias in path_aliases(key):
                    existing = self.entries.get(alias)
                    if existing is not None and existing != value:
                        raise ValueError("caption path alias collision: %s" % alias)
                    self.entries[alias] = value

    def caption_for(self, relative_path: Path) -> str:
        for key in path_aliases(relative_path):
            if key in self.entries:
                return self.entries[key]
        raise KeyError("caption missing for %s" % relative_path.as_posix())
