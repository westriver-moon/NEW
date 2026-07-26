from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from favta.data.text import DEFAULT_CAPTION_PREFIXES, path_aliases
from favta.plugins.registry import CaptionAugmentationPlugin


class QwenParaphrasePlugin(CaptionAugmentationPlugin):
    """Uniformly sample one faithful Qwen paraphrase at training time."""

    def __init__(self, config: Mapping[str, Any]):
        path = config.get("index")
        if not path:
            raise ValueError("qwen_paraphrases requires an augmentation index")
        self.path = Path(str(path))
        self.probability = float(config.get("probability", 1.0))
        self.strict = bool(config.get("strict", True))
        self.expected_count = int(config.get("paraphrases_per_caption", 4))
        self.seed = int(config.get("seed", 0))
        self.epoch = 0
        self.strip_prefixes = tuple(config.get("strip_prefixes") or DEFAULT_CAPTION_PREFIXES)
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("augmentation probability must be in [0, 1]")
        if self.expected_count < 1:
            raise ValueError("paraphrases_per_caption must be positive")
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("augmentation index must be a non-empty JSON object")
        self.entries: Dict[str, List[str]] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                value = value.get("paraphrases")
            if not isinstance(key, str) or not isinstance(value, list):
                raise ValueError("augmentation entries must map paths to paraphrase lists")
            paraphrases = [text.strip() for text in value if isinstance(text, str) and text.strip()]
            if len(paraphrases) != self.expected_count:
                raise ValueError("expected %d paraphrases for %s" % (self.expected_count, key))
            if len({text.casefold() for text in paraphrases}) != self.expected_count:
                raise ValueError("paraphrases must be unique for %s" % key)
            for alias in path_aliases(key, self.strip_prefixes):
                existing = self.entries.get(alias)
                if existing is not None and existing != paraphrases:
                    raise ValueError("augmentation path alias collision: %s" % alias)
                self.entries[alias] = paraphrases

    def _paraphrases_for(self, key: Any) -> List[str]:
        for alias in path_aliases(key, self.strip_prefixes):
            if alias in self.entries:
                return self.entries[alias]
        if self.strict:
            raise KeyError("augmented captions missing for %s" % key)
        return []

    def validate_keys(self, keys: Iterable[Any]) -> None:
        if not self.strict:
            return
        missing = [
            str(key)
            for key in keys
            if not any(alias in self.entries for alias in path_aliases(key, self.strip_prefixes))
        ]
        if missing:
            raise KeyError("augmented captions missing for %d paths; first=%s" % (len(missing), missing[0]))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def select_caption(self, key: Any, original: str, sample_index=None) -> str:
        paraphrases = self._paraphrases_for(key)
        normalized_key = path_aliases(key, self.strip_prefixes)[-1]
        payload = "%d\0%d\0%s\0%s" % (self.seed, self.epoch, sample_index, normalized_key)
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
        probability_draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if not paraphrases or probability_draw >= self.probability:
            return original
        choice = int.from_bytes(digest[8:], "big") % len(paraphrases)
        return paraphrases[choice]
