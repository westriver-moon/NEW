from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from favta.data.text import DEFAULT_CAPTION_PREFIXES, path_aliases
from favta.plugins.registry import CaptionAugmentationPlugin


SUPPORTED_STRATEGIES = {"iid_uniform", "balanced_cycle"}


class QwenParaphrasePlugin(CaptionAugmentationPlugin):
    """Select one of four faithful Qwen RGB paraphrases at training time."""

    def __init__(self, config: Mapping[str, Any]):
        path = config.get("index")
        if not path:
            raise ValueError("qwen_paraphrases requires an augmentation index")
        self.path = Path(str(path))
        self.probability = float(config.get("probability", 1.0))
        self.strict = bool(config.get("strict", True))
        self.expected_count = int(config.get("paraphrases_per_caption", 4))
        self.seed = int(config.get("seed", 0))
        self.strategy = str(config.get("strategy", "balanced_cycle"))
        self.validate_source_caption = bool(config.get("validate_source_caption", True))
        self.epoch = 0
        self.strip_prefixes = tuple(config.get("strip_prefixes") or DEFAULT_CAPTION_PREFIXES)
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("augmentation probability must be in [0, 1]")
        if self.expected_count != 4:
            raise ValueError("qwen_paraphrases requires exactly four paraphrases per caption")
        if self.strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                "unsupported Qwen caption strategy %r; expected one of %s"
                % (self.strategy, sorted(SUPPORTED_STRATEGIES))
            )
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("augmentation index must be a non-empty JSON object")
        self.entries: Dict[str, List[str]] = {}
        self.descriptions: Dict[str, Optional[str]] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                description = value.get("description")
                value = value.get("paraphrases")
            else:
                description = None
            if not isinstance(key, str) or not isinstance(value, list):
                raise ValueError("augmentation entries must map paths to paraphrase lists")
            if description is not None and (not isinstance(description, str) or not description.strip()):
                raise ValueError("augmentation description must be a non-empty string for %s" % key)
            paraphrases = [text.strip() for text in value if isinstance(text, str) and text.strip()]
            if len(paraphrases) != self.expected_count:
                raise ValueError("expected %d paraphrases for %s" % (self.expected_count, key))
            if len({text.casefold() for text in paraphrases}) != self.expected_count:
                raise ValueError("paraphrases must be unique for %s" % key)
            for alias in path_aliases(key, self.strip_prefixes):
                existing = self.entries.get(alias)
                if existing is not None and existing != paraphrases:
                    raise ValueError("augmentation path alias collision: %s" % alias)
                existing_description = self.descriptions.get(alias)
                if existing_description is not None and existing_description != description:
                    raise ValueError("augmentation description alias collision: %s" % alias)
                self.entries[alias] = paraphrases
                self.descriptions[alias] = description

    def _entry_for(self, key: Any) -> Tuple[List[str], Optional[str], str]:
        for alias in path_aliases(key, self.strip_prefixes):
            if alias in self.entries:
                return self.entries[alias], self.descriptions[alias], alias
        if self.strict:
            raise KeyError("augmented captions missing for %s" % key)
        aliases = path_aliases(key, self.strip_prefixes)
        return [], None, aliases[-1] if aliases else str(key)

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

    def validate_captions(self, items: Iterable[Tuple[Any, str]]) -> None:
        missing = []
        mismatched = []
        for key, original in items:
            paraphrases, description, _ = self._entry_for(key)
            if not paraphrases:
                missing.append(str(key))
                continue
            if self.validate_source_caption and description is not None and description != str(original):
                mismatched.append(str(key))
        if self.strict and missing:
            raise KeyError("augmented captions missing for %d paths; first=%s" % (len(missing), missing[0]))
        if mismatched:
            raise ValueError(
                "Qwen source caption mismatch for %d paths; first=%s"
                % (len(mismatched), mismatched[0])
            )

    def validate_tokenization(self, items, tokenizer, minimum_coverage: float):
        augmented = []
        per_caption = []
        collapsed = []
        for key, _ in items:
            paraphrases, _, _ = self._entry_for(key)
            if not paraphrases:
                continue
            augmented.extend(paraphrases)
            per_caption.extend(tokenizer.coverage([caption]) for caption in paraphrases)
            token_sequences = {tuple(tokenizer(caption).tolist()) for caption in paraphrases}
            if len(token_sequences) != len(paraphrases):
                collapsed.append(str(key))
        overall = tokenizer.coverage(augmented)
        minimum = min(per_caption) if per_caption else 1.0
        if overall < minimum_coverage or minimum < minimum_coverage:
            raise ValueError(
                "Qwen caption vocabulary coverage is too low: overall=%.3f minimum=%.3f "
                "required=%.3f" % (overall, minimum, minimum_coverage)
            )
        if collapsed:
            raise ValueError(
                "Qwen paraphrases collapse to duplicate token ID sequences for %d paths; first=%s"
                % (len(collapsed), collapsed[0])
            )
        return {"overall": overall, "minimum": minimum}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def select_caption(self, key: Any, original: str, sample_index=None) -> str:
        paraphrases, description, normalized_key = self._entry_for(key)
        if self.validate_source_caption and description is not None and description != str(original):
            raise ValueError("Qwen source caption mismatch for %s" % key)
        payload = "%d\0%d\0%s\0%s" % (self.seed, self.epoch, sample_index, normalized_key)
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
        probability_draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if not paraphrases or probability_draw >= self.probability:
            return original
        if self.strategy == "iid_uniform":
            choice = int.from_bytes(digest[8:], "big") % len(paraphrases)
        else:
            # A stable random start and direction guarantee that every training
            # sample sees all four paraphrases exactly once per four epochs when
            # probability is 1.0. Across samples, each epoch remains marginally
            # uniform over the four choices.
            base_payload = "%d\0%s\0%s" % (self.seed, sample_index, normalized_key)
            base_digest = hashlib.blake2b(base_payload.encode("utf-8"), digest_size=16).digest()
            offset = int.from_bytes(base_digest[:8], "big") % len(paraphrases)
            direction = 1 if (base_digest[8] & 1) == 0 else -1
            choice = (offset + direction * self.epoch) % len(paraphrases)
        return paraphrases[choice]
