from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any, Iterable, Mapping, Optional


ENTRY_POINT_GROUP = "favta.caption_augmentation"
BUILTIN_PLUGINS = {
    "qwen_paraphrases": "favta.plugins.qwen_caption:QwenParaphrasePlugin",
}


class CaptionAugmentationPlugin(ABC):
    def set_epoch(self, epoch: int) -> None:
        return None

    def validate_keys(self, keys: Iterable[Any]) -> None:
        return None

    @abstractmethod
    def select_caption(self, key: Any, original: str, sample_index: Optional[int] = None) -> str:
        raise NotImplementedError


def _resolve_reference(reference: str):
    module_name, separator, object_name = reference.partition(":")
    if not separator or not module_name or not object_name:
        raise ValueError("plugin reference must use module:object syntax")
    return getattr(import_module(module_name), object_name)


def _entry_point(name: str):
    discovered = entry_points()
    candidates = (
        discovered.select(group=ENTRY_POINT_GROUP, name=name)
        if hasattr(discovered, "select")
        else [item for item in discovered.get(ENTRY_POINT_GROUP, ()) if item.name == name]
    )
    matches = list(candidates)
    if len(matches) != 1:
        raise ValueError("unknown or ambiguous caption augmentation plugin: %s" % name)
    return matches[0].load()


def build_caption_augmentation_plugin(config: Mapping[str, Any]) -> Optional[CaptionAugmentationPlugin]:
    if not config.get("enabled"):
        return None
    name = str(config.get("plugin", "")).strip()
    reference = BUILTIN_PLUGINS.get(name)
    factory = _resolve_reference(reference or name) if reference or ":" in name else _entry_point(name)
    plugin = factory(config)
    if not isinstance(plugin, CaptionAugmentationPlugin):
        raise TypeError("caption augmentation factory did not return a CaptionAugmentationPlugin")
    return plugin
