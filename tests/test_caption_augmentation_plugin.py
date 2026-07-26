import json
from pathlib import Path

import pytest

from favta.config import ConfigError, load_config
from favta.data.text import CaptionIndex
from favta.plugins import build_caption_augmentation_plugin
from favta.plugins.qwen_caption import QwenParaphrasePlugin
from favta.plugins.qwen_caption.generate import load_input, normalize_paraphrases


ROOT = Path(__file__).resolve().parents[1]


def write_augmented_index(path: Path):
    payload = {
        "datasets/sysu/cam4/0001/frame.jpg": {
            "description": "A person in a red coat.",
            "paraphrases": ["first", "second", "third", "fourth"],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def plugin_config(path: Path, **updates):
    config = {
        "enabled": True,
        "plugin": "qwen_paraphrases",
        "index": str(path),
        "probability": 1.0,
        "strict": True,
        "paraphrases_per_caption": 4,
        "strip_prefixes": ["datasets/sysu", "datasets/regdb"],
    }
    config.update(updates)
    return config


def test_caption_index_accepts_description_objects_and_dataset_prefix(tmp_path):
    path = tmp_path / "captions.json"
    path.write_text(
        json.dumps({"datasets/sysu/cam4/0001/frame.jpg": {"description": "original text"}}),
        encoding="utf-8",
    )
    index = CaptionIndex(str(path))
    assert index.caption_for(Path("cam4/0001/frame.jpg")) == "original text"


def test_qwen_plugin_selects_paraphrases_uniformly_and_deterministically(tmp_path):
    path = tmp_path / "augmented.json"
    write_augmented_index(path)
    plugin = QwenParaphrasePlugin(plugin_config(path))
    selected = [
        plugin.select_caption(Path("cam4/0001/frame.jpg"), "original", sample_index=index)
        for index in range(4000)
    ]
    assert selected == [
        plugin.select_caption(Path("cam4/0001/frame.jpg"), "original", sample_index=index)
        for index in range(4000)
    ]
    assert set(selected) == {"first", "second", "third", "fourth"}
    assert all(850 <= selected.count(text) <= 1150 for text in set(selected))


def test_qwen_plugin_probability_zero_preserves_original_and_strictly_checks_coverage(tmp_path):
    path = tmp_path / "augmented.json"
    write_augmented_index(path)
    plugin = QwenParaphrasePlugin(plugin_config(path, probability=0.0))
    assert plugin.select_caption(Path("cam4/0001/frame.jpg"), "original", sample_index=0) == "original"
    with pytest.raises(KeyError, match="missing for 1 paths"):
        plugin.validate_keys([Path("cam4/9999/missing.jpg")])


def test_registry_builds_builtin_plugin_and_disabled_is_noop(tmp_path):
    path = tmp_path / "augmented.json"
    write_augmented_index(path)
    assert build_caption_augmentation_plugin({"enabled": False}) is None
    assert isinstance(build_caption_augmentation_plugin(plugin_config(path)), QwenParaphrasePlugin)


def test_config_expands_plugin_index_and_validates_probability(monkeypatch, tmp_path):
    monkeypatch.setenv("FAVTA_SR_ROOT", str(tmp_path / "sr"))
    config = load_config(
        str(ROOT / "configs" / "sysu" / "baseline.yaml"),
        [
            "text_augmentation.enabled=true",
            "text_augmentation.index=augmented.json",
            "text_augmentation.probability=0.75",
        ],
    )
    assert config["text_augmentation"]["index"] == str((Path.cwd() / "augmented.json").resolve())
    assert config["text_augmentation"]["probability"] == 0.75
    with pytest.raises(ConfigError, match="probability"):
        load_config(
            str(ROOT / "configs" / "sysu" / "baseline.yaml"),
            [
                "text_augmentation.enabled=true",
                "text_augmentation.index=augmented.json",
                "text_augmentation.probability=1.5",
            ],
        )


def test_generator_accepts_flat_caption_index_and_validates_four_unique_outputs(tmp_path):
    path = tmp_path / "captions.json"
    path.write_text(json.dumps({"cam4/0001/frame.jpg": "A person in red."}), encoding="utf-8")
    assert load_input(path)["cam4/0001/frame.jpg"]["description"] == "A person in red."
    output = '{"paraphrases":["one","two","three","four"]}'
    assert normalize_paraphrases(output, 2) == ["one", "two", "three", "four"]
