import json
from pathlib import Path

import pytest

from favta.config import ConfigError, load_config
from favta.data.text import CaptionIndex, CaptionTokenizer
from favta.plugins import build_caption_augmentation_plugin
from favta.plugins.qwen_caption import QwenParaphrasePlugin
from favta.plugins.qwen_caption import merge
from favta.plugins.qwen_caption.generate import (
    load_input,
    normalize_paraphrases,
    sha256_file,
    validate_completed_sources,
    validate_resume_manifest,
)


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


def test_qwen_plugin_balanced_cycle_is_uniform_and_deterministic_within_epoch(tmp_path):
    path = tmp_path / "augmented.json"
    write_augmented_index(path)
    plugin = QwenParaphrasePlugin(plugin_config(path))
    selected = [
        plugin.select_caption(
            Path("cam4/0001/frame.jpg"), "A person in a red coat.", sample_index=index
        )
        for index in range(4000)
    ]
    assert selected == [
        plugin.select_caption(
            Path("cam4/0001/frame.jpg"), "A person in a red coat.", sample_index=index
        )
        for index in range(4000)
    ]
    assert set(selected) == {"first", "second", "third", "fourth"}
    assert all(850 <= selected.count(text) <= 1150 for text in set(selected))


def test_qwen_plugin_balanced_cycle_covers_all_four_per_sample_in_four_epochs(tmp_path):
    path = tmp_path / "augmented.json"
    write_augmented_index(path)
    plugin = QwenParaphrasePlugin(plugin_config(path, strategy="balanced_cycle"))
    for sample_index in range(32):
        selected = []
        for epoch in range(4):
            plugin.set_epoch(epoch)
            selected.append(
                plugin.select_caption(Path("cam4/0001/frame.jpg"), "A person in a red coat.", sample_index)
            )
        assert set(selected) == {"first", "second", "third", "fourth"}
        plugin.set_epoch(4)
        assert plugin.select_caption(
            Path("cam4/0001/frame.jpg"), "A person in a red coat.", sample_index
        ) == selected[0]


def test_qwen_plugin_iid_uniform_strategy_remains_available(tmp_path):
    path = tmp_path / "augmented.json"
    write_augmented_index(path)
    plugin = QwenParaphrasePlugin(plugin_config(path, strategy="iid_uniform"))
    selected = [
        plugin.select_caption(Path("cam4/0001/frame.jpg"), "A person in a red coat.", sample_index=index)
        for index in range(4000)
    ]
    assert set(selected) == {"first", "second", "third", "fourth"}
    assert all(850 <= selected.count(text) <= 1150 for text in set(selected))


def test_qwen_plugin_probability_zero_preserves_original_and_strictly_checks_coverage(tmp_path):
    path = tmp_path / "augmented.json"
    write_augmented_index(path)
    plugin = QwenParaphrasePlugin(plugin_config(path, probability=0.0))
    original = "A person in a red coat."
    assert plugin.select_caption(Path("cam4/0001/frame.jpg"), original, sample_index=0) == original
    with pytest.raises(KeyError, match="missing for 1 paths"):
        plugin.validate_keys([Path("cam4/9999/missing.jpg")])


def test_qwen_plugin_preflight_rejects_source_caption_mismatch(tmp_path):
    path = tmp_path / "augmented.json"
    write_augmented_index(path)
    plugin = QwenParaphrasePlugin(plugin_config(path))
    with pytest.raises(ValueError, match="source caption mismatch"):
        plugin.validate_captions([(Path("cam4/0001/frame.jpg"), "A different caption.")])


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
    assert config["text_augmentation"]["strategy"] == "balanced_cycle"
    with pytest.raises(ConfigError, match="probability"):
        load_config(
            str(ROOT / "configs" / "sysu" / "baseline.yaml"),
            [
                "text_augmentation.enabled=true",
                "text_augmentation.index=augmented.json",
                "text_augmentation.probability=1.5",
            ],
        )
    with pytest.raises(ConfigError, match="strategy"):
        load_config(
            str(ROOT / "configs" / "sysu" / "baseline.yaml"),
            [
                "text_augmentation.enabled=true",
                "text_augmentation.index=augmented.json",
                "text_augmentation.strategy=unknown",
            ],
        )


def test_generator_accepts_flat_caption_index_and_validates_four_unique_outputs(tmp_path):
    path = tmp_path / "captions.json"
    path.write_text(json.dumps({"cam4/0001/frame.jpg": "A person in red."}), encoding="utf-8")
    assert load_input(path)["cam4/0001/frame.jpg"]["description"] == "A person in red."
    output = '{"paraphrases":["one","two","three","four"]}'
    assert normalize_paraphrases(output, 2) == ["one", "two", "three", "four"]


def test_tokenizer_normalizes_vocab_case_and_reports_coverage(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("Person\nRED\ncoat\n", encoding="utf-8")
    tokenizer = CaptionTokenizer(str(vocab), length=8, vocab_size=32)
    encoded = tokenizer("PERSON red missing")
    assert encoded.tolist()[:5] == [1, 4, 5, 3, 2]
    assert tokenizer.coverage(["person RED", "coat missing"]) == 0.75


def test_tokenizer_preserves_end_token_when_caption_is_truncated(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    tokenizer = CaptionTokenizer(str(vocab), length=4, vocab_size=32)
    assert tokenizer("one two three four").tolist() == [1, 4, 5, 2]
    with pytest.raises(ValueError, match="at least 2"):
        CaptionTokenizer(str(vocab), length=1, vocab_size=32)


def test_qwen_plugin_rejects_low_coverage_and_token_id_collapse(tmp_path):
    path = tmp_path / "augmented.json"
    write_augmented_index(path)
    plugin = QwenParaphrasePlugin(plugin_config(path))
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("first\n", encoding="utf-8")
    tokenizer = CaptionTokenizer(str(vocab), length=8, vocab_size=32)
    items = [(Path("cam4/0001/frame.jpg"), "A person in a red coat.")]
    with pytest.raises(ValueError, match="coverage is too low"):
        plugin.validate_tokenization(items, tokenizer, 0.5)

    with pytest.raises(ValueError, match="duplicate token ID"):
        plugin.validate_tokenization(items, tokenizer, 0.0)


def test_merge_normalizes_flat_input_and_uses_manifest_word_limit(monkeypatch, tmp_path):
    source = tmp_path / "captions.json"
    source.write_text(json.dumps({"a.jpg": "source caption"}), encoding="utf-8")
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard = shard_dir / "caption_qwen3_14b_awq_4x.shard-000-of-001.json"
    shard.write_text(
        json.dumps(
            {
                "a.jpg": {
                    "description": "source caption",
                    "paraphrases": ["one two", "three four", "five six", "seven eight"],
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "complete": True,
        "source_sha256": sha256_file(source),
        "shard_id": 0,
        "num_shards": 1,
        "expected_for_shard": 1,
        "generation": {"max_words": 2},
    }
    (shard_dir / "manifest.shard-000-of-001.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    output = tmp_path / "merged.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "merge",
            "--input",
            str(source),
            "--shard-dir",
            str(shard_dir),
            "--output",
            str(output),
        ],
    )
    assert merge.main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["a.jpg"]["description"] == "source caption"


def test_resume_rejects_changed_generation_or_source_caption(tmp_path):
    manifest = tmp_path / "manifest.json"
    expected = {
        "schema_version": 1,
        "model": "model",
        "model_source": "source",
        "revision": "revision",
        "prompt_version": "prompt",
        "source_sha256": "first",
        "shard_id": 0,
        "num_shards": 1,
        "expected_total": 1,
        "expected_for_shard": 1,
        "seed": 7,
        "limit": None,
        "generation": {"temperature": 0.6},
    }
    manifest.write_text(json.dumps(expected), encoding="utf-8")
    validate_resume_manifest(manifest, expected, journal_exists=True)
    changed = dict(expected)
    changed["source_sha256"] = "second"
    with pytest.raises(ValueError, match="source_sha256"):
        validate_resume_manifest(manifest, changed, journal_exists=True)
    with pytest.raises(ValueError, match="source-caption mismatch"):
        validate_completed_sources(
            {"a.jpg": {"description": "old"}},
            {"a.jpg": {"description": "new"}},
        )
