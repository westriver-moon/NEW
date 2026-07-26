from pathlib import Path

import pytest

from favta.config import ConfigError, load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset", ["sysu", "regdb"])
@pytest.mark.parametrize(
    "variant,enhanced,aligned,branches,weight",
    [
        ("baseline", False, False, 1, 1.0),
        ("favta", False, True, 1, 1.0),
        ("visual", True, False, 2, 1.0),
        ("full", True, True, 2, 1.25),
    ],
)
def test_eight_configs_map_exactly(monkeypatch, tmp_path, dataset, variant, enhanced, aligned, branches, weight):
    monkeypatch.setenv("FAVTA_SR_ROOT", str(tmp_path / "sr"))
    config = load_config(str(ROOT / "configs" / dataset / (variant + ".yaml")))
    assert config["experiment"]["variant"] == variant
    assert config["visual_enhancement"]["enabled"] is enhanced
    assert config["loss"]["favta_enabled"] is aligned
    assert len(config["model"]["tokenizer"]["branches"]) == branches
    assert config["loss"]["favta_weight"] == weight


def test_precedence_is_default_then_yaml_then_set_then_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("FAVTA_SR_ROOT", str(tmp_path / "sr"))
    path = ROOT / "configs" / "sysu" / "baseline.yaml"
    config = load_config(
        str(path),
        ["train.lr_text=0.002", "train.output_dir=set-output"],
        {"train.output_dir": str(tmp_path / "explicit output")},
    )
    assert config["train"]["lr_text"] == 0.002
    assert config["train"]["output_dir"] == str((tmp_path / "explicit output").resolve())
    assert config["train"]["epochs"] == 33


def test_inconsistent_variant_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("FAVTA_SR_ROOT", str(tmp_path / "sr"))
    with pytest.raises(ConfigError):
        load_config(
            str(ROOT / "configs" / "sysu" / "baseline.yaml"),
            ["visual_enhancement.enabled=true"],
        )


def test_unresolved_environment_path_is_rejected(monkeypatch):
    monkeypatch.delenv("FAVTA_SR_ROOT", raising=False)
    with pytest.raises(ConfigError):
        load_config(str(ROOT / "configs" / "sysu" / "visual.yaml"))
