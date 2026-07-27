from pathlib import Path

import numpy as np
import pytest

from favta.cli import evaluate
from favta.data.protocols import discover_regdb_evaluation, discover_sysu_gallery
from favta.engine.metrics import evaluate_rank


def test_sysu_camera_filtering_removes_camera_three_to_two():
    distances = np.asarray([[0.0, 0.1, 0.2]])
    result = evaluate_rank(
        distances,
        np.asarray([1]),
        np.asarray([1, 2, 1]),
        np.asarray([3]),
        np.asarray([2, 1, 1]),
        dataset="sysu",
        max_rank=3,
    )
    assert np.allclose(result["cmc"], [0.0, 1.0, 1.0])
    assert result["mAP"] == 0.5
    assert result["mINP"] == 0.5


def test_regdb_directions_and_trials_do_not_share_records(tmp_path):
    index = tmp_path / "idx"
    index.mkdir()
    for trial, prefix in ((1, "first"), (2, "second")):
        (index / ("test_visible_%d.txt" % trial)).write_text("visible/%s.jpg 0\n" % prefix, encoding="utf-8")
        (index / ("test_thermal_%d.txt" % trial)).write_text("thermal/%s.jpg 0\n" % prefix, encoding="utf-8")
    visible, thermal = discover_regdb_evaluation(str(tmp_path), 1, "visible_to_thermal")
    reverse_query, reverse_gallery = discover_regdb_evaluation(str(tmp_path), 1, "thermal_to_visible")
    second, _ = discover_regdb_evaluation(str(tmp_path), 2, "visible_to_thermal")
    assert visible[0].modality == "rgb" and thermal[0].modality == "ir"
    assert reverse_query[0].modality == "ir" and reverse_gallery[0].modality == "rgb"
    assert visible[0].relative_path != second[0].relative_path


def test_sysu_indoor_gallery_uses_only_indoor_cameras(tmp_path):
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "test_id.txt").write_text("1", encoding="utf-8")
    for camera in (1, 2, 4, 5):
        directory = tmp_path / ("cam%d" % camera) / "0001"
        directory.mkdir(parents=True)
        (directory / "frame.jpg").write_bytes(b"not-decoded-by-discovery")
    indoor = discover_sysu_gallery(str(tmp_path), "indoor", 0)
    all_search = discover_sysu_gallery(str(tmp_path), "all", 0)
    assert {record.camera for record in indoor} == {1, 2}
    assert {record.camera for record in all_search} == {1, 2, 4, 5}


def test_sysu_multi_shot_samples_ten_images_per_identity_and_camera(tmp_path):
    (tmp_path / "exp").mkdir()
    (tmp_path / "exp" / "test_id.txt").write_text("1", encoding="utf-8")
    for camera in (1, 2):
        directory = tmp_path / ("cam%d" % camera) / "0001"
        directory.mkdir(parents=True)
        for index in range(15):
            (directory / ("frame-%02d.jpg" % index)).write_bytes(b"discovery-only")
    first = discover_sysu_gallery(str(tmp_path), "indoor", trial=3, mode="multi")
    second = discover_sysu_gallery(str(tmp_path), "indoor", trial=3, mode="multi")
    assert len(first) == 20
    assert [record.relative_path for record in first] == [record.relative_path for record in second]
    assert {camera: sum(record.camera == camera for record in first) for camera in (1, 2)} == {1: 10, 2: 10}


def test_regdb_checkpoint_mapping_loads_one_model_per_trial(monkeypatch, tmp_path, capsys):
    loaded = []
    evaluated = []
    first = tmp_path / "trial-1.pth"
    second = tmp_path / "trial-2.pth"
    first.write_bytes(b"trial-one")
    second.write_bytes(b"trial-two")

    def fake_loaded_model(config, checkpoint, requested_classes, device):
        loaded.append((config["dataset"]["regdb_trial"], checkpoint))
        return checkpoint

    def fake_one_trial(config, model, device, trial):
        evaluated.append((trial, config["dataset"]["regdb_trial"], model))
        return {"cmc": np.asarray([0.5]), "mAP": 0.4, "mINP": 0.3}

    monkeypatch.setattr(evaluate, "_loaded_model", fake_loaded_model)
    monkeypatch.setattr(evaluate, "_one_trial", fake_one_trial)
    config = Path(__file__).resolve().parents[1] / "configs" / "regdb" / "baseline.yaml"
    evaluate.main(
        [
            "--config",
            str(config),
            "--data-root",
            "/external/regdb",
            "--device",
            "cpu",
            "--allow-partial-regdb",
            "--regdb-checkpoint",
            "1=%s" % first,
            "--regdb-checkpoint",
            "2=%s" % second,
        ]
    )
    assert loaded == [(1, str(first)), (2, str(second))]
    assert evaluated == [(1, 1, str(first)), (2, 2, str(second))]
    output = capsys.readouterr().out
    assert "evaluation_mode=image-only" in output
    assert "partial average over 2 trials mode=image-only" in output


def test_regdb_benchmark_rejects_partial_or_identical_checkpoints(monkeypatch, tmp_path):
    config = Path(__file__).resolve().parents[1] / "configs" / "regdb" / "baseline.yaml"
    first = tmp_path / "trial-1.pth"
    second = tmp_path / "trial-2.pth"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    with pytest.raises(SystemExit):
        evaluate.main(
            [
                "--config",
                str(config),
                "--data-root",
                "/external/regdb",
                "--device",
                "cpu",
                "--regdb-checkpoint",
                "1=%s" % first,
            ]
        )
    with pytest.raises(ValueError, match="byte-identical"):
        evaluate._validate_regdb_checkpoint_files({1: str(first), 2: str(second)})
