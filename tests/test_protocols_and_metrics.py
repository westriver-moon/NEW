from pathlib import Path

import numpy as np

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
