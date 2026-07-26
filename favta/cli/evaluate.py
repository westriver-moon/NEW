from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from favta.cli.common import base_parser, resolved_config, seed_everything
from favta.data import build_evaluation_sets
from favta.engine import evaluate_features, extract_image_features, load_checkpoint
from favta.models import build_model


def _one_trial(config, model, device, trial):
    query_set, gallery_set = build_evaluation_sets(config, trial)
    evaluation = config["evaluation"]
    kwargs = {"batch_size": int(evaluation["batch_size"]), "num_workers": int(evaluation["num_workers"]), "shuffle": False}
    query_loader = DataLoader(query_set, **kwargs)
    gallery_loader = DataLoader(gallery_set, **kwargs)
    query_feature, query_pid, query_camera = extract_image_features(model, query_loader, device)
    gallery_feature, gallery_pid, gallery_camera = extract_image_features(model, gallery_loader, device)
    return evaluate_features(query_feature, gallery_feature, query_pid, gallery_pid, query_camera, gallery_camera, config["dataset"]["name"])


def main(argv=None):
    parser = base_parser("Evaluate a FAVTA checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--regdb-trials", default=None, help="comma-separated trial numbers")
    args = parser.parse_args(argv)
    config = resolved_config(args)
    seed_everything(int(config["experiment"]["seed"]))
    device = torch.device(args.device)
    num_classes = args.num_classes
    if num_classes is None:
        try:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(args.checkpoint, map_location="cpu")
        classifier = payload.get("model", {}).get("classifier.linear.weight")
        if classifier is None:
            raise ValueError("checkpoint does not contain the FAVTA classifier shape")
        num_classes = int(classifier.shape[0])
    model = build_model(config, num_classes).to(device)
    load_checkpoint(args.checkpoint, model, restore_rng=False)
    if config["dataset"]["name"] == "sysu":
        trials = range(int(config["evaluation"]["gallery_trials"]))
    else:
        trials = [int(value) for value in args.regdb_trials.split(",")] if args.regdb_trials else [int(config["dataset"]["regdb_trial"])]
    results = []
    for trial in trials:
        trial_config = copy.deepcopy(config)
        if trial_config["dataset"]["name"] == "regdb":
            trial_config["dataset"]["regdb_trial"] = trial
        result = _one_trial(trial_config, model, device, trial)
        results.append(result)
        print("trial=%d Rank-1=%.6f mAP=%.6f mINP=%.6f" % (trial, result["cmc"][0], result["mAP"], result["mINP"]), flush=True)
    print(
        "average Rank-1=%.6f mAP=%.6f mINP=%.6f"
        % (
            np.mean([item["cmc"][0] for item in results]),
            np.mean([item["mAP"] for item in results]),
            np.mean([item["mINP"] for item in results]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
