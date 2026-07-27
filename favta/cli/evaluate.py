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


def _num_classes(checkpoint, requested=None):
    if requested is not None:
        return int(requested)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    classifier = payload.get("model", {}).get("classifier.linear.weight")
    if classifier is None:
        raise ValueError("checkpoint does not contain the FAVTA classifier shape")
    return int(classifier.shape[0])


def _loaded_model(config, checkpoint, requested_classes, device):
    model = build_model(config, _num_classes(checkpoint, requested_classes)).to(device)
    load_checkpoint(checkpoint, model, restore_rng=False)
    return model


def _regdb_checkpoint_map(values):
    checkpoints = {}
    for expression in values:
        if "=" not in expression:
            raise ValueError("RegDB checkpoints must use TRIAL=PATH")
        raw_trial, path = expression.split("=", 1)
        trial = int(raw_trial)
        if trial < 1 or not path:
            raise ValueError("RegDB checkpoints must use a positive TRIAL and a non-empty PATH")
        if trial in checkpoints:
            raise ValueError("duplicate RegDB checkpoint for trial %d" % trial)
        if path in checkpoints.values():
            raise ValueError("each RegDB trial must use a distinct checkpoint path")
        checkpoints[trial] = path
    return checkpoints


def main(argv=None):
    parser = base_parser("Evaluate a FAVTA checkpoint")
    parser.add_argument("--checkpoint")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument(
        "--regdb-checkpoint",
        action="append",
        default=[],
        metavar="TRIAL=PATH",
        help="repeat for independently trained RegDB trial checkpoints",
    )
    args = parser.parse_args(argv)
    config = resolved_config(args)
    seed_everything(int(config["experiment"]["seed"]))
    device = torch.device(args.device)
    try:
        regdb_checkpoints = _regdb_checkpoint_map(args.regdb_checkpoint)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if config["dataset"]["name"] == "sysu":
        if not args.checkpoint:
            parser.error("--checkpoint is required for SYSU evaluation")
        if regdb_checkpoints:
            parser.error("--regdb-checkpoint is only valid for RegDB")
        shared_model = _loaded_model(config, args.checkpoint, args.num_classes, device)
        work = [
            (trial, config, args.checkpoint)
            for trial in range(int(config["evaluation"]["gallery_trials"]))
        ]
    else:
        if args.checkpoint and regdb_checkpoints:
            parser.error("use either --checkpoint for one RegDB trial or --regdb-checkpoint mappings")
        if not args.checkpoint and not regdb_checkpoints:
            parser.error("RegDB evaluation requires --checkpoint or --regdb-checkpoint TRIAL=PATH")
        if regdb_checkpoints:
            work = []
            for trial, checkpoint in sorted(regdb_checkpoints.items()):
                trial_config = copy.deepcopy(config)
                trial_config["dataset"]["regdb_trial"] = trial
                work.append((trial, trial_config, checkpoint))
        else:
            trial = int(config["dataset"]["regdb_trial"])
            work = [(trial, config, args.checkpoint)]
    results = []
    for trial, trial_config, checkpoint in work:
        model = (
            shared_model
            if trial_config["dataset"]["name"] == "sysu"
            else _loaded_model(trial_config, checkpoint, args.num_classes, device)
        )
        result = _one_trial(trial_config, model, device, trial)
        results.append(result)
        print("trial=%d Rank-1=%.6f mAP=%.6f mINP=%.6f" % (trial, result["cmc"][0], result["mAP"], result["mINP"]), flush=True)
        if trial_config["dataset"]["name"] == "regdb":
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
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
