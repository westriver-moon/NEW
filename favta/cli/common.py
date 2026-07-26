from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from favta.config import load_config
from favta.data.sampler import AutoReplaceIdentityBatchSampler


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--sr-root")
    parser.add_argument("--caption-index")
    parser.add_argument("--caption-augmentation-index")
    parser.add_argument("--vocab-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def resolved_config(args):
    explicit = {
        "dataset.root": args.data_root,
        "dataset.sr_root": args.sr_root,
        "dataset.caption_index": args.caption_index,
        "text_augmentation.index": args.caption_augmentation_index,
        "dataset.vocab_path": args.vocab_path,
        "train.output_dir": args.output_dir,
    }
    return load_config(args.config, args.overrides, explicit)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_loader(dataset, config):
    train = config["train"]
    sampler = AutoReplaceIdentityBatchSampler(
        dataset.pid_to_indices,
        int(train["batch_size"]),
        int(train["instances_per_identity"]),
        int(config["experiment"]["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(train["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


def output_checkpoint(config, name="last.pth") -> str:
    output = config["train"].get("output_dir")
    if not output:
        raise ValueError("train.output_dir or --output-dir is required")
    return str(Path(output) / name)
