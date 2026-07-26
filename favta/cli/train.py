from __future__ import annotations

import torch

from favta.cli.common import base_parser, output_checkpoint, resolved_config, seed_everything, training_loader
from favta.data import build_training_set
from favta.engine import WarmupCosineScheduler, build_optimizer, load_checkpoint, save_checkpoint, train_one_epoch
from favta.losses import FAVTALoss
from favta.models import build_model


def main(argv=None):
    parser = base_parser("Train a four-view FAVTA model")
    args = parser.parse_args(argv)
    config = resolved_config(args)
    seed_everything(int(config["experiment"]["seed"]))
    device = torch.device(args.device)
    dataset = build_training_set(config)
    loader, sampler = training_loader(dataset, config)
    model = build_model(config, dataset.num_classes).to(device)
    criterion = FAVTALoss(config).to(device)
    optimizer = build_optimizer(config, model)
    train = config["train"]
    scheduler = WarmupCosineScheduler(
        optimizer,
        int(train["epochs"]),
        int(train["warmup_epochs"]),
        float(train["warmup_factor"]),
        float(train["min_lr_factor"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train["amp"]) and device.type == "cuda")
    start = 0
    if train.get("resume"):
        payload = load_checkpoint(train["resume"], model, optimizer, scheduler, scaler)
        start = int(payload["epoch"]) + 1
    for epoch in range(start, int(train["epochs"])):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        scheduler.step(epoch)
        losses = train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            scaler,
            device,
            float(train["gradient_clip_norm"]),
        )
        save_checkpoint(output_checkpoint(config), epoch, model, optimizer, scheduler, scaler, {"variant": config["experiment"]["variant"]})
        print("epoch=%d %s" % (epoch, " ".join("%s=%.6f" % item for item in sorted(losses.items()))), flush=True)


if __name__ == "__main__":
    main()
