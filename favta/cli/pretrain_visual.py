from __future__ import annotations

import torch
from collections import defaultdict

from favta.cli.common import base_parser, output_checkpoint, resolved_config, seed_everything, training_loader
from favta.data import build_visual_training_set
from favta.engine import WarmupCosineScheduler, build_optimizer, load_checkpoint, save_checkpoint, save_model_state
from favta.losses import StageALoss
from favta.models.visual_pretrain import build_visual_pretrain_model


def main(argv=None):
    parser = base_parser("Pretrain the shared RGB/IR visual encoder")
    args = parser.parse_args(argv)
    config = resolved_config(args)
    seed_everything(int(config["experiment"]["seed"]))
    device = torch.device(args.device)
    dataset = build_visual_training_set(config)
    loader, sampler = training_loader(dataset, config)
    model = build_visual_pretrain_model(config, dataset.num_classes).to(device)
    optimizer = build_optimizer(config, model)
    train = config["train"]
    stage = config["stage_a"]
    scheduler = WarmupCosineScheduler(
        optimizer,
        stage["epochs"],
        train["warmup_epochs"],
        train["warmup_factor"],
        train["min_lr_factor"],
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(train["amp"]) and device.type == "cuda",
        init_scale=float(stage["amp_init_scale"]),
    )
    criterion = StageALoss(config).to(device)
    start = 0
    if train.get("resume"):
        payload = load_checkpoint(train["resume"], model, optimizer, scheduler, scaler)
        start = int(payload["epoch"]) + 1
    for epoch in range(start, int(stage["epochs"])):
        sampler.set_epoch(epoch)
        scheduler.step(epoch)
        model.train()
        running = defaultdict(float)
        count = 0
        for iteration, batch in enumerate(loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            gray = batch["gray"].to(device, non_blocking=True)
            ir = batch["ir"].to(device, non_blocking=True)
            labels = batch["pid"].to(device, non_blocking=True)
            visible, transition = criterion.select_visible(
                rgb,
                gray,
                labels,
                epoch=epoch,
                iteration=iteration,
                iterations_per_epoch=len(loader),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output = model(visible, ir)
                losses = criterion(output, labels, transition)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Stage A produced a non-finite total loss")
            scaler.scale(losses["total"]).backward()
            if float(train["gradient_clip_norm"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            for name, value in losses.items():
                running[name] += float(value.detach())
            count += 1
        save_checkpoint(output_checkpoint(config, "visual_resume.pth"), epoch, model, optimizer, scheduler, scaler)
        save_model_state(output_checkpoint(config, "visual_encoder.pth"), model.vision, "visual_encoder")
        summary = " ".join(
            "%s=%.6f" % (name, value / max(1, count))
            for name, value in sorted(running.items())
        )
        print("epoch=%d %s" % (epoch, summary), flush=True)


if __name__ == "__main__":
    main()
