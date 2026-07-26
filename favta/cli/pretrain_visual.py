from __future__ import annotations

import torch
import torch.nn.functional as F

from favta.cli.common import base_parser, output_checkpoint, resolved_config, seed_everything, training_loader
from favta.data import build_visual_training_set
from favta.engine import WarmupCosineScheduler, build_optimizer, load_checkpoint, save_checkpoint, save_model_state
from favta.losses import WeightedRegularizedTripletLoss
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
    scheduler = WarmupCosineScheduler(optimizer, train["epochs"], train["warmup_epochs"], train["warmup_factor"], train["min_lr_factor"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train["amp"]) and device.type == "cuda")
    wrt = WeightedRegularizedTripletLoss().to(device)
    start = 0
    if train.get("resume"):
        payload = load_checkpoint(train["resume"], model, optimizer, scheduler, scaler)
        start = int(payload["epoch"]) + 1
    for epoch in range(start, int(train["epochs"])):
        sampler.set_epoch(epoch)
        scheduler.step(epoch)
        model.train()
        running = 0.0
        count = 0
        for batch in loader:
            rgb, ir, labels = batch["rgb"].to(device), batch["ir"].to(device), batch["pid"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output = model(rgb, ir)
                identity = 0.5 * sum(F.cross_entropy(logit, labels) for logit in output["logits"].values())
                features = torch.cat((output["features"]["RGB"], output["features"]["IR"]), dim=0)
                repeated = torch.cat((labels, labels), dim=0)
                loss = identity + wrt(features, repeated)
            scaler.scale(loss).backward()
            if float(train["gradient_clip_norm"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            count += 1
        save_checkpoint(output_checkpoint(config, "visual_resume.pth"), epoch, model, optimizer, scheduler, scaler)
        save_model_state(output_checkpoint(config, "visual_encoder.pth"), model.vision, "visual_encoder")
        print("epoch=%d total=%.6f" % (epoch, running / max(1, count)), flush=True)


if __name__ == "__main__":
    main()
