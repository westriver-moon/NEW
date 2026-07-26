from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from favta.cli.common import base_parser, resolved_config
from favta.data.protocols import IMAGE_SUFFIXES


def main(argv=None):
    parser = base_parser("Prepare mirrored dual-modality super-resolution inputs")
    parser.add_argument("--sr-model", required=True, help="external TorchScript image-to-image model")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    config = resolved_config(args)
    source = Path(config["dataset"]["root"])
    output = Path(args.output_root)
    if source.resolve() == output.resolve():
        raise ValueError("SR output root must differ from the source dataset")
    device = torch.device(args.device)
    model = torch.jit.load(args.sr_model, map_location=device).eval()
    for path in sorted(item for item in source.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES):
        relative = path.relative_to(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).to(device)
        with torch.no_grad():
            enhanced = model(tensor).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        Image.fromarray((enhanced * 255.0).round().astype(np.uint8)).save(target)
    for metadata in ("exp", "idx"):
        if (source / metadata).is_dir():
            shutil.copytree(source / metadata, output / metadata, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
