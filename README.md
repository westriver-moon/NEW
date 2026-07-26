# FAVTA

FAVTA is a clean research implementation for visible-infrared person re-identification. It provides two independently controlled contributions:

- dual-modality super-resolution inputs with multi-scale overlapping tokenization;
- four-view bidirectional hard-triplet alignment across RGB, IR, Text, and Fusion representations.

The repository supports SYSU-MM01 and RegDB. Datasets, generated super-resolution images, tokenizer vocabularies, pretrained parameters, checkpoints, and run outputs are external assets and are never stored in this repository.

## Variants

- `baseline`: original images, one overlapping tokenizer, ID + WRT losses.
- `favta`: baseline vision plus six-pair bidirectional alignment.
- `visual`: dual-modality SR plus two overlapping tokenizer branches, ID + WRT losses.
- `full`: visual enhancement plus six-pair bidirectional alignment.

Each variant is available under `configs/sysu` and `configs/regdb`.

## Commands

```bash
python -m favta.cli.pretrain_visual --config configs/sysu/baseline.yaml --data-root /path/to/sysu --output-dir /path/to/output
python -m favta.cli.prepare_sr --config configs/sysu/full.yaml --data-root /path/to/sysu --sr-root /path/to/sysu-sr --sr-model /path/to/model.ts --output-root /path/to/sysu-sr
python -m favta.cli.train --config configs/sysu/full.yaml --data-root /path/to/sysu --sr-root /path/to/sysu-sr --caption-index /path/to/captions.json --vocab-path /path/to/vocab.txt --output-dir /path/to/output --set model.vision_pretrained=/path/to/visual_encoder.pth
python -m favta.cli.evaluate --config configs/sysu/full.yaml --checkpoint /path/to/output/last.pth --data-root /path/to/sysu --sr-root /path/to/sysu-sr --caption-index /path/to/captions.json --vocab-path /path/to/vocab.txt
```

Explicit CLI values override YAML values. Runtime outputs belong outside the repository or under an ignored directory.
