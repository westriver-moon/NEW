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

## Optional Qwen caption augmentation plugin

The core data path uses original captions unless a caption plugin is explicitly
enabled. The bundled `qwen_paraphrases` plugin consumes an external JSON index
whose values contain exactly four unique `paraphrases`. It validates complete
RGB training coverage and source-caption alignment before the first batch. The
plugin is attached only to the RGB training caption branch; IR data and all
evaluation captions remain original.

Generate the external index with the local AWQ checkpoint:

```bash
python -m venv --system-site-packages /home/cgv841/.venvs/qwen-caption
/home/cgv841/.venvs/qwen-caption/bin/pip install --no-deps autoawq==0.2.9
/home/cgv841/.venvs/qwen-caption/bin/pip install '.[qwen-caption]'
source /home/cgv841/.venvs/qwen-caption/bin/activate

favta-qwen-caption-augment \
  --input /path/to/caption_dict_Blip_RGB.json \
  --output-dir /path/to/unified-dataset/Text/Blip_RGB_Qwen3_14B_AWQ \
  --model /home/cgv841/models/Qwen3-14B-AWQ \
  --batch-size 4
```

Multi-GPU generation writes one journal and materialized JSON per shard. Merge
only after all shards are complete:

```bash
favta-qwen-caption-merge \
  --input /path/to/caption_dict_Blip_RGB.json \
  --shard-dir /path/to/unified-dataset/Text/Blip_RGB_Qwen3_14B_AWQ \
  --output /path/to/unified-dataset/Text/Blip_RGB_Qwen3_14B_AWQ/caption_qwen3_14b_awq_4x.json
```

Enable the training-validated four-caption schedule with command-line overrides:

```bash
python -m favta.cli.train --config configs/sysu/full.yaml \
  --data-root /home/cgv841/datasets/SYSU-MM01 \
  --caption-index /path/to/caption_dict_Blip_RGB.json \
  --caption-augmentation-index /home/cgv841/datasets/SYSU-MM01/Text/Blip_RGB_Qwen3_14B_AWQ/caption_qwen3_14b_awq_4x.json \
  --set text_augmentation.enabled=true \
  --set text_augmentation.plugin=qwen_paraphrases \
  --set text_augmentation.probability=1.0 \
  --set text_augmentation.strategy=balanced_cycle
```

`balanced_cycle` gives every sample a stable random starting caption and
direction. With `probability=1.0`, each sample therefore sees all four Qwen
paraphrases exactly once in every four consecutive epochs, while the marginal
choice in each epoch remains `1/4`. `iid_uniform` is also available for
independent deterministic draws. More generally, each enhanced caption has
probability `probability / 4`, while the original caption has probability
`1 - probability`. A custom plugin can be supplied as `module:object` or via
the `favta.caption_augmentation` Python entry-point group. Selection is a
stable hash of seed, epoch, sample index, and RGB image path, so resumed runs
make the same choice for the same training sample.
