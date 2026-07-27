# FAVTA

FAVTA is a clean research implementation for visible-infrared person re-identification. It provides two independently controlled contributions:

- dual-modality super-resolution inputs with multi-scale overlapping tokenization;
- four-view bidirectional hard-triplet alignment across RGB, IR, Text, and Fusion representations.

The repository supports SYSU-MM01 and RegDB. Datasets, generated super-resolution images, tokenizer vocabularies, pretrained parameters, checkpoints, and run outputs are external assets and are never stored in this repository.

## Stage A visual training

Stage A uses one fixed objective, with no legacy-loss selector. Epochs 0--5 use
Gray--IR inputs and identity plus modality-internal batch-hard triplet learning.
Epochs 6--9 use one iteration-level cosine coefficient to increase both the
identity-level RGB sampling probability and the cross-modal loss weight. Epoch
10 onward uses RGB--IR with identity classification, bidirectional cross-modal
batch-hard triplet, mean sample alignment, and modality-separated center
triplet losses. The default Stage A budget is 24 epochs.

RGB and Gray tensors are independently transformed and never blended. During
the transition, all visible samples of one identity choose the same input type.
The choice is deterministic from the experiment seed, epoch, iteration, and
identity slot, so an epoch-boundary resume reproduces it exactly.
Stage A AMP starts with a conservative scale of 2048 by default.

## Variants

- `baseline`: original images, one overlapping tokenizer, ID + WRT losses.
- `favta`: baseline vision plus six-pair bidirectional alignment.
- `visual`: dual-modality SR plus two overlapping tokenizer branches, ID + WRT losses.
- `full`: visual enhancement plus six-pair bidirectional alignment.

Each variant is available under `configs/sysu` and `configs/regdb`.

## Commands

```bash
python -m favta.cli.prepare_sr --config configs/sysu/full.yaml --data-root /path/to/sysu --sr-root /path/to/sysu-sr --sr-model /path/to/model.ts --output-root /path/to/sysu-sr
python -m favta.cli.pretrain_visual --config configs/sysu/full.yaml --data-root /path/to/sysu --sr-root /path/to/sysu-sr --output-dir /path/to/output
python -m favta.cli.train --config configs/sysu/full.yaml --data-root /path/to/sysu --sr-root /path/to/sysu-sr --caption-index /path/to/captions.json --vocab-path /path/to/vocab.txt --output-dir /path/to/output --set model.vision_pretrained=/path/to/visual_encoder.pth
python -m favta.cli.evaluate --config configs/sysu/full.yaml --checkpoint /path/to/output/last.pth --data-root /path/to/sysu --sr-root /path/to/sysu-sr
```

The Stage A command always uses the objective above. Its schedule and weights
are configured under `stage_a`; there is no switch back to the superseded
visual-pretraining loss. Stage A and Stage B must use the same variant config:
`baseline` and `favta` use the one-branch 288x144 encoder, while `visual` and
`full` use the two-branch 512x256 encoder. Visual checkpoints are loaded
strictly; checkpoints are not adapted across these architectures.

Evaluation is image-only by default. To evaluate IR/Text fusion, explicitly add
`--set evaluation.use_text_fusion=true` together with `--caption-index` and
`--vocab-path`. The evaluation caption index must cover every IR/thermal test
image; an RGB-only caption file is intentionally rejected during evaluation
preflight. Four-view training always requires a vocabulary and rejects caption
sets whose known-token coverage is below `dataset.min_vocab_coverage`.

RegDB benchmark averaging requires an independently trained checkpoint for
each split. Supply mappings instead of testing one checkpoint on several
splits:

```bash
python -m favta.cli.evaluate --config configs/regdb/full.yaml \
  --data-root /path/to/regdb --sr-root /path/to/regdb-sr \
  --regdb-checkpoint 1=/path/to/trial-1.pth \
  --regdb-checkpoint 2=/path/to/trial-2.pth
```

SYSU single-shot and multi-shot gallery construction follows the official
one/ten-images-per-identity-per-RGB-camera protocol and averages the configured
number of gallery trials.

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
