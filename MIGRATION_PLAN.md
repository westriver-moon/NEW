# FAVTA Migration Reference

## Objective

Build a fresh, independent FAVTA repository from behavior-level requirements. The legacy repository is read-only reference material. No source tree, experiment artifact, dataset, generated image, pretrained parameter, checkpoint, log, report, result table, or historical record is imported.

## Included behavior

- SYSU-MM01 all-search and indoor-search evaluation with ten single-shot gallery trials.
- RegDB visible-to-thermal and thermal-to-visible evaluation with explicit trial isolation.
- Identity-balanced sampling with replacement for identities that have too few samples.
- A one-branch overlapping visual tokenizer for baseline experiments.
- Dual-modality SR path selection and two-branch overlapping tokenization for visual enhancement.
- RGB, IR, Text, and Fusion features with all six unordered pairs aligned in both directions.
- Two-stage training: cosine Gray-to-RGB visual training with modality-separated
  center alignment, followed by a frozen-vision four-view stage.
- AMP, gradient clipping, warmup-cosine scheduling, and complete strict checkpoint resume.
- An opt-in caption-augmentation plugin boundary; the Qwen plugin remains disabled unless explicitly configured.

## Excluded behavior

- Any third-dataset adapter or protocol.
- Test-identity injection and any test-pool training path.
- Region-conditioned, cyclic-interaction, metric-boost, reranking, and historical orchestration code.
- Language-augmentation logic embedded in the core dataset or model; optional plugins own generation and selection.
- Compatibility mappings for legacy checkpoint keys.
- Bundled vocabularies, data, generated SR content, or weights.

## Implementation sequence

1. Establish package structure, configuration precedence, hygiene rules, and eight ablation presets.
2. Implement SYSU-MM01 and RegDB discovery, paired training data, balanced sampling, and protocol-correct evaluation.
3. Implement the visual transformer, the fixed Stage A transition objective,
   text transformer, fusion, classifiers, WRT, and four-view alignment.
4. Implement visual pretraining, four-view training, evaluation, SR preparation, optimizer grouping, scheduler, and checkpoint lifecycle.
5. Verify all presets, protocol fixtures, gradients, CPU smoke tests, one CUDA AMP update, forbidden-name scanning, and repository artifact hygiene.

## Acceptance

The package must import without the legacy repository on `PYTHONPATH`; all tests must pass; all eight configurations must parse; Full must activate both contributions; Baseline must activate neither contribution; and tracked files must contain only source, configuration, tests, and necessary documentation.
