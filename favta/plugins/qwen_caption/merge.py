#!/usr/bin/env python3
"""Merge and verify completed Qwen caption-augmentation shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .generate import load_input, sha256_file


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-words", type=int, default=None)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    source = load_input(input_path)
    source_sha256 = sha256_file(input_path)
    merged = {}
    shard_paths = sorted(Path(args.shard_dir).glob("caption_qwen3_14b_awq_4x.shard-*.json"))
    if not shard_paths:
        raise ValueError("no completed caption shards were found")
    manifests = []
    for path in shard_paths:
        manifest_path = path.with_name(path.name.replace("caption_qwen3_14b_awq_4x.", "manifest.", 1))
        if not manifest_path.is_file():
            raise ValueError("missing shard manifest: %s" % manifest_path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict) or not manifest.get("complete"):
            raise ValueError("shard manifest is incomplete: %s" % manifest_path)
        if manifest.get("source_sha256") != source_sha256:
            raise ValueError("shard source does not match merge input: %s" % manifest_path)
        manifests.append(manifest)
        with path.open("r", encoding="utf-8") as handle:
            shard = json.load(handle)
        overlap = set(merged).intersection(shard)
        if overlap:
            raise ValueError(f"duplicate keys across shards: {sorted(overlap)[:3]}")
        merged.update(shard)
    shard_layouts = {(item.get("shard_id"), item.get("num_shards")) for item in manifests}
    expected_layout = {(index, len(manifests)) for index in range(len(manifests))}
    if shard_layouts != expected_layout:
        raise ValueError("shard manifests do not form one complete shard layout")
    shared_fields = (
        "schema_version",
        "model",
        "model_source",
        "revision",
        "prompt_version",
        "source_sha256",
        "num_shards",
        "expected_total",
        "seed",
        "generation",
    )
    reference = manifests[0]
    inconsistent = [
        field
        for field in shared_fields
        if any(item.get(field) != reference.get(field) for item in manifests[1:])
    ]
    if inconsistent:
        raise ValueError("shard manifests disagree on: %s" % ", ".join(inconsistent))
    if sum(int(item.get("expected_for_shard", -1)) for item in manifests) != len(source):
        raise ValueError("shard manifests do not cover the complete source")
    manifest_limits = {item.get("generation", {}).get("max_words") for item in manifests}
    if len(manifest_limits) != 1 or None in manifest_limits:
        raise ValueError("shard manifests disagree on generation.max_words")
    generated_max_words = int(next(iter(manifest_limits)))
    max_words = generated_max_words if args.max_words is None else int(args.max_words)
    if max_words != generated_max_words:
        raise ValueError("--max-words does not match shard generation metadata")
    missing = set(source).difference(merged)
    extra = set(merged).difference(source)
    invalid = []
    for key, value in merged.items():
        if key not in source or not isinstance(value, dict):
            invalid.append(key)
            continue
        paraphrases = value.get("paraphrases")
        valid = (
            isinstance(paraphrases, list)
            and len(paraphrases) == 4
            and all(isinstance(text, str) and text.strip() for text in paraphrases)
            and len({text.strip().casefold() for text in paraphrases}) == 4
            and all(len(text.split()) <= max_words for text in paraphrases)
            and value.get("description") == source[key].get("description")
        )
        if not valid:
            invalid.append(key)
    if missing or extra or invalid:
        raise ValueError(
            f"incomplete merge: expected={len(source)} actual={len(merged)} "
            f"missing={len(missing)} extra={len(extra)} invalid={len(invalid)}"
        )
    atomic_json(Path(args.output), {key: merged[key] for key in sorted(merged)})
    print(f"verified {len(merged)} records and {len(merged) * 4} paraphrases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
