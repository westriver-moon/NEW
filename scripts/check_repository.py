from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".md", ".txt", ".gitignore"}
BLOCKED_SUFFIXES = {".pth", ".pt", ".ckpt", ".npy", ".npz", ".csv", ".jsonl", ".log"}
BLOCKED_DIRECTORIES = {"data", "datasets", "outputs", "runs", "reports", "checkpoints", "weights"}
FORBIDDEN_BRANDS = ("P" + "MT", "TVI" + "LFM", "LL" + "CM")


def tracked_files():
    root_result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        check=False,
        capture_output=True,
        text=True,
    )
    is_repository_root = (
        root_result.returncode == 0
        and Path(root_result.stdout.strip()).resolve() == ROOT.resolve()
    )
    if is_repository_root and result.returncode == 0:
        return [ROOT / line for line in result.stdout.splitlines() if line]
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    ]


def violations():
    found = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            found.append("blocked artifact: %s" % relative)
        if relative.parts and relative.parts[0] in BLOCKED_DIRECTORIES:
            found.append("blocked directory: %s" % relative)
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
            text = path.read_text(encoding="utf-8")
            lowered = text.lower()
            for brand in FORBIDDEN_BRANDS:
                if brand.lower() in lowered:
                    found.append("legacy brand in %s" % relative)
    return found


def main():
    found = violations()
    if found:
        raise SystemExit("\n".join(found))
    print("repository hygiene checks passed")


if __name__ == "__main__":
    main()
