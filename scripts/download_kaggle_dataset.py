#!/usr/bin/env python3
"""
Download the Kaggle skin-cancer dataset (default: mahdavi1202/skin-cancer) into a local folder.

Used inside Docker so `Datasets/` is not baked into git. Requires Kaggle API credentials:

  - Mount `~/.kaggle/kaggle.json` read-only, or
  - Pass `-e KAGGLE_USERNAME=... -e KAGGLE_KEY=...` (from Kaggle Account > Create New API Token)

Official dataset page (user-provided mirror): https://www.kaggle.com/datasets/mahdavi1202/skin-cancer
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


DEFAULT_SLUG = "mahdavi1202/skin-cancer"
DEFAULT_TARGET = "/app/Datasets/PAD-UFES-20"


def _has_dataset_marker(root: Path) -> bool:
    if not root.is_dir():
        return False
    meta = root / "metadata.csv"
    if not meta.is_file():
        return False
    # Heuristic: at least one image somewhere under root
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        if any(root.rglob(pat)):
            return True
    return False


def _resolve_inner_root(staging: Path) -> Path:
    entries = [p for p in staging.iterdir() if p.name not in (".DS_Store", "__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return staging


def _install_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        dest = dst / child.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(child), str(dest))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Kaggle skin-cancer / PAD-style dataset.")
    p.add_argument(
        "--slug",
        default=os.environ.get("KAGGLE_DATASET", DEFAULT_SLUG),
        help=f"Kaggle dataset slug (default: {DEFAULT_SLUG}).",
    )
    p.add_argument(
        "--target",
        type=Path,
        default=Path(os.environ.get("PAD_UFES20_ROOT", DEFAULT_TARGET)),
        help=f"Destination directory (default: {DEFAULT_TARGET}).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Remove existing target and re-download.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target: Path = args.target.resolve()

    if not args.force and _has_dataset_marker(target):
        print(f"Dataset already present (metadata + images): {target}", flush=True)
        return 0

    if target.exists() and not _has_dataset_marker(target) and not args.force:
        print(
            f"Incomplete or unexpected layout at {target}. Re-run with --force to replace.",
            file=sys.stderr,
        )
        return 2

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Missing package: kaggle. Install with: pip install kaggle", file=sys.stderr)
        return 1

    if args.force and target.exists():
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory(prefix="kaggle_dl_") as tmp:
        staging = Path(tmp)
        print(f"Downloading {args.slug} -> {staging} ...", flush=True)
        api.dataset_download_files(
            args.slug,
            path=str(staging),
            unzip=True,
            quiet=False,
            force=args.force,
        )
        inner = _resolve_inner_root(staging)
        print(f"Installing files into {target} ...", flush=True)
        target.mkdir(parents=True, exist_ok=True)
        _install_tree(inner, target)

    if not _has_dataset_marker(target):
        print(
            "Warning: downloaded data may use a different layout than expected "
            "(look for metadata.csv and image files under the target).",
            flush=True,
        )
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
