"""
Train YOLO26 (n/s/m) or RT-DETR on a YOLO-format dataset (images + one label .txt per image).

Dataset layout (typical):
  <root>/
    data.yaml
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt   # same stem as image, YOLO normalized boxes
    labels/val/*.txt

Your `data.yaml` should point `train` / `val` at the image folders (Ultralytics resolves sibling `labels/` automatically).
"""
from __future__ import annotations

import argparse
import random
import secrets
import tempfile
from contextlib import chdir
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS

# Preset -> Ultralytics hub / local weight filename (downloaded on first use).
MODEL_PRESETS: dict[str, str] = {
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "rtdetr": "rtdetr-l.pt",
    "rtdetr-l": "rtdetr-l.pt",
    "rtdetr-x": "rtdetr-x.pt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLO26n/s/m or RT-DETR on images + YOLO labels via a dataset YAML."
    )
    p.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS.keys()),
        default="yolo26n",
        help="Architecture preset (downloads pretrained weights on first run unless --weights is set).",
    )
    p.add_argument(
        "--weights",
        default="",
        help="Optional path or hub name to override --model (e.g. /dataset/yolo26n.pt).",
    )
    p.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to dataset YAML (Ultralytics `data` arg: train/val images, class names).",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--device",
        default="auto",
        metavar="DEVICE",
        help="auto (GPU 0 if CUDA available, else cpu), a GPU index (e.g. 0), or cpu.",
    )
    p.add_argument("--project", type=str, default="runs/train")
    p.add_argument("--name", type=str, default="", help="Run name under --project (default: auto).")
    p.add_argument("--resume", action="store_true", help="Resume last interrupted run in --project/--name.")
    p.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without val improvement).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="DataLoader workers (default: Ultralytics default). Use 0 in Docker if you hit shm/bus errors.",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        metavar="F",
        help=(
            "If < 1.0, train on a random subset of that fraction of the training images "
            "(uniform sample without replacement; not the alphabetically-first slice Ultralytics uses). "
            "Default 1.0 uses the full train set."
        ),
    )
    p.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "RNG seed for picking images when --fraction < 1.0. "
            "Omit for a random seed each run (printed to the console). "
            "Pass an integer to reproduce the same subset."
        ),
    )
    return p.parse_args()


def _list_images_under_dir(d: Path) -> list[str]:
    """All image files under d (recursive), sorted for stable iteration before shuffling."""
    out: list[str] = []
    for f in d.rglob("*"):
        if not f.is_file():
            continue
        suf = f.suffix
        if len(suf) < 2:
            continue
        if suf[1:].lower() in IMG_FORMATS:
            out.append(str(f.resolve()))
    return sorted(out)


def _resolve_train_image_paths(train_spec: str | Path, dataset_root: Path) -> list[str]:
    """Match Ultralytics: train can be a folder, a .txt list, or a glob-like path."""
    p = Path(train_spec)
    if not p.is_absolute():
        p = (dataset_root / p).resolve()

    if p.is_file() and p.suffix.lower() == ".txt":
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        paths: list[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ip = Path(line)
            if not ip.is_absolute():
                ip = (p.parent / ip).resolve()
            else:
                ip = ip.resolve()
            paths.append(str(ip))
        return paths

    if p.is_dir():
        return _list_images_under_dir(p)

    raise SystemExit(f"Unsupported train path (need dir or .txt): {p}")


def _write_random_train_subset(
    train_paths: list[str], fraction: float, seed: int
) -> tuple[Path, int]:
    """Pick random subset without replacement; write one image path per line."""
    n = len(train_paths)
    if n == 0:
        raise SystemExit("No training images found for --fraction subset.")
    k = min(n, max(1, round(n * fraction)))
    rng = random.Random(seed)
    chosen = rng.sample(train_paths, k)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_train_subset.txt", delete=False, encoding="utf-8", newline="\n"
    )
    with tmp as f:
        for line in chosen:
            f.write(line + "\n")
    return Path(tmp.name), k


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    yaml_path = data_path.resolve()
    if not yaml_path.is_file():
        raise SystemExit(f"Dataset YAML not found: {yaml_path}")

    weights = args.weights.strip() or MODEL_PRESETS[args.model]
    print("weights:", weights, "| data:", yaml_path)

    dev = args.device
    if dev == "auto":
        dev = "0" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "| cuda_available:", torch.cuda.is_available())

    model = YOLO(weights)
    train_kw: dict = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": dev,
        "project": args.project,
        "patience": args.patience,
        "resume": args.resume,
    }
    if args.name:
        train_kw["name"] = args.name
    if args.workers is not None:
        train_kw["workers"] = args.workers
    elif Path("/.dockerenv").is_file():
        # Default Docker shm is small; multiprocessing workers often OOM or bus-error.
        train_kw["workers"] = 0

    frac = float(args.fraction)
    if not (0.0 < frac <= 1.0):
        raise SystemExit("--fraction must be in (0, 1], e.g. 0.25")

    # Ultralytics 8.4.x expects `data` as a path string, not a dict. `path: .` in the user's
    # YAML is resolved against cwd (/app in Docker). Write a short-lived YAML with an absolute
    # `path` and pass that file to `model.train`.
    root = yaml_path.parent.resolve()
    tmp_path: Path | None = None
    subset_txt: Path | None = None
    try:
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            raise ValueError("data.yaml must be a YAML mapping")
        cfg["path"] = str(root)
        train_key = cfg.get("train")
        if frac < 1.0:
            if not train_key:
                raise SystemExit("data.yaml must define `train` when using --fraction < 1.0")
            all_train = _resolve_train_image_paths(train_key, root)
            subset_seed = (
                int(args.subset_seed)
                if args.subset_seed is not None
                else secrets.randbelow(2**31)
            )
            subset_txt, subset_k = _write_random_train_subset(
                all_train, fraction=frac, seed=subset_seed
            )
            cfg["train"] = str(subset_txt.resolve())
            seed_note = (
                ""
                if args.subset_seed is not None
                else " (random each run; use --subset-seed to reproduce)"
            )
            print(
                f"train subset: {len(all_train)} -> {subset_k} images "
                f"(fraction={frac}, subset_seed={subset_seed}){seed_note}"
            )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False, default_flow_style=False)
            tmp_path = Path(tmp.name)
    except Exception as exc:
        print("warning: could not build patched dataset YAML:", exc)

    try:
        with chdir(root):
            train_kw["data"] = str(tmp_path.resolve()) if tmp_path else yaml_path.name
            print("dataset root:", root, "| data:", train_kw["data"])
            model.train(**train_kw)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if subset_txt is not None:
            subset_txt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
