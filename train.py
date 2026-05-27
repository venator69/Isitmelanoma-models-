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
import tempfile
from contextlib import chdir
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

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
    return p.parse_args()


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

    # Ultralytics 8.4.x expects `data` as a path string, not a dict. `path: .` in the user's
    # YAML is resolved against cwd (/app in Docker). Write a short-lived YAML with an absolute
    # `path` and pass that file to `model.train`.
    root = yaml_path.parent.resolve()
    tmp_path: Path | None = None
    try:
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            raise ValueError("data.yaml must be a YAML mapping")
        cfg["path"] = str(root)
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


if __name__ == "__main__":
    main()
