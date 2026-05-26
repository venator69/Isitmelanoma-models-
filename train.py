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
from pathlib import Path

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
        default="0",
        help="GPU index (e.g. 0) or cpu.",
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_path.resolve()}")

    weights = args.weights.strip() or MODEL_PRESETS[args.model]
    print("weights:", weights, "| data:", data_path)

    model = YOLO(weights)
    train_kw: dict = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "patience": args.patience,
        "resume": args.resume,
    }
    if args.name:
        train_kw["name"] = args.name

    model.train(**train_kw)


if __name__ == "__main__":
    main()
