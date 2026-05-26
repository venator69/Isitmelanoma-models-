"""
Example: YOLO26 (Ultralytics) + DETR (Transformers) on CUDA inside the container.

Mount weights at runtime, e.g.:
  docker run --rm --gpus all -v /path/to/weights:/weights yolo26-detr:cuda \\
    python3 example_infer.py --image /weights/sample.jpg
"""
from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, default="", help="Path to input image")
    p.add_argument("--yolo-weights", type=str, default="yolo26n.pt", help="Ultralytics .pt path or hub name")
    p.add_argument(
        "--detr-model",
        type=str,
        default="facebook/detr-resnet-50",
        help='HF model id (use a smaller checkpoint for "mobile", e.g. same id with fewer layers is not built-in; try "facebook/detr-resnet-50-dc5" or a community distilled DETR)',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| torch:", torch.__version__, "| cuda:", torch.version.cuda)

    # --- YOLO (Ultralytics) ---
    from ultralytics import YOLO

    yolo = YOLO(args.yolo_weights)
    if args.image:
        yolo.predict(args.image, device=0 if device == "cuda" else "cpu", verbose=True)

    # --- DETR (Transformers) ---
    image = Image.open(args.image).convert("RGB") if args.image else Image.new("RGB", (640, 480), color=128)
    processor = AutoImageProcessor.from_pretrained(args.detr_model)
    model = AutoModelForObjectDetection.from_pretrained(args.detr_model).to(device)
    model.eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    print("DETR logits shape:", outputs.logits.shape)


if __name__ == "__main__":
    main()
