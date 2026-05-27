#!/usr/bin/env python3
"""
Convert common annotation formats into an Ultralytics-ready YOLO dataset.

Output layout:
  output_dataset/
    images/train|val|test/
    labels/train|val|test/
    data.yaml

The generated labels are plain YOLO TXT files, so the output can be used by
YOLO26 and RT-DETR through Ultralytics with the same data.yaml.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class BBox:
    """YOLO-normalized bounding box."""

    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class Sample:
    image_path: Path
    boxes: tuple[BBox, ...]


class ProgressBar:
    def __init__(self, total: int, label: str) -> None:
        self.total = max(total, 1)
        self.label = label
        self.current = 0
        self._last_percent = -1

    def update(self, step: int = 1) -> None:
        self.current += step
        percent = int((self.current / self.total) * 100)
        if percent == self._last_percent and self.current != self.total:
            return
        self._last_percent = percent
        filled = min(30, int(30 * self.current / self.total))
        bar = "#" * filled + "-" * (30 - filled)
        print(f"\r{self.label}: [{bar}] {self.current}/{self.total} ({percent:3d}%)", end="")
        if self.current >= self.total:
            print()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.size
    except Exception as exc:  # noqa: BLE001 - validation should report any image read error.
        logging.warning("Corrupt/unreadable image skipped: %s (%s)", path, exc)
        return None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def box_from_xyxy(class_name: str, xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int) -> BBox | None:
    if width <= 0 or height <= 0 or xmax <= xmin or ymax <= ymin:
        return None
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    if box_width <= 0 or box_height <= 0:
        return None
    return BBox(class_name, clamp(x_center), clamp(y_center), clamp(box_width), clamp(box_height))


def full_image_box(class_name: str) -> BBox:
    """Use the whole image as one object when a dataset has image-level labels only."""
    return BBox(class_name, 0.5, 0.5, 1.0, 1.0)


def parse_class_override(value: str) -> list[str]:
    if not value:
        return []
    path = Path(value)
    if path.is_file():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_simple_data_yaml(path: Path) -> tuple[Path | None, list[str]]:
    """Small YAML reader for Ultralytics data.yaml without adding a PyYAML dependency."""
    if not path.is_file():
        return None, []
    base_path: Path | None = None
    names: list[str] = []
    in_names = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        stripped = line.strip()
        if stripped.startswith("path:"):
            value = stripped.split(":", 1)[1].strip().strip("'\"")
            base_path = Path(value) if value else None
        elif stripped.startswith("names:"):
            in_names = True
            tail = stripped.split(":", 1)[1].strip()
            if tail.startswith("[") and tail.endswith("]"):
                names = [p.strip().strip("'\"") for p in tail[1:-1].split(",") if p.strip()]
        elif in_names and ":" in stripped:
            _, name = stripped.split(":", 1)
            names.append(name.strip().strip("'\""))
        elif not raw.startswith((" ", "\t", "-")):
            in_names = False
    return base_path, names


def detect_format(input_dir: Path, args: argparse.Namespace) -> str:
    if args.format != "auto":
        return args.format
    if (input_dir / "metadata.csv").is_file():
        return "pad-ufes"
    if args.coco_json or any(input_dir.rglob("*.json")):
        for json_path in ([args.coco_json] if args.coco_json else input_dir.rglob("*.json")):
            if not json_path:
                continue
            try:
                data = json.loads(Path(json_path).read_text(encoding="utf-8"))
            except Exception:
                continue
            if {"images", "annotations", "categories"}.issubset(data):
                return "coco"
    if any(input_dir.rglob("*.xml")):
        return "voc"
    if any(input_dir.rglob("*.txt")):
        return "yolo"
    if any(input_dir.rglob("*.json")):
        return "labelme"
    raise SystemExit("Could not auto-detect dataset format. Pass --format explicitly.")


def load_pad_ufes(input_dir: Path, args: argparse.Namespace) -> tuple[list[Sample], list[str]]:
    metadata = args.metadata_csv or input_dir / "metadata.csv"
    if not metadata.is_file():
        raise SystemExit(f"PAD-UFES metadata not found: {metadata}")

    image_map = {p.name: p for p in find_images(input_dir)}
    samples: list[Sample] = []
    classes: list[str] = []
    seen_classes: set[str] = set()

    with metadata.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    progress = ProgressBar(len(rows), "Reading PAD-UFES")
    for row in rows:
        progress.update()
        image_name = (row.get("img_id") or "").strip()
        class_name = (row.get("diagnostic") or "").strip()
        image_path = image_map.get(image_name)
        if not image_name or not class_name or image_path is None:
            logging.warning("PAD-UFES row skipped; missing img_id/diagnostic/image: %s", row)
            continue
        if class_name not in seen_classes:
            seen_classes.add(class_name)
            classes.append(class_name)
        samples.append(Sample(image_path=image_path, boxes=(full_image_box(class_name),)))

    logging.info("PAD-UFES uses image-level labels; generated one full-image box per sample.")
    return samples, classes


def load_coco(input_dir: Path, args: argparse.Namespace) -> tuple[list[Sample], list[str]]:
    json_path = args.coco_json or next(input_dir.rglob("*.json"), None)
    if json_path is None or not Path(json_path).is_file():
        raise SystemExit("COCO JSON not found. Pass --coco-json.")

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]) for cat in data.get("categories", [])}
    classes = [categories[key] for key in sorted(categories)]
    image_root = args.image_dir or input_dir
    image_by_id: dict[int, dict] = {int(img["id"]): img for img in data.get("images", [])}
    ann_by_image: dict[int, list[dict]] = {image_id: [] for image_id in image_by_id}
    for ann in data.get("annotations", []):
        ann_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    samples: list[Sample] = []
    progress = ProgressBar(len(image_by_id), "Reading COCO")
    for image_id, image_info in image_by_id.items():
        progress.update()
        image_path = Path(image_info.get("file_name", ""))
        if not image_path.is_absolute():
            candidate = Path(image_root) / image_path
            image_path = candidate if candidate.is_file() else next(input_dir.rglob(image_path.name), candidate)
        width = int(image_info.get("width") or 0)
        height = int(image_info.get("height") or 0)
        if width <= 0 or height <= 0:
            size = read_image_size(image_path)
            if size is None:
                continue
            width, height = size

        boxes: list[BBox] = []
        for ann in ann_by_image.get(image_id, []):
            if ann.get("iscrowd"):
                continue
            category = categories.get(int(ann.get("category_id", -1)))
            bbox = ann.get("bbox") or []
            if category is None or len(bbox) != 4:
                continue
            x, y, w, h = map(float, bbox)
            box = box_from_xyxy(category, x, y, x + w, y + h, width, height)
            if box:
                boxes.append(box)
        samples.append(Sample(image_path=image_path, boxes=tuple(boxes)))
    return samples, classes


def load_voc(input_dir: Path, args: argparse.Namespace) -> tuple[list[Sample], list[str]]:
    ann_dir = args.voc_ann_dir or input_dir
    image_dir = args.image_dir or input_dir
    xml_files = sorted(Path(ann_dir).rglob("*.xml"))
    samples: list[Sample] = []
    classes: list[str] = []
    seen_classes: set[str] = set()

    progress = ProgressBar(len(xml_files), "Reading Pascal VOC")
    for xml_path in xml_files:
        progress.update()
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            logging.warning("Corrupt XML skipped: %s (%s)", xml_path, exc)
            continue

        filename = (root.findtext("filename") or "").strip()
        image_path = Path(image_dir) / filename if filename else Path(image_dir) / f"{xml_path.stem}.jpg"
        if not image_path.is_file() and filename:
            image_path = next(input_dir.rglob(filename), image_path)
        if not image_path.is_file():
            for suffix in IMAGE_SUFFIXES:
                candidate = next(input_dir.rglob(f"{xml_path.stem}{suffix}"), None)
                if candidate:
                    image_path = candidate
                    break

        size_node = root.find("size")
        width = int(size_node.findtext("width", "0")) if size_node is not None else 0
        height = int(size_node.findtext("height", "0")) if size_node is not None else 0
        if width <= 0 or height <= 0:
            size = read_image_size(image_path)
            if size is None:
                continue
            width, height = size

        boxes: list[BBox] = []
        for obj in root.findall("object"):
            class_name = (obj.findtext("name") or "").strip()
            bnd = obj.find("bndbox")
            if not class_name or bnd is None:
                continue
            coords = [float(bnd.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax")]
            box = box_from_xyxy(class_name, coords[0], coords[1], coords[2], coords[3], width, height)
            if box:
                boxes.append(box)
                if class_name not in seen_classes:
                    seen_classes.add(class_name)
                    classes.append(class_name)
        samples.append(Sample(image_path=image_path, boxes=tuple(boxes)))
    return samples, classes


def load_yolo(input_dir: Path, args: argparse.Namespace) -> tuple[list[Sample], list[str]]:
    data_yaml = args.data_yaml or input_dir / "data.yaml"
    yaml_path, classes = parse_simple_data_yaml(data_yaml)
    image_dir = args.image_dir or (input_dir / "images" if (input_dir / "images").is_dir() else input_dir)
    label_dir = args.yolo_label_dir or (input_dir / "labels" if (input_dir / "labels").is_dir() else input_dir)
    if yaml_path and not Path(yaml_path).is_absolute():
        image_dir = data_yaml.parent / yaml_path / "images"
        label_dir = data_yaml.parent / yaml_path / "labels"

    images = find_images(Path(image_dir))
    samples: list[Sample] = []
    max_class_idx = -1
    progress = ProgressBar(len(images), "Reading YOLO")
    for image_path in images:
        progress.update()
        rel = image_path.relative_to(image_dir)
        label_path = Path(label_dir) / rel.with_suffix(".txt")
        if not label_path.is_file():
            label_path = Path(label_dir) / f"{image_path.stem}.txt"
        boxes: list[BBox] = []
        if label_path.is_file():
            for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = raw.split()
                if not parts:
                    continue
                if len(parts) < 5:
                    logging.warning("Corrupt YOLO label skipped: %s:%d", label_path, line_no)
                    continue
                try:
                    class_idx = int(float(parts[0]))
                    coords = [float(value) for value in parts[1:5]]
                except ValueError:
                    logging.warning("Corrupt YOLO label skipped: %s:%d", label_path, line_no)
                    continue
                max_class_idx = max(max_class_idx, class_idx)
                class_name = classes[class_idx] if class_idx < len(classes) else f"class_{class_idx}"
                if all(0.0 <= value <= 1.0 for value in coords) and coords[2] > 0 and coords[3] > 0:
                    boxes.append(BBox(class_name, coords[0], coords[1], coords[2], coords[3]))
                else:
                    logging.warning("Out-of-range YOLO box skipped: %s:%d", label_path, line_no)
        samples.append(Sample(image_path=image_path, boxes=tuple(boxes)))

    if not classes and max_class_idx >= 0:
        classes = [f"class_{idx}" for idx in range(max_class_idx + 1)]
    return samples, classes


def load_labelme(input_dir: Path, args: argparse.Namespace) -> tuple[list[Sample], list[str]]:
    json_files = sorted((args.labelme_dir or input_dir).rglob("*.json"))
    image_dir = args.image_dir or input_dir
    samples: list[Sample] = []
    classes: list[str] = []
    seen_classes: set[str] = set()

    progress = ProgressBar(len(json_files), "Reading LabelMe")
    for json_path in json_files:
        progress.update()
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logging.warning("Corrupt LabelMe JSON skipped: %s (%s)", json_path, exc)
            continue

        image_name = data.get("imagePath") or f"{json_path.stem}.jpg"
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = Path(image_dir) / image_name
        if not image_path.is_file():
            image_path = next(input_dir.rglob(Path(image_name).name), image_path)

        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            size = read_image_size(image_path)
            if size is None:
                continue
            width, height = size

        boxes: list[BBox] = []
        for shape in data.get("shapes", []):
            class_name = str(shape.get("label", "")).strip()
            points = shape.get("points") or []
            if not class_name or len(points) < 2:
                continue
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            box = box_from_xyxy(class_name, min(xs), min(ys), max(xs), max(ys), width, height)
            if box:
                boxes.append(box)
                if class_name not in seen_classes:
                    seen_classes.add(class_name)
                    classes.append(class_name)
        samples.append(Sample(image_path=image_path, boxes=tuple(boxes)))
    return samples, classes


def load_samples(input_dir: Path, args: argparse.Namespace) -> tuple[list[Sample], list[str], str]:
    fmt = detect_format(input_dir, args)
    loaders = {
        "pad-ufes": load_pad_ufes,
        "coco": load_coco,
        "voc": load_voc,
        "yolo": load_yolo,
        "labelme": load_labelme,
    }
    samples, detected_classes = loaders[fmt](input_dir, args)
    override_classes = parse_class_override(args.classes)
    classes = override_classes or detected_classes
    if not classes:
        raise SystemExit("No classes were found. Pass --classes with comma-separated names or a class file.")
    logging.info("Detected format: %s", fmt)
    logging.info("Classes (%d): %s", len(classes), ", ".join(classes))
    return samples, classes, fmt


def validate_samples(samples: Iterable[Sample], classes: list[str], empty_policy: str) -> list[Sample]:
    class_set = set(classes)
    valid: list[Sample] = []
    empty_count = 0
    corrupt_label_count = 0

    sample_list = list(samples)
    progress = ProgressBar(len(sample_list), "Validating")
    for sample in sample_list:
        progress.update()
        if not sample.image_path.is_file():
            logging.warning("Missing image skipped: %s", sample.image_path)
            continue
        if read_image_size(sample.image_path) is None:
            continue

        clean_boxes: list[BBox] = []
        for box in sample.boxes:
            if box.class_name not in class_set:
                logging.warning("Unknown class skipped for %s: %s", sample.image_path, box.class_name)
                corrupt_label_count += 1
                continue
            values = (box.x_center, box.y_center, box.width, box.height)
            if not all(0.0 <= value <= 1.0 for value in values) or box.width <= 0 or box.height <= 0:
                logging.warning("Invalid bbox skipped for %s: %s", sample.image_path, box)
                corrupt_label_count += 1
                continue
            clean_boxes.append(box)

        if not clean_boxes:
            empty_count += 1
            if empty_policy == "skip":
                logging.warning("Empty-label image skipped: %s", sample.image_path)
                continue
            logging.debug("Empty-label image kept as negative sample: %s", sample.image_path)
        valid.append(Sample(sample.image_path, tuple(clean_boxes)))

    logging.info("Validated samples: %d kept, %d empty labels, %d corrupt labels skipped", len(valid), empty_count, corrupt_label_count)
    return valid


def split_samples(samples: list[Sample], ratios: tuple[float, float, float], seed: int) -> dict[str, list[Sample]]:
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        key = sample.boxes[0].class_name if sample.boxes else "__empty__"
        grouped.setdefault(key, []).append(sample)

    rng = random.Random(seed)
    result = {split: [] for split in SPLITS}
    train_ratio, val_ratio, _test_ratio = ratios
    for group_samples in grouped.values():
        rng.shuffle(group_samples)
        total = len(group_samples)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        if total >= 3:
            if train_count == 0:
                train_count = 1
            if val_count == 0:
                val_count = 1
        if train_count + val_count > total:
            val_count = max(0, total - train_count)
        result["train"].extend(group_samples[:train_count])
        result["val"].extend(group_samples[train_count : train_count + val_count])
        result["test"].extend(group_samples[train_count + val_count :])

    for split in SPLITS:
        result[split].sort(key=lambda sample: str(sample.image_path))
    return result


def parse_split_ratios(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise SystemExit("--split-ratios expects exactly three values, e.g. 0.8 0.1 0.1")
    total = sum(values)
    if total <= 0:
        raise SystemExit("--split-ratios must sum to a positive value")
    ratios = tuple(value / total for value in values)
    return ratios  # type: ignore[return-value]


def safe_output_name(image_path: Path, used_names: set[str]) -> str:
    candidate = image_path.name
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate
    prefix = image_path.parent.name
    candidate = f"{prefix}_{image_path.name}"
    counter = 1
    while candidate in used_names:
        candidate = f"{prefix}_{counter}_{image_path.name}"
        counter += 1
    used_names.add(candidate)
    return candidate


def copy_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        target = src.resolve()
        try:
            dst.symlink_to(target)
            return
        except OSError:
            logging.warning("Symlink failed, falling back to copy: %s", dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            logging.warning("Hardlink failed, falling back to copy: %s", dst)
    shutil.copy2(src, dst)


def write_dataset(output_dir: Path, splits: dict[str, list[Sample]], classes: list[str], args: argparse.Namespace) -> None:
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    used_names: set[str] = set()
    total = sum(len(items) for items in splits.values())
    progress = ProgressBar(total, "Writing output")
    for split, split_samples in splits.items():
        for sample in split_samples:
            progress.update()
            image_name = safe_output_name(sample.image_path, used_names)
            image_dst = output_dir / "images" / split / image_name
            label_dst = output_dir / "labels" / split / f"{Path(image_name).stem}.txt"
            copy_image(sample.image_path, image_dst, args.copy_mode)
            lines = [
                f"{class_to_idx[box.class_name]} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
                for box in sample.boxes
            ]
            label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(classes)}",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {class_name}" for idx, class_name in enumerate(classes))
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PAD-UFES, COCO, VOC, YOLO TXT, or LabelMe into Ultralytics YOLO format.")
    parser.add_argument("--input", required=True, type=Path, help="Input dataset root.")
    parser.add_argument("--output", required=True, type=Path, help="Output dataset root.")
    parser.add_argument("--format", choices=["auto", "pad-ufes", "coco", "voc", "yolo", "labelme"], default="auto")
    parser.add_argument("--split-ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument("--classes", default="", help="Comma-separated class names or a text file with one class per line.")
    parser.add_argument("--empty-labels", choices=["keep", "skip"], default="keep", help="Keep or skip images without valid labels.")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy", help="How to place images in the output dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it already exists.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed debug logs.")

    parser.add_argument("--metadata-csv", type=Path, help="PAD-UFES metadata.csv path.")
    parser.add_argument("--coco-json", type=Path, help="COCO annotations JSON path.")
    parser.add_argument("--voc-ann-dir", type=Path, help="Pascal VOC XML annotation directory.")
    parser.add_argument("--yolo-label-dir", type=Path, help="YOLO TXT label directory.")
    parser.add_argument("--labelme-dir", type=Path, help="LabelMe JSON annotation directory.")
    parser.add_argument("--image-dir", type=Path, help="Image directory override.")
    parser.add_argument("--data-yaml", type=Path, help="Existing YOLO data.yaml for class names/path hints.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    ratios = parse_split_ratios(args.split_ratios)
    samples, classes, fmt = load_samples(input_dir, args)
    valid_samples = validate_samples(samples, classes, args.empty_labels)
    if not valid_samples:
        raise SystemExit("No valid samples found after validation.")

    splits = split_samples(valid_samples, ratios, args.seed)
    logging.info("Split counts: train=%d, val=%d, test=%d", len(splits["train"]), len(splits["val"]), len(splits["test"]))
    write_dataset(output_dir, splits, classes, args)
    logging.info("Wrote %s-compatible dataset (%s source) to: %s", "YOLO26/RT-DETR", fmt, output_dir)
    logging.info("Use with: python3 train.py --data %s --model yolo26n", output_dir / "data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
