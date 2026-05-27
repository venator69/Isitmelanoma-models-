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
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.nn.tasks import unwrap_model

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
            "Base RNG seed when --fraction < 1.0: each epoch uses a new random subset derived from this base "
            "(omit: random base each run, printed once). Pass an int for reproducible per-epoch sampling across runs."
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


def _paths_with_label_txt(image_paths: list[str]) -> list[str]:
    """Keep only images whose YOLO sibling label .txt exists (same rule as Ultralytics)."""
    out: list[str] = []
    for img, lab in zip(image_paths, img2label_paths(image_paths)):
        if Path(lab).is_file():
            out.append(img)
    return out


def _write_path_list_txt(paths: list[str]) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_train_paths.txt", delete=False, encoding="utf-8", newline="\n"
    )
    with tmp as f:
        for line in paths:
            f.write(line + "\n")
    return Path(tmp.name)


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


def _epoch_subset_rng(base_seed: int, epoch: int) -> random.Random:
    """Deterministic per-epoch RNG from a single run-level base seed."""
    # Mix epoch into seed (stable across platforms for a given base + epoch).
    mixed = (base_seed ^ (epoch * 1_000_003) ^ (epoch * epoch * 7)) % (2**31)
    return random.Random(mixed)


def _apply_train_subset_to_dataset(
    trainer: object, chosen_paths: list[str], label_map: dict[str, dict]
) -> None:
    """Replace train dataset image list and labels in-place (same length k each epoch)."""
    ds = trainer.train_loader.dataset
    labels = [deepcopy(label_map[p]) for p in chosen_paths]
    ds.im_files = list(chosen_paths)
    ds.labels = labels
    ds.ni = len(labels)
    ds.npy_files = [Path(f).with_suffix(".npy") for f in chosen_paths]
    ds.ims = [None] * ds.ni
    ds.im_hw0 = [None] * ds.ni
    ds.im_hw = [None] * ds.ni
    ds.buffer = []
    if ds.augment:
        ds.max_buffer_length = min((ds.ni, ds.batch_size * 8, 1000))
    if getattr(ds, "rect", False) and ds.batch_size:
        ds.set_rectangle()


def install_per_epoch_subset_callbacks(
    model: YOLO,
    *,
    full_train_txt: Path,
    subset_k: int,
    base_seed: int,
) -> None:
    """Resample train subset at each epoch start (single training run, no model re-init)."""

    def on_train_start(trainer: object) -> None:
        gs = max(int(unwrap_model(trainer.model).stride.max()), 32)
        full_ds = build_yolo_dataset(
            trainer.args,
            str(full_train_txt.resolve()),
            trainer.batch_size,
            trainer.data,
            mode="train",
            stride=gs,
        )
        label_map = {lb["im_file"]: deepcopy(lb) for lb in full_ds.labels}
        pool = list(label_map.keys())
        del full_ds
        if not pool:
            raise RuntimeError("Per-epoch subset: no valid labeled images in train pool.")
        if len(pool) < subset_k:
            raise RuntimeError(
                f"After scanning labels, only {len(pool)} usable train images remain (need k={subset_k}). "
                "Fix corrupt images/labels or lower --fraction."
            )
        trainer._subset_pool = pool  # type: ignore[attr-defined]
        trainer._subset_k = subset_k  # type: ignore[attr-defined]
        trainer._subset_label_map = label_map  # type: ignore[attr-defined]
        trainer._subset_base_seed = base_seed  # type: ignore[attr-defined]

    def on_train_epoch_start(trainer: object) -> None:
        pool: list[str] = trainer._subset_pool  # type: ignore[attr-defined]
        k: int = trainer._subset_k  # type: ignore[attr-defined]
        label_map: dict[str, dict] = trainer._subset_label_map  # type: ignore[attr-defined]
        base_seed: int = trainer._subset_base_seed  # type: ignore[attr-defined]
        epoch = int(trainer.epoch)
        rng = _epoch_subset_rng(base_seed, epoch)
        chosen = rng.sample(pool, k)
        _apply_train_subset_to_dataset(trainer, chosen, label_map)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)


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
    frac = float(args.fraction)
    if frac < 1.0:
        # Workers fork a snapshot of the dataset; in-place resampling would not reach worker processes.
        if args.workers is not None and args.workers != 0:
            print(
                "warning: --fraction uses per-epoch subset resampling; forcing workers=0 "
                f"(ignoring --workers {args.workers})."
            )
        train_kw["workers"] = 0
    elif args.workers is not None:
        train_kw["workers"] = args.workers
    elif Path("/.dockerenv").is_file():
        # Default Docker shm is small; multiprocessing workers often OOM or bus-error.
        train_kw["workers"] = 0

    if not (0.0 < frac <= 1.0):
        raise SystemExit("--fraction must be in (0, 1], e.g. 0.25")

    # Ultralytics 8.4.x expects `data` as a path string, not a dict. `path: .` in the user's
    # YAML is resolved against cwd (/app in Docker). Write a short-lived YAML with an absolute
    # `path` and pass that file to `model.train`.
    root = yaml_path.parent.resolve()
    tmp_path: Path | None = None
    subset_txt: Path | None = None
    full_train_txt: Path | None = None
    try:
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(cfg, dict):
            raise ValueError("data.yaml must be a YAML mapping")
        cfg["path"] = str(root)
        train_key = cfg.get("train")
        if frac < 1.0:
            if args.resume:
                raise SystemExit(
                    "--resume is not supported together with --fraction "
                    "(per-epoch subset + checkpoint). Train without --resume or use fraction=1.0."
                )
            if not train_key:
                raise SystemExit("data.yaml must define `train` when using --fraction < 1.0")
            all_train = _paths_with_label_txt(_resolve_train_image_paths(train_key, root))
            if not all_train:
                raise SystemExit("No training images with existing label .txt files found.")
            subset_seed = (
                int(args.subset_seed)
                if args.subset_seed is not None
                else secrets.randbelow(2**31)
            )
            subset_k = min(len(all_train), max(1, round(len(all_train) * frac)))
            full_train_txt = _write_path_list_txt(all_train)
            # Initial list for Ultralytics: fixed size k so dataloader length / LR warmup stay consistent.
            subset_txt, _ = _write_random_train_subset(all_train, fraction=frac, seed=subset_seed)
            cfg["train"] = str(subset_txt.resolve())
            seed_note = (
                ""
                if args.subset_seed is not None
                else " (random base each run; use --subset-seed to reproduce)"
            )
            print(
                f"train subset per epoch: pool={len(all_train)} images, k={subset_k} per epoch "
                f"(fraction={frac}, subset_seed_base={subset_seed}){seed_note}"
            )
            install_per_epoch_subset_callbacks(
                model,
                full_train_txt=full_train_txt,
                subset_k=subset_k,
                base_seed=subset_seed,
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
        if full_train_txt is not None:
            full_train_txt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
