# YOLO26 + DETR (Docker)

Docker image with **PyTorch**, **Ultralytics** (YOLO26, RT-DETR), and **Transformers** (DETR). **GPU is optional** — add `--gpus all` when you have NVIDIA + [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html); otherwise everything runs on **CPU** (slower).

## Quick workflow

1. **Build** the image (from this repo folder).
2. **Download** PAD-UFES (optional) or place data under `Datasets/`.
3. **Convert** to YOLO layout with `scripts/convert_dataset.py` if needed.
4. **Train** with `train.py`, mounting the folder that contains `data.yaml`.

## Build

Run all build commands from the project folder:

```powershell
cd "c:\Users\denni\Desktop\isitmelanoma-docker"
docker build -t yolo26-detr:cuda .
```

This builds the default CUDA-capable image and prefetches:

- `yolo26n.pt`
- `yolo26s.pt`
- `yolo26m.pt`
- `rtdetr-l.pt`
- `rtdetr-x.pt`

Build faster without downloading weights:

```powershell
docker build --build-arg SKIP_PREFETCH=1 -t yolo26-detr:cuda .
```

CPU-only / smaller image:

```powershell
docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.5.1-runtime -t yolo26-detr:cpu .
```

Use another CUDA PyTorch base:

```powershell
docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime -t yolo26-detr:cuda .
```

`.` at the end means “use this folder as the Docker build context.”

## Run

Test the image:

```powershell
docker run --rm yolo26-detr:cuda
```

With GPU:

```powershell
docker run --rm --gpus all yolo26-detr:cuda
```

Open a shell:

```powershell
docker run --rm -it yolo26-detr:cuda bash
```

Add `--gpus all` to shell/training/inference commands when you want CUDA.

## Train

Ultralytics layout: `data.yaml` at the dataset root, plus `images/{train,val,test}/` and matching `labels/{train,val,test}/`. See `dataset.example.yaml`.

**Important:** The path left of `:` in `-v` must be a **real folder on your machine** that contains `data.yaml`. Do not use placeholder paths like `C:\path\to\dataset`.

From the repo directory, after converting PAD-UFES (see below):

```powershell
cd "c:\Users\denni\Desktop\isitmelanoma-docker"
docker run --rm --gpus all -v "$(Get-Location)\Datasets\PAD-UFES-20-yolo:/dataset" yolo26-detr:cuda python3 train.py --data /dataset/data.yaml --model yolo26n --epochs 100 --batch 16
```

Omit `--gpus all` if you are on CPU only.

**`--model`:** `yolo26n` | `yolo26s` | `yolo26m` | `rtdetr` / `rtdetr-l` / `rtdetr-x` (aliases map to `.pt` weights). **`--device`** defaults to **`auto`** (GPU 0 if CUDA exists, else `cpu`). Override weights: **`--weights path.pt`**.

Training writes runs under `/app/runs` in the container unless you mount a host folder (optional):

```powershell
docker run --rm --gpus all `
  -v "$(Get-Location)\Datasets\PAD-UFES-20-yolo:/dataset" `
  -v "$(Get-Location)\runs:/app/runs" `
  yolo26-detr:cuda python3 train.py --data /dataset/data.yaml --model yolo26n --epochs 100 --batch 16
```

## Convert dataset

Convert PAD-UFES, COCO JSON, Pascal VOC XML, YOLO TXT, or LabelMe JSON into the shared YOLO/RT-DETR layout:

```powershell
cd "c:\Users\denni\Desktop\isitmelanoma-docker"
python scripts/convert_dataset.py --input Datasets/PAD-UFES-20 --output Datasets/PAD-UFES-20-yolo --format pad-ufes --split-ratios 0.8 0.1 0.1 --seed 42 --overwrite
```

On Windows, **`--copy-mode hardlink`** avoids duplicating large image files when source and output are on the same volume:

```powershell
python scripts/convert_dataset.py --input Datasets/PAD-UFES-20 --output Datasets/PAD-UFES-20-yolo --format pad-ufes --split-ratios 0.8 0.1 0.1 --seed 42 --copy-mode hardlink --overwrite
```

Then train using the generated `data.yaml` (see **Train** above).

Use `--format auto` for automatic detection, or choose `coco`, `voc`, `yolo`, or `labelme`. PAD-UFES has **image-level** labels in `metadata.csv`; the converter writes **one full-image bounding box** per image so Ultralytics detection training can run. For true lesion localization you need real box/segmentation annotations, not this shortcut.

### PAD-UFES `metadata.csv` columns (cheat sheet)

| Column | Meaning |
|--------|--------|
| `patient_id` | Patient identifier |
| `lesion_id` | Lesion identifier |
| `img_id` | Image filename (matches file under `imgs_part_*`) |
| `diagnostic` | Lesion class label used for training (e.g. NEV, BCC, MEL) |
| `smoke`, `drink`, `age`, `gender`, … | Clinical / demographic covariates |
| `diameter_1`, `diameter_2` | Approximate lesion diameters (when recorded) |
| `biopsed` | Whether the lesion was biopsied |

See the [PAD-UFES-20 paper](https://arxiv.org/abs/2007.00478) for full variable definitions.

## Kaggle dataset

Data lives under `Datasets/` (gitignored). Download [mahdavi1202/skin-cancer](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer) with a Kaggle token (`%USERPROFILE%\.kaggle\kaggle.json`):

```powershell
docker run --rm -v "$env:USERPROFILE\.kaggle\kaggle.json:/root/.kaggle/kaggle.json:ro" -v "pad_ufes20_data:/app/Datasets" yolo26-detr:cuda python3 scripts/download_kaggle_dataset.py
```

To use a **named volume** but still run conversion on the host, copy data out of the volume or mount a **bind path** instead, for example:

```powershell
docker run --rm -v "$env:USERPROFILE\.kaggle\kaggle.json:/root/.kaggle/kaggle.json:ro" -v "$(Get-Location)\Datasets:/app/Datasets" yolo26-detr:cuda python3 scripts/download_kaggle_dataset.py
```

`--force` replaces an existing folder. If `metadata.csv` + images are already there, the script **skips** download (no Kaggle call). Env: `KAGGLE_DATASET`, `PAD_UFES20_ROOT`, or `KAGGLE_USERNAME` + `KAGGLE_KEY`.

## Inference example

Mount a folder or single file that contains your image and weights:

```powershell
cd "c:\Users\denni\Desktop\isitmelanoma-docker"
docker run --rm --gpus all -v "$(Get-Location)\runs\train\exp\weights:/weights" -v "$(Get-Location)\some_image.jpg:/data/photo.jpg:ro" yolo26-detr:cuda python3 example_infer.py --image /data/photo.jpg --yolo-weights /weights/best.pt
```

Replace `runs\train\exp\weights` with your actual run folder.

## Troubleshooting

**`Dataset YAML not found: /dataset/data.yaml`**

- The host path in `-v HOST:/dataset` must exist and must contain `data.yaml` at `HOST\data.yaml`.
- Fix: use `$(Get-Location)\Datasets\PAD-UFES-20-yolo` (or your real output folder), not a placeholder like `C:\path\to\dataset`.

**`user config directory '/root/.config/Ultralytics' is not writable`**

- Harmless in Docker; Ultralytics falls back to `/tmp/Ultralytics`. Optional: `docker run -e YOLO_CONFIG_DIR=/tmp/Ultralytics ...`.

## Main files

| Path | Role |
|------|------|
| `Dockerfile` | Image build |
| `requirements.txt` | Python deps |
| `train.py` | Train YOLO26 / RT-DETR |
| `example_infer.py` | YOLO + HF DETR demo |
| `scripts/download_kaggle_dataset.py` | Pull Kaggle dataset into `/app/Datasets/PAD-UFES-20` |
| `scripts/convert_dataset.py` | Convert PAD-UFES / COCO / VOC / YOLO / LabelMe to training layout |
| `prefetch_weights.py` | Optional build-time weight cache |

**Push fails / huge repo:** large files under `Datasets/` must not be committed; use `.gitignore` and remove them from Git history if they were added before.
