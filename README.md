# YOLO26 + DETR (Docker)

Docker image with **PyTorch**, **Ultralytics** (YOLO26, RT-DETR), and **Transformers** (DETR). **GPU is optional** — add `--gpus all` when you have NVIDIA + [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html); otherwise everything runs on **CPU** (slower).

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

Ultralytics layout: `data.yaml` + `images/{train,val}/` + matching `labels/{train,val}/`. See `dataset.example.yaml`.

```powershell
docker run --rm -v "C:\path\to\dataset:/dataset" yolo26-detr:cuda python3 train.py --data /dataset/data.yaml --model yolo26n --epochs 100 --batch 16
```

**`--model`:** `yolo26n` | `yolo26s` | `yolo26m` | `rtdetr` / `rtdetr-l` / `rtdetr-x` (aliases map to `.pt` weights). **`--device`** defaults to **`auto`** (GPU 0 if CUDA exists, else `cpu`). Override weights: **`--weights path.pt`**.

## Kaggle dataset

Data lives under `Datasets/` (gitignored). Download [mahdavi1202/skin-cancer](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer) with a Kaggle token (`%USERPROFILE%\.kaggle\kaggle.json`):

```powershell
docker run --rm -v "$env:USERPROFILE\.kaggle\kaggle.json:/root/.kaggle/kaggle.json:ro" -v "pad_ufes20_data:/app/Datasets" yolo26-detr:cuda python3 scripts/download_kaggle_dataset.py
```

`--force` replaces an existing folder. If `metadata.csv` + images are already there, the script **skips** download (no Kaggle call). Env: `KAGGLE_DATASET`, `PAD_UFES20_ROOT`, or `KAGGLE_USERNAME` + `KAGGLE_KEY`.

## Inference example

```powershell
docker run --rm -v "C:\path\to\data:/data" yolo26-detr:cuda python3 example_infer.py --image /data/photo.jpg --yolo-weights /data/best.pt
```

## Main files

| Path | Role |
|------|------|
| `Dockerfile` | Image build |
| `requirements.txt` | Python deps |
| `train.py` | Train YOLO26 / RT-DETR |
| `example_infer.py` | YOLO + HF DETR demo |
| `scripts/download_kaggle_dataset.py` | Pull Kaggle dataset into `/app/Datasets/PAD-UFES-20` |
| `prefetch_weights.py` | Optional build-time weight cache |

**Push fails / huge repo:** large files under `Datasets/` must not be committed; use `.gitignore` and remove them from Git history if they were added before.
