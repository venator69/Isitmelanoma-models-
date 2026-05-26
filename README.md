# YOLO26 + DETR (CUDA) Docker

This project builds a GPU-enabled Docker image with **PyTorch (CUDA)**, **Ultralytics** (YOLO26 and related models), and **Hugging Face Transformers** (DETR and other object-detection checkpoints).

## Prerequisites

Before you build or run, your **host** must have:

1. **NVIDIA GPU driver** installed (check with `nvidia-smi`).
2. **Docker** with GPU support:
   - **Linux**: Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so `docker run --gpus all` works.
   - **Windows**: Use **Docker Desktop** with the **WSL2** backend and a WSL2 distro that supports your GPU, or run Docker on a Linux machine with the toolkit.

If `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi` fails, fix GPU passthrough before building this image.

## Clone or open the project

```powershell
cd "c:\Users\denni\Desktop\isitmelanoma-docker"
```

(Adjust the path if you moved the folder.)

## Build the image

From the directory that contains the `Dockerfile`:

```powershell
docker build -t yolo26-detr:cuda .
```

- **`-t yolo26-detr:cuda`** sets the image name and tag; change it if you prefer another name.
- **`.`** is the build context (this folder).

First build may take several minutes while dependencies download.

## Skin-cancer dataset (Kaggle)

Large image data is **not** committed to git (see **`.gitignore`**: `Datasets/`). Inside Docker, download the mirror you specified on Kaggle:

- Dataset: **[mahdavi1202 / skin-cancer](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer)**  
- Script: **`scripts/download_kaggle_dataset.py`** (installs files under **`/app/Datasets/PAD-UFES-20`** by default, with `metadata.csv` + images).

### 1. Kaggle API token

1. On Kaggle: **Account** → **Create New API Token** → downloads **`kaggle.json`**.
2. Place it at **`%USERPROFILE%\.kaggle\kaggle.json`** (Windows) or **`~/.kaggle/kaggle.json`** (Linux/macOS).

Never commit `kaggle.json` (it is listed in **`.gitignore`**).

### 2. Download inside a container

Mount the token read-only, then run the downloader (no GPU required for this step):

**Windows (PowerShell):**

```powershell
docker run --rm `
  -v "$env:USERPROFILE\.kaggle\kaggle.json:/root/.kaggle/kaggle.json:ro" `
  -v "pad_ufes20_data:/app/Datasets" `
  yolo26-detr:cuda python3 scripts/download_kaggle_dataset.py
```

The named volume **`pad_ufes20_data`** keeps the dataset between runs so you do not re-download every time. Omit that `-v` line if you only want a throwaway download inside the container filesystem.

**Override slug or destination:**

```powershell
docker run --rm `
  -e KAGGLE_DATASET=mahdavi1202/skin-cancer `
  -e PAD_UFES20_ROOT=/app/Datasets/PAD-UFES-20 `
  -v "$env:USERPROFILE\.kaggle\kaggle.json:/root/.kaggle/kaggle.json:ro" `
  yolo26-detr:cuda python3 scripts/download_kaggle_dataset.py --force
```

Use **`--force`** to replace an incomplete or old folder (see script help: `python3 scripts/download_kaggle_dataset.py --help`).

Alternative to mounting the file: pass **`KAGGLE_USERNAME`** and **`KAGGLE_KEY`** from the JSON as `-e` variables (avoid logging these in shared CI logs).

### Optional: use a different PyTorch base image

The default base is `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. To override it (for example to match another CUDA minor on your driver):

```powershell
docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime -t yolo26-detr:cuda .
```

Choose a tag whose CUDA version is **supported by your NVIDIA driver** (your driver’s `nvidia-smi` output lists a maximum CUDA version).

## Run the container (GPU smoke test)

```powershell
docker run --rm --gpus all yolo26-detr:cuda
```

The default command prints whether **CUDA** is visible and imports **Ultralytics** and **Transformers**. You should see `cuda: True` and `ok` when the GPU is passed through correctly.

### Build without prefetching weights (offline / air-gapped builds)

By default the image runs `prefetch_weights.py` during `docker build` to cache **yolo26n**, **yolo26s**, **yolo26m**, **rtdetr-l**, and **rtdetr-x** checkpoints (needs internet during build). To skip that step:

```powershell
docker build --build-arg SKIP_PREFETCH=1 -t yolo26-detr:cuda .
```

Weights will download on the first `train.py` run instead.

## Train YOLO26 or RT-DETR on your dataset (images + YOLO labels)

The image includes **`train.py`**, which trains on a standard **Ultralytics YOLO dataset**: RGB images plus one **`.txt` label file per image** (normalized class + box), described by a **`data.yaml`**.

### Supported `--model` presets

**RT-DETR** (Ultralytics real-time DETR) uses the same YOLO-style `data.yaml` and label files as YOLO26; pick a size below.

| `--model` | Pretrained weights used |
|-----------|-------------------------|
| `yolo26n` | `yolo26n.pt` |
| `yolo26s` | `yolo26s.pt` |
| `yolo26m` | `yolo26m.pt` |
| `rtdetr` | `rtdetr-l.pt` (alias for RT-DETR-L) |
| `rtdetr-l` | `rtdetr-l.pt` |
| `rtdetr-x` | `rtdetr-x.pt` |

Override any preset with **`--weights /path/to/custom.pt`**.

### Dataset layout

Next to your `data.yaml` (see `dataset.example.yaml` in this repo):

- `images/train/`, `images/val/` — image files (for example `.jpg`)
- `labels/train/`, `labels/val/` — YOLO-format `.txt` files, **same base name** as each image

Your YAML should set `path`, `train`, `val`, `nc`, and `names` so Ultralytics can find those folders.

### Example: train inside Docker

Mount the folder that contains `data.yaml` and the `images` / `labels` trees at `/dataset`:

```powershell
docker run --rm --gpus all -v "C:\path\to\your\dataset:/dataset" yolo26-detr:cuda python3 train.py --model yolo26n --data /dataset/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0 --project /dataset/runs --name exp1
```

RT-DETR-L (default alias `rtdetr`):

```powershell
docker run --rm --gpus all -v "C:\path\to\your\dataset:/dataset" yolo26-detr:cuda python3 train.py --model rtdetr --data /dataset/data.yaml --epochs 100 --batch 8 --device 0
```

RT-DETR-X (larger; reduce `--batch` if you run out of VRAM):

```powershell
docker run --rm --gpus all -v "C:\path\to\your\dataset:/dataset" yolo26-detr:cuda python3 train.py --model rtdetr-x --data /dataset/data.yaml --epochs 100 --batch 4 --device 0
```

Use a smaller **`--batch`** if you hit GPU out-of-memory (especially for `yolo26m` or `rtdetr-x`).

## Run example inference (YOLO + DETR)

Mount a folder on your PC that contains an image and (optionally) your YOLO weights, then run `example_infer.py` inside the container.

**Windows (PowerShell), example:**

```powershell
docker run --rm --gpus all -v "C:\path\to\your\data:/data" yolo26-detr:cuda python3 example_infer.py --image /data/photo.jpg --yolo-weights /data/best.pt --detr-model facebook/detr-resnet-50
```

- **`--image`**: path **inside the container** (under `/data` if you mounted `C:\path\to\your\data` to `/data`).
- **`--yolo-weights`**: your `.pt` checkpoint or an Ultralytics-compatible weight name.
- **`--detr-model`**: any Hugging Face model id that exposes `AutoModelForObjectDetection` (swap for your “mobile” or custom DETR checkpoint).

## Useful commands

| Goal | Command |
|------|---------|
| List images | `docker images` |
| Open a shell in the image | `docker run --rm -it --gpus all yolo26-detr:cuda bash` |
| Rebuild after code changes | `docker build -t yolo26-detr:cuda .` |

## Troubleshooting

- **`could not select device driver "" with capabilities: [[gpu]]`**  
  Install and configure the **NVIDIA Container Toolkit** (Linux) or enable GPU in **Docker Desktop** / WSL2 (Windows).

- **`cuda: False` inside the container**  
  You ran without `--gpus all`, or the host cannot see the GPU. Confirm `nvidia-smi` on the host, then retry with `--gpus all`.

- **Build fails pulling `pytorch/pytorch`**  
  Check network and Docker Hub rate limits; retry later or use a mirror if your organization provides one.

## Project layout

| File | Purpose |
|------|---------|
| `Dockerfile` | CUDA PyTorch base, installs deps, optionally prefetches YOLO26/RT-DETR weights. |
| `requirements.txt` | Python packages (Ultralytics, Transformers, OpenCV headless, etc.). |
| `.dockerignore` | Reduces build context size (excludes venvs, large weights if present locally). |
| `prefetch_weights.py` | Build-time download of default checkpoints into the image cache. |
| `train.py` | Train `yolo26n` / `yolo26s` / `yolo26m` / RT-DETR (`rtdetr`, `rtdetr-l`, `rtdetr-x`) on a YOLO-format dataset YAML. |
| `scripts/download_kaggle_dataset.py` | Download [Kaggle skin-cancer](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer) into `/app/Datasets/PAD-UFES-20` using API credentials. |
| `example_infer.py` | Minimal example: YOLO predict + DETR forward on one image. |
