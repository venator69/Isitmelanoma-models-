# YOLO26 + DETR (Docker)

This project builds a Docker image with **PyTorch**, **Ultralytics** (YOLO26 and related models), and **Hugging Face Transformers** (DETR and other object-detection checkpoints). **A GPU is optional:** the same image runs on **CPU** (slower); use **`--gpus all`** when you have an NVIDIA GPU and want CUDA.

## Prerequisites

1. **Docker** installed.
2. **Optional — NVIDIA GPU:** install the **NVIDIA driver** and (on Linux / WSL2 GPU) the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)** so `docker run --gpus all` exposes the GPU. You can skip this for CPU-only use.

If you use a **CUDA** base image on a machine **without** a GPU, PyTorch will report `cuda: False` and training/inference will use **CPU** (no `--gpus` flag needed).

For a **smaller CPU-only image** (no CUDA toolkit in the base), build with `PYTORCH_IMAGE=pytorch/pytorch:2.5.1-runtime` (see **Build** below).

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

### CPU-only image (optional)

No NVIDIA hardware or drivers required:

```powershell
docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.5.1-runtime -t yolo26-detr:cpu .
```

Run without `--gpus`:

```powershell
docker run --rm yolo26-detr:cpu
```

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

The default base is `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (CUDA-capable; still runs on CPU if no GPU is passed). To override it (for example to match another CUDA minor on your driver):

```powershell
docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime -t yolo26-detr:cuda .
```

Choose a CUDA tag whose CUDA version is **supported by your NVIDIA driver** when you plan to use `--gpus all` (your driver’s `nvidia-smi` output lists a maximum CUDA version).

## Run the container (smoke test)

**CPU (no GPU flags):**

```powershell
docker run --rm yolo26-detr:cuda
```

**GPU (when NVIDIA + toolkit are available):**

```powershell
docker run --rm --gpus all yolo26-detr:cuda
```

The default command prints whether **CUDA** is available (`True`/`False`) and imports **Ultralytics** and **Transformers**. You should see `ok` in both cases; `cuda: True` only when the GPU is passed through to the container.

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
docker run --rm -v "C:\path\to\your\dataset:/dataset" yolo26-detr:cuda python3 train.py --model yolo26n --data /dataset/data.yaml --epochs 100 --imgsz 640 --batch 16 --project /dataset/runs --name exp1
```

Add **`--gpus all`** when you want GPU acceleration. **`train.py` defaults to `--device auto`** (uses GPU `0` if CUDA is available, otherwise `cpu`). Pass **`--device cpu`** or **`--device 0`** to force.

RT-DETR-L (default alias `rtdetr`):

```powershell
docker run --rm -v "C:\path\to\your\dataset:/dataset" yolo26-detr:cuda python3 train.py --model rtdetr --data /dataset/data.yaml --epochs 100 --batch 8
```

RT-DETR-X (larger; reduce `--batch` if you run out of VRAM on GPU):

```powershell
docker run --rm --gpus all -v "C:\path\to\your\dataset:/dataset" yolo26-detr:cuda python3 train.py --model rtdetr-x --data /dataset/data.yaml --epochs 100 --batch 4
```

Use a smaller **`--batch`** if you hit out-of-memory (especially for `yolo26m` or `rtdetr-x` on GPU).

## Run example inference (YOLO + DETR)

Mount a folder on your PC that contains an image and (optionally) your YOLO weights, then run `example_infer.py` inside the container.

**Windows (PowerShell), example:**

```powershell
docker run --rm -v "C:\path\to\your\data:/data" yolo26-detr:cuda python3 example_infer.py --image /data/photo.jpg --yolo-weights /data/best.pt --detr-model facebook/detr-resnet-50
```

Add **`--gpus all`** to use the GPU when available.

- **`--image`**: path **inside the container** (under `/data` if you mounted `C:\path\to\your\data` to `/data`).
- **`--yolo-weights`**: your `.pt` checkpoint or an Ultralytics-compatible weight name.
- **`--detr-model`**: any Hugging Face model id that exposes `AutoModelForObjectDetection` (swap for your “mobile” or custom DETR checkpoint).

## Useful commands

| Goal | Command |
|------|---------|
| List images | `docker images` |
| Open a shell in the image | `docker run --rm -it yolo26-detr:cuda bash` (add `--gpus all` for GPU) |
| Rebuild after code changes | `docker build -t yolo26-detr:cuda .` |

## Troubleshooting

- **`could not select device driver "" with capabilities: [[gpu]]`**  
  You used **`--gpus all`** but this host has no NVIDIA Container Toolkit / no GPU. Omit **`--gpus all`** for CPU, or install the toolkit and use an NVIDIA-capable environment.

- **`cuda: False` inside the container**  
  Normal when running **without** `--gpus all`, on a **CPU-only** image, or on a host without a visible GPU. Training and inference still work on **CPU** (slower). To use CUDA, run with **`--gpus all`** and a CUDA-capable base image on a machine with a supported GPU.

- **Build fails pulling `pytorch/pytorch`**  
  Check network and Docker Hub rate limits; retry later or use a mirror if your organization provides one.

## Project layout

| File | Purpose |
|------|---------|
| `Dockerfile` | PyTorch base (CUDA default or CPU via build-arg), deps, optional weight prefetch. |
| `requirements.txt` | Python packages (Ultralytics, Transformers, OpenCV headless, etc.). |
| `.dockerignore` | Reduces build context size (excludes venvs, large weights if present locally). |
| `prefetch_weights.py` | Build-time download of default checkpoints into the image cache. |
| `train.py` | Train `yolo26n` / `yolo26s` / `yolo26m` / RT-DETR (`rtdetr`, `rtdetr-l`, `rtdetr-x`) on a YOLO-format dataset YAML. |
| `scripts/download_kaggle_dataset.py` | Download [Kaggle skin-cancer](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer) into `/app/Datasets/PAD-UFES-20` using API credentials. |
| `example_infer.py` | Minimal example: YOLO predict + DETR forward on one image. |
