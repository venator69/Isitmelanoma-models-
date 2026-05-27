# YOLO26 + DETR (Docker)

PyTorch image with **Ultralytics** (YOLO26, RT-DETR) and **Transformers** (DETR). **GPU:** add `--gpus all` when you have NVIDIA + [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html); otherwise runs on **CPU**.

## Build

From this repo folder:

```powershell
cd "c:\Users\denni\Desktop\isitmelanoma-docker"
docker build -t yolo26-detr:cuda .
```

Skip prefetching weights at build time:

```powershell
docker build --build-arg SKIP_PREFETCH=1 -t yolo26-detr:cuda .
```

CPU-only base: `--build-arg PYTORCH_IMAGE=pytorch/pytorch:2.5.1-runtime -t yolo26-detr:cpu .`

## Run

```powershell
docker run --rm yolo26-detr:cuda
docker run --rm --gpus all yolo26-detr:cuda
docker run --rm -it yolo26-detr:cuda bash
```

## Train

Dataset layout: `data.yaml` plus `images/{train,val,test}/` and `labels/{train,val,test}/` (see `dataset.example.yaml`). Mount a **real** host folder that contains `data.yaml` (not a placeholder path).

```powershell
cd "c:\Users\denni\Desktop\isitmelanoma-docker"
docker run --rm --shm-size=2g --gpus all -v "$(Get-Location)\Datasets\PAD-UFES-20-yolo:/dataset" yolo26-detr:cuda python3 train.py --data /dataset/data.yaml --model yolo26n --epochs 100 --batch 16
```

- Omit `--gpus all` for CPU.
- `--shm-size` helps with Docker DataLoader shared memory; **`workers` defaults to `0` inside Docker** (override with `--workers N`).
- `train.py` writes a short-lived YAML with an absolute `path`, points Ultralytics at that file, and sets cwd to the dataset folder (image `WORKDIR` is `/app`).

**Rebuild** the image after updating `train.py`, or mount the file: `-v "$(Get-Location)\train.py:/app/train.py:ro"`.

**Models:** `yolo26n` | `yolo26s` | `yolo26m` | `rtdetr` / `rtdetr-l` | `rtdetr-x`. **`--device`** default is `auto`. **`--weights`** overrides the preset. Save runs on the host: add `-v "$(Get-Location)\runs:/app/runs"`.

## Convert dataset

PAD-UFES, COCO, VOC, YOLO TXT, or LabelMe → YOLO layout + `data.yaml`:

```powershell
python scripts/convert_dataset.py --input Datasets/PAD-UFES-20 --output Datasets/PAD-UFES-20-yolo --format pad-ufes --split-ratios 0.8 0.1 0.1 --seed 42 --overwrite
```

Same with hardlinks on Windows (saves disk): add `--copy-mode hardlink`. Other formats: `--format auto` or `coco` / `voc` / `yolo` / `labelme`. PAD-UFES uses image-level labels; output uses one full-image box per image for detection training.

## Kaggle download

Needs `%USERPROFILE%\.kaggle\kaggle.json`. Data goes under `Datasets/` (gitignored).

```powershell
docker run --rm -v "$env:USERPROFILE\.kaggle\kaggle.json:/root/.kaggle/kaggle.json:ro" -v "$(Get-Location)\Datasets:/app/Datasets" yolo26-detr:cuda python3 scripts/download_kaggle_dataset.py
```

`--force` re-downloads. If `metadata.csv` and images already exist, download is skipped.

## Inference

```powershell
docker run --rm --gpus all -v "$(Get-Location)\runs\train\exp\weights:/weights" -v "$(Get-Location)\some_image.jpg:/data/photo.jpg:ro" yolo26-detr:cuda python3 example_infer.py --image /data/photo.jpg --yolo-weights /weights/best.pt
```

## Tips

| Issue | What to try |
|--------|-------------|
| `Dataset YAML not found` | Fix the host path in `-v ...:/dataset` so `data.yaml` exists there. |
| `/app/images/val` not found | Use this repo’s `train.py` (rebuild image), or `docker run -w /dataset` if using `yolo` CLI directly. |
| Ultralytics config warning | Harmless; or set `YOLO_CONFIG_DIR=/tmp/Ultralytics`. |
| DataLoader / shm errors | `docker run --shm-size=2g ...`; in Docker, `train.py` defaults `--workers` to `0` unless you set `--workers N`. |

## Main files

| Path | Role |
|------|------|
| `Dockerfile` | Build image |
| `requirements.txt` | Python deps |
| `train.py` | Train YOLO26 / RT-DETR |
| `example_infer.py` | YOLO + HF DETR demo |
| `scripts/download_kaggle_dataset.py` | Kaggle → `Datasets/` |
| `scripts/convert_dataset.py` | Formats → YOLO layout |
| `prefetch_weights.py` | Optional build-time weights |

Do not commit large files under `Datasets/`; keep them gitignored.
