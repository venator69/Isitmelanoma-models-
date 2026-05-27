# PyTorch image for YOLO26 (Ultralytics) + DETR (Hugging Face Transformers).
# GPU is optional: use default CUDA base for NVIDIA hosts, or a CPU-only base (see below).
#
# Prerequisites (host):
#   - Docker
#   - Optional: NVIDIA driver + NVIDIA Container Toolkit for `docker run --gpus all`
#
# Build (CUDA, default):
#   docker build -t yolo26-detr:cuda .
#
# Build (CPU-only, smaller image — no NVIDIA required):
#   docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.5.1-runtime -t yolo26-detr:cpu .
#
# Run (CPU — no --gpus):
#   docker run --rm yolo26-detr:cuda
#
# Run (GPU when available):
#   docker run --rm --gpus all yolo26-detr:cuda
#
# Run with your weights + image:
#   docker run --rm -v C:\path\to\data:/data yolo26-detr:cuda \
#     python3 example_infer.py --image /data/img.jpg --yolo-weights /data/best.pt
#
# Train (YOLO-format images + labels; see README + dataset.example.yaml):
#   docker run --rm -v C:\path\to\dataset:/dataset yolo26-detr:cuda \
#     python3 train.py --model yolo26n --data /dataset/data.yaml --epochs 100 --batch 16
#   # --model: yolo26n | yolo26s | yolo26m | rtdetr | rtdetr-l | rtdetr-x
#
# Download skin-cancer dataset from Kaggle (not in git; see README):
#   docker run --rm -v %USERPROFILE%\.kaggle\kaggle.json:/root/.kaggle/kaggle.json:ro yolo26-detr:cuda \
#     python3 scripts/download_kaggle_dataset.py

ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Base image includes torch/torchvision (CUDA or CPU depending on PYTORCH_IMAGE).
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# OpenCV (via ultralytics) needs these at import time; base PyTorch image omits them.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libgl1 libglib2.0-0 libsm6 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Cache yolo26n/s/m + RT-DETR weights in the image (requires network at build time).
# Skip with: docker build --build-arg SKIP_PREFETCH=1 -t yolo26-detr:cuda .
ARG SKIP_PREFETCH=0
RUN if [ "$SKIP_PREFETCH" != "1" ]; then python3 /app/prefetch_weights.py; fi

CMD ["python3", "-c", "import torch; print('cuda:', torch.cuda.is_available(), getattr(torch.version, 'cuda', None)); import ultralytics; from transformers import AutoModel; print('ok')"]
