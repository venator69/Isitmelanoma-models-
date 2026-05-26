# GPU image for YOLO26 (Ultralytics) + DETR (Hugging Face Transformers).
#
# Prerequisites (host):
#   - NVIDIA driver
#   - Docker with NVIDIA Container Toolkit (Linux / WSL2 with GPU)
#
# Build:
#   docker build -t yolo26-detr:cuda .
#
# Run (GPU):
#   docker run --rm --gpus all yolo26-detr:cuda
#
# Run with your weights + image:
#   docker run --rm --gpus all -v C:\path\to\data:/data yolo26-detr:cuda \
#     python3 example_infer.py --image /data/img.jpg --yolo-weights /data/best.pt
#
# Train (YOLO-format images + labels; see README + dataset.example.yaml):
#   docker run --rm --gpus all -v C:\path\to\dataset:/dataset yolo26-detr:cuda \
#     python3 train.py --model yolo26n --data /dataset/data.yaml --epochs 100 --batch 16
#   # --model: yolo26n | yolo26s | yolo26m | rtdetr | rtdetr-l | rtdetr-x
#
# Download skin-cancer dataset from Kaggle (not in git; see README):
#   docker run --rm -v %USERPROFILE%\.kaggle\kaggle.json:/root/.kaggle/kaggle.json:ro yolo26-detr:cuda \
#     python3 scripts/download_kaggle_dataset.py

ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Base image already includes CUDA-enabled torch/torchvision.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

# Cache yolo26n/s/m + RT-DETR-L weights in the image (requires network at build time).
# Skip with: docker build --build-arg SKIP_PREFETCH=1 -t yolo26-detr:cuda .
ARG SKIP_PREFETCH=0
RUN if [ "$SKIP_PREFETCH" != "1" ]; then python3 /app/prefetch_weights.py; fi

CMD ["python3", "-c", "import torch; print('cuda:', torch.cuda.is_available(), torch.version.cuda); import ultralytics; from transformers import AutoModel; print('ok')"]
