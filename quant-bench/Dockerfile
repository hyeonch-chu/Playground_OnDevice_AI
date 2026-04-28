# CNN / RNN / Attention Quantization Benchmark Environment
# Target GPU: NVIDIA RTX A6000 (Ampere, sm_86)
# Backends: PyTorch FP32/FP16, torch.compile, torchao INT8/INT4, ONNX Runtime, TensorRT INT8
#
# Base image: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel  (Docker Hub, no NGC credentials required)
#   - CUDA 12.1 runtime  (host driver 580 supports CUDA 12.6 → 12.1 fully compatible)
#   - PyTorch 2.5.1
#   - Python 3.11
#   - cuDNN 9.x
#
# NOTE: NGC nvcr.io/nvidia/pytorch:24.12-py3 bundles TensorRT 10.x pre-installed
#       and is preferred when NGC credentials are available. Switch the FROM line
#       back to nvcr.io/nvidia/pytorch:24.12-py3 in that case.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

LABEL description="CNN/RNN/Attention Quantization Benchmark — NVIDIA RTX A6000"

WORKDIR /workspace

# Minimal system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip toolchain
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
# torch is pre-installed in the pytorch/pytorch base image;
# requirements.txt re-lists torchvision and torchaudio to ensure they are
# present and at a compatible version if not already installed.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

CMD ["/bin/bash"]
