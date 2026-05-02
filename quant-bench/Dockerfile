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
# 패키지 설치 전략: 변경 확률이 낮은 것을 먼저 설치해 Docker layer cache를 최대화.
# 새 패키지 추가 시 파일 맨 아래 RUN 레이어를 추가하면 기존 캐시가 유지됨.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

LABEL description="CNN/RNN/Attention Quantization Benchmark — NVIDIA RTX A6000"

WORKDIR /workspace

# ── System packages ───────────────────────────────────────────────────────────
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

# ── pip toolchain ─────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Layer: Scientific computing (가장 stable — 버전 거의 고정) ────────────────
RUN pip install --no-cache-dir \
    "numpy>=1.24.0" \
    "scipy>=1.11.0"

# ── Layer: TensorRT (CUDA 버전에 묶임 — 거의 변경 없음) ──────────────────────
# trtexec CLI 포함 여부는 설치 후 `which trtexec`로 확인.
# 없으면 TensorRT Python API(import tensorrt)로 engine build/run 진행.
RUN pip install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    tensorrt-cu12

# ── Layer: ONNX ecosystem ─────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "onnx>=1.16.0" \
    "onnxruntime-gpu>=1.19.2"

# ── Layer: PyTorch ecosystem (base image PyTorch 버전에 묶임) ─────────────────
RUN pip install --no-cache-dir \
    torchvision \
    torchaudio

# ── Layer: torchao (PyTorch 2.5.x 호환 버전 고정) ────────────────────────────
# 0.8+ 은 torch>=2.6.0, 최신 버전은 torch>=2.11.0 필요.
# INT4 weight-only 사용 시 모델을 .to(torch.bfloat16)으로 먼저 변환할 것.
RUN pip install --no-cache-dir \
    torchao==0.7.0

# ── Layer: GPU monitoring ─────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "nvidia-ml-py>=12.560.30"

# ── Layer: Utility packages ───────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "tqdm>=4.66.0" \
    "rich>=13.7.0" \
    "psutil>=5.9.0" \
    "tabulate>=0.9.0"

# ── Layer: Visualization / data ───────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "matplotlib>=3.8.0" \
    "pandas>=2.0.0"

# ── Layer: Deep learning helpers ─────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "einops>=0.7.0" \
    "timm>=1.0.0"

# ── Layer: HuggingFace infrastructure ────────────────────────────────────────
# transformers / accelerate는 제외 (VLM 실험 단계가 아님).
RUN pip install --no-cache-dir \
    safetensors \
    tokenizers \
    huggingface_hub \
    datasets

# ── Layer: Jupyter ────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "jupyter>=1.0.0" \
    "ipykernel>=6.29.0"

# ── Layer: ONNX simplifier ────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    onnxsim

# ── Layer: PyCUDA ─────────────────────────────────────────────────────────────
# devel 이미지에 nvcc + CUDA 12.1 headers 포함 → 소스 빌드 또는 pre-built wheel 설치.
RUN pip install --no-cache-dir \
    pycuda

# ── Layer: nvitop (GPU process / resource monitor) ────────────────────────────
RUN pip install --no-cache-dir \
    nvitop

# ── Layer: trtexec CLI ────────────────────────────────────────────────────────
# trtexec is not shipped in the pip tensorrt wheel.
# libnvinfer-bin from the CUDA apt repo provides the binary at /usr/bin/trtexec.
# Version is pinned to match the pip tensorrt version (10.16.1.11).
# The +cuda13.2 build works on this host: driver 580 supports CUDA 12.9+.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer-bin=10.16.1.11-1+cuda13.2 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]
