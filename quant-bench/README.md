# CNN / RNN / Attention Quantization Benchmark Environment

**Target GPU:** NVIDIA RTX A6000 (Ampere, sm_86)  
**Scope:** INT8 / INT4 quantization + multi-backend inference comparison for CNN, RNN, Attention toy models  
**This repo provides environment setup only — no benchmark scripts are included.**

---

## 1. Purpose

Establish a reproducible Docker environment for benchmarking quantized inference of small/toy CNN, RNN, and Attention models on an **NVIDIA RTX A6000 (Ampere, sm_86)** GPU.

Key goals:

- Compare FP32 / FP16 / INT8 / INT4 precision across multiple backends
- Measure: **inference latency**, **throughput**, **GPU memory usage**, **power draw**, **energy per inference**
- Validate torchao INT4 weight-only quantization for Linear-heavy Attention layers
- Identify which quantization modes deliver real kernel-level speedup on Ampere

> VLM (Vision-Language Models) are out of scope for this environment.

---

## 2. Included Backends

| # | Backend | Precision | Notes |
|---|---------|-----------|-------|
| 1 | PyTorch eager | FP32 | Reference baseline |
| 2 | PyTorch eager | FP16 | `.half()` — activates Tensor Cores |
| 3 | `torch.compile` (Inductor) | FP32 / FP16 | Kernel fusion via Triton |
| 4 | torchao | INT8 | `int8_dynamic_activation_int8_weight` |
| 5 | torchao | INT4 weight-only | `int4_weight_only()` — Linear layers first |
| 6 | ONNX Runtime | FP32 / FP16 | CUDA execution provider |
| 7 | TensorRT | INT8 | `trtexec` or Python TRT API (optional) |
| 8 | Torch-TensorRT | INT8 | `torch_tensorrt.compile()` (optional) |

### Excluded from default targets

> **TensorRT FP4 / NVFP4** — Blackwell (sm_100+) required.  
> The RTX A6000 is Ampere (sm_86). TensorRT's FP4/NVFP4 quantization paths (introduced for H100/B200-class GPUs) require native FP4 tensor cores that are not present on sm_86. Targeting FP4 on A6000 will either fall back silently to FP16 or fail. Do not set FP4 as a benchmark goal on this hardware.

---

## 3. INT8 / INT4 Experiment Scope

### INT8

Broadly supported across all model types via torchao and TensorRT.

| Model | Layer | INT8 |
|-------|-------|:----:|
| CNN | Conv2d, BatchNorm, Linear | ✅ Primary target |
| RNN | LSTM / GRU / RNN cells, Linear | ✅ (Linear layers) |
| Attention | QKV Linear, Output Linear, FFN Linear | ✅ Primary target |

### INT4 (torchao weight-only)

INT4 support is **not uniform across layer types**. `torchao`'s `int4_weight_only()` primarily targets `nn.Linear`. Apply it to Attention first.

| Model | Layer | INT4 | Status |
|-------|-------|:----:|--------|
| Attention | QKV Linear | ✅ | Primary target — highest impact |
| Attention | Output Linear | ✅ | Primary target |
| Attention | FFN Linear | ✅ | Primary target |
| CNN | Linear (classifier head) | ✅ | Secondary |
| RNN | Linear projection | ✅ | Secondary — limited gain expected |
| CNN | Conv2d | ⚠️ | **Verify support** — see §10 |
| RNN | Recurrent cells (LSTM/GRU) | ⚠️ | **Verify support** — see §10 |

---

## 4. Build Command

```bash
cd quant-bench
docker build -t quant-bench:latest .
```

Expected build time: 5–15 minutes (network-dependent).

> **Prerequisite:** `nvidia-container-toolkit` must be installed on the host.

---

## 5. Run Commands

### Interactive shell

```bash
docker run --gpus all --rm -it \
    -v $(pwd):/workspace \
    --shm-size=16g \
    quant-bench:latest
```

### Jupyter Notebook

```bash
docker run --gpus all --rm -it \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    --shm-size=16g \
    quant-bench:latest \
    jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

Then open `http://localhost:8888` (copy the token from container stdout).

### Verify GPU access inside container

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
nvidia-smi
```

---

## 6. Metrics to Measure on A6000

| Metric | Unit | Measurement Method |
|--------|------|--------------------|
| Latency | ms / inference | `torch.cuda.Event` (start/end, `elapsed_time()`) |
| Throughput | samples / sec | `batch_size / latency_s` |
| GPU Memory | MiB | `torch.cuda.max_memory_allocated()` or `nvidia-smi` |
| Power Draw | W | `nvidia-smi dmon` or `pynvml.nvmlDeviceGetPowerUsage()` |
| Energy per Inference | mJ | `avg_power_W × latency_s × 1000` |

**Warm-up is mandatory** — discard the first 10–50 iterations before recording any metric.  
Power draw in particular needs 3–5 seconds of stable load before sampling.

---

## 7. nvidia-smi Example Commands

```bash
# One-shot GPU status
nvidia-smi

# Power draw only, every 1 second
nvidia-smi dmon -s p -d 1

# SM utilization + memory + power + temperature, every 1 second
nvidia-smi dmon -s u,m,p,t -d 1

# Detailed power query
nvidia-smi -q -d POWER

# Detailed memory query
nvidia-smi -q -d MEMORY

# Watch mode, refresh every 2 seconds
watch -n 2 nvidia-smi

# Log to CSV during a benchmark run
nvidia-smi dmon -s u,m,p -d 1 -f gpu_log.csv

# Persistent mode (reduces first-inference latency spike)
sudo nvidia-smi -pm 1

# Lock GPU clocks for reproducible latency (optional — requires root)
sudo nvidia-smi --lock-gpu-clocks=<min>,<max>
```

---

## 8. Python Power Measurement with pynvml

`pynvml` exposes the NVML API, allowing power and memory monitoring **from within the benchmark process** without spawning `nvidia-smi` as a subprocess.

```python
import pynvml
import time

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0

# Instantaneous power (milliwatts → watts)
power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
print(f"Power: {power_w:.1f} W")

# Sample power over a measurement window
def measure_avg_power(handle, duration_s=2.0, interval_s=0.05):
    samples = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        time.sleep(interval_s)
    return sum(samples) / len(samples)

# Example: energy per inference
latency_s = 0.005          # measure separately with torch.cuda.Event
avg_power = measure_avg_power(handle, duration_s=2.0)
energy_mj = avg_power * latency_s * 1000
print(f"Avg power: {avg_power:.1f} W  |  Energy/inference: {energy_mj:.3f} mJ")

pynvml.nvmlShutdown()
```

> `nvmlDeviceGetPowerUsage()` returns milliwatts. Sampling at 50 ms intervals is usually sufficient for stable averages. Always run the sampler **after warm-up**.

---

## 9. Recommended Experiment Order

Run in this order to build a clean comparison baseline before adding complexity:

| Step | Backend | What it tests |
|------|---------|---------------|
| 1 | **PyTorch FP32** | Reference — establishes absolute baseline |
| 2 | **PyTorch FP16** | Tensor Core activation, memory halving |
| 3 | **`torch.compile`** | Inductor kernel fusion over FP16 baseline |
| 4 | **torchao INT8** | `int8_dynamic_activation_int8_weight` via `quantize_()` |
| 5 | **torchao INT4 weight-only** | `int4_weight_only()` via `quantize_()` — Attention Linear layers first |
| 6 | **ONNX Runtime GPU** | Export → `.onnx` → CUDA execution provider |
| 7 | **TensorRT INT8** | `trtexec --int8` or Python TRT API (optional install) |
| 8 | **Torch-TensorRT INT8** | `torch_tensorrt.compile(model, ...)` (optional install) |

> Complete steps 1–5 (pure Python, no optional installs) before moving to TRT-based steps. This ensures a solid CPU/GPU baseline before introducing TRT build complexity.

### torchao quick reference

```python
import torch
from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, int4_weight_only

model = MyModel().cuda().eval()

# INT8
quantize_(model, int8_dynamic_activation_int8_weight())

# INT4 weight-only (Linear layers only)
quantize_(model, int4_weight_only())

# With torch.compile (recommended for INT4 kernel fusion)
model = torch.compile(model, mode="max-autotune")
```

---

## 10. INT4 Experiment Notes

### Primary targets (Linear layers in Attention)

Apply `int4_weight_only()` here first — these are the largest `nn.Linear` modules and yield the highest memory bandwidth reduction:

- **QKV projection** (`nn.Linear(d_model, 3 * d_model)`)
- **Output projection** (`nn.Linear(d_model, d_model)`)
- **FFN layers** (`nn.Linear(d_model, d_ff)`, `nn.Linear(d_ff, d_model)`)

### Support verification required

| Layer | Issue |
|-------|-------|
| **CNN `Conv2d` INT4** | `torchao int4_weight_only()` targets `nn.Linear`. `nn.Conv2d` requires explicit Conv INT4 kernel support, which is limited in current torchao versions. **Check torchao release notes before benchmarking.** |
| **RNN recurrent cells INT4** | LSTM/GRU internal operations are not standard `nn.Linear` calls. INT4 benefit is expected to be minimal, and kernel availability is not guaranteed. Apply INT4 only to Linear projection layers within RNN models. |

### A6000-specific INT4 behavior

- **Memory reduction is achievable**: INT4 stores weights in 4 bits → ~4× memory savings vs FP32 for weight tensors.
- **Latency reduction is not guaranteed**: actual speedup requires a fused dequant+GEMM kernel. Without `torch.compile`, INT4 may be slower than FP16.
- **Use `torch.compile(mode="max-autotune")`** with torchao INT4 for Triton-based kernel fusion.
- **Do not target TensorRT FP4/NVFP4**: these require Blackwell (sm_100+) hardware not present on A6000.

### RNN GPU quantization note

RNN (LSTM, GRU) inference on GPU is sequential and often memory-bound in a way that is different from matrix multiplications. GPU utilization is typically low for small sequence lengths. **Quantization benefit (INT8 or INT4) is expected to be smaller than for CNN/Attention**, and may not exceed measurement noise. Document this expectation before benchmarking.

---

## 11. Optional Install

These packages are **excluded from `requirements.txt`** due to strict version coupling between CUDA, PyTorch, and TensorRT. Install them manually inside the running container after verifying your exact versions.

> The NGC `24.12-py3` base image already includes **TensorRT 10.x**. For `trtexec` or the Python TRT API, TensorRT may already be available. Run `python -c "import tensorrt; print(tensorrt.__version__)"` to check before installing.

### TensorRT (if not already in NGC image)

```bash
# Replace version with one matching your CUDA version
pip install tensorrt==10.7.0.post1 \
    --extra-index-url https://pypi.nvidia.com

# Verify
python -c "import tensorrt; print(tensorrt.__version__)"
trtexec --version
```

### Torch-TensorRT

Must match **both** the PyTorch version and the TensorRT version in the container.

```bash
# For PyTorch 2.5.0 + CUDA 12.4/12.6
pip install torch-tensorrt==2.5.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Verify
python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
```

### NVIDIA Model Optimizer (nvidia-modelopt)

Provides PTQ (post-training quantization) calibration pipelines for TensorRT INT8. Useful for calibrating CNN/Attention models before TRT engine export.

```bash
pip install nvidia-modelopt

# Verify
python -c "import modelopt; print(modelopt.__version__)"
```

> **Warning:** These optional packages can conflict with each other or with pre-installed NGC libraries. If installation fails, try `pip install --no-deps` and resolve dependencies manually, or use a different NGC container tag.

---

## 12. Caveats

### Quantization does not always reduce latency

Quantization reduces model weight size and memory bandwidth pressure, but actual latency improvement depends on:
- Whether a specialized quantized kernel (INT8 GEMM, INT4 GEMM) is available for the specific layer type
- Batch size (small batches are often memory-latency-bound, not compute-bound)
- Model size (very small models may see overhead exceed the gain)

Always compare quantized vs. FP16 baseline — not FP32 — since FP16 already activates Tensor Cores on A6000.

### INT4 latency improvement requires kernel support

`torchao int4_weight_only()` stores weights in 4-bit integers but dequantizes to a higher precision for the actual GEMM computation. A latency reduction is only realized when a **fused dequant+GEMM kernel** is invoked (e.g., via `torch.compile` + Triton codegen). Without kernel fusion, INT4 may be **slower than FP16** due to dequantization overhead.

### RNN GPU quantization benefit is limited

LSTM/GRU operations are sequential and cannot be fully parallelized across time steps. GPU utilization is often low for common sequence lengths, making quantization overhead more visible. Expect smaller (or no) improvement vs. CNN/Attention baselines.

### Measure only after warm-up

GPU clock frequencies, power state, and memory allocations stabilize after the first several inference passes. **Discard the first 10–50 iterations** before recording any metric. For power/energy measurements, allow at least 3–5 seconds of stable load before sampling.

### TensorRT FP4 / NVFP4 requires Blackwell (sm_100+)

TensorRT's FP4/NVFP4 quantization paths are designed for the FP4 tensor cores introduced in Blackwell-generation GPUs (H100, B200, GB200). The RTX A6000 is Ampere (sm_86) and lacks this hardware capability. Setting FP4 as a TensorRT calibration target on A6000 will either silently fall back to FP16 or raise an error. Do not use FP4/NVFP4 as a benchmark target on this hardware.

### CNN Conv2d INT4 and RNN recurrent cell INT4 — verify before benchmarking

These layer types are not the primary targets of `torchao int4_weight_only()` (which targets `nn.Linear`). Backend and kernel support for Conv2d INT4 and LSTM/GRU cell INT4 may be unavailable or incomplete in current library versions. Always verify by checking torchao and TensorRT release notes before including these in a benchmark run.

---

## Environment Summary

| Component | Value |
|-----------|-------|
| Base image | `nvcr.io/nvidia/pytorch:24.12-py3` |
| CUDA | 12.6.x |
| PyTorch | 2.5.0 |
| Python | 3.10 |
| torchao | >= 0.7.0 |
| onnxruntime-gpu | >= 1.19.2 |
| Target GPU | NVIDIA RTX A6000 (Ampere, sm_86) |
| TensorRT | Pre-installed in NGC image (10.x) |
| Torch-TensorRT | Optional — install separately |
