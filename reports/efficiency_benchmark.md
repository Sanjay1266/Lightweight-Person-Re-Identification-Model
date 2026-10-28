# Computational Efficiency & Lightweight Benchmark Report

**Project**: Lightweight Person Re-Identification Model (DCR-ReID)  
**Date**: 2026-10-28 20:36:20 IST  
**Hardware Platforms**:
- Host CPU: AMD / Intel x86_64 Host Processor (Windows 11)
- Hardware Accelerator: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB physical VRAM, SM 8.9)
**Benchmarking Protocol**: 25 warmup iterations followed by 100 synchronized timed iterations using `torch.cuda.synchronize()` and `time.perf_counter()`.

---

## 1. Architectural Parameter Decomposition

Total parameters: **9,752,277** (Trainable: **9,751,253**, Non-trainable BN running statistics: 1,024).

| Architectural Module | Parameter Count | Percentage of Model | Functional Role in DCR-ReID |
|:---|:---:|:---:|:---|
| **Residual Backbone (Stages 1–4)** | 4,897,392 | 50.22% | Deep spatial feature extraction ($P_i$) with stride-1 Stage 4 |
| **CRD Disentanglement Decoders** | 4,166,659 | 42.72% | $\psi^-, \psi^+, \psi^t$ 4-block decoders for visual mask reconstruction |
| **DAD Dual Attention Module** | 558,178 | 5.72% | CBAM Channel Attention + 7×7 Spatial Attention feature refinement |
| **Dual BNNeck & Linear Classifiers**| 130,048 | 1.33% | Metric projection, $L_2$ unit normalization, identity & clothes heads |
| **Total Model Footprint** | **9,752,277** | **100.0%** | Full Lightweight DCR-ReID Network |

Checkpoint disk storage:
- `checkpoints/best_model.pth`: **39,117,107 bytes (37.30 MB)**
- `checkpoints/experiments/dcr_reid_cuda_v1/best_model.pth`: **39,323,744 bytes (37.50 MB)**

---

## 2. Theoretical Computational Complexity (FLOPs & MACs)

Evaluated at standard Person Re-ID input resolution ($256 \times 128 \times 3$) using PyTorch's official `FlopCounterMode`:

- **Counting Convention**: $1 \text{ MAC (Multiply-Accumulate)} = 2 \text{ FLOPs}$ (standard IEEE convention).
- **Exact FLOPs**: **3,001,491,968 FLOPs (3.0015 GFLOPs)**.
- **Exact MACs**: **1,500,745,984 MACs (1.5007 GMACs)**.

### Comparison Against Standard ResNet-50 Baseline:
| Architecture | Input Resolution | Parameter Count | FLOPs (GFLOPs) | Computational Reduction |
|:---|:---:|:---:|:---:|:---:|
| **Standard ResNet-50 Re-ID** | $256 \times 128$ | 25.56 M | 8.20 GFLOPs | Baseline reference (1.0x) |
| **Lightweight DCR-ReID (Ours)**| $256 \times 128$ | **9.75 M** | **3.00 GFLOPs** | **63.4% FLOP Reduction (-5.20 GFLOPs)** |

---

## 3. Empirical Inference Latency & Throughput Benchmark

Timing measurements isolate neural network inference on CPU and CUDA. All timers wrapped with `torch.cuda.synchronize()` on CUDA to prevent asynchronous pipelining bias.

| Device Platform | Execution Mode | Median Latency (ms) | Mean Latency (ms) | p95 Latency (ms) | Throughput (FPS) | Amortized Time / Image |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Host CPU (Single Thread)** | Batch Size = 1 | 31.44 ms | 33.33 ms | 61.33 ms | **31.8 FPS** | 31.44 ms |
| **Host CPU** | Batch Size = 32 | — | — | — | **57.2 FPS** | 17.48 ms |
| **NVIDIA RTX 4060 GPU** | Batch Size = 1 | **11.07 ms** | **11.45 ms** | **13.63 ms** | **90.3 FPS** | 11.07 ms |
| **NVIDIA RTX 4060 GPU** | Batch Size = 32 | — | — | — | **2,694.9 FPS** | **0.371 ms** |

---

## 4. Hardware VRAM & Memory Footprint

Measured via `torch.cuda.max_memory_allocated()` during high-throughput batched inference on CUDA:

- **Active Model Weight Footprint**: ~37.5 MB
- **Single-Image Inference Allocation**: 49.3 MB
- **Batched (BS=32) Peak VRAM Allocated**: **517.16 MB**
- **Batched (BS=32) Peak VRAM Reserved**: **556.00 MB**
- **Hardware Capacity Utilized**: **6.3%** of 8,188 MiB total VRAM (over 7,600 MB headroom remaining).

---

## 5. End-to-End Surveillance Pipeline Context

> [!NOTE]
> The single-image neural network inference latency of **11.07 ms on GPU** (90.3 FPS) and batched throughput of **2,694.9 FPS** measure the forward embedding computation $f = \mathcal{M}(x)$.
> In real-world edge or smart-city multi-target surveillance deployments, the total frame processing latency also encompasses:
> 1. **Upstream Video Decoding & Ingestion**: RTSP stream H.264/H.265 decoding (~2–4 ms).
> 2. **Object Detection**: YOLOv8/v10 person bounding-box detection (~8–15 ms).
> 3. **Multi-Object Tracking**: SORT/ByteTrack identity association (~1–3 ms).
> 4. **Image Preprocessing**: Cropping, aspect-ratio padding, resizing to $256 \times 128$, normalization (~1–2 ms).
> 5. **Re-ID Feature Extraction**: Lightweight DCR-ReID ($0.37$ ms per crop when batched).
> 6. **Gallery Index Search**: Cosine similarity matching against enrolled gallery identities (~0.5–2 ms).
>
> Lightweight DCR-ReID's 0.37 ms amortized extraction ensures that the Re-ID module never constitutes the system bottleneck.

