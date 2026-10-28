# DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification
### 23CSE373 - Computer Vision | Amrita School of Engineering

An advanced Computer Vision engineering implementation of **Deep Component Reconstruction for Cloth-Changing Person Re-Identification (DCR-ReID)** based on the IEEE TCSVT 2023 research paper.

This repository features a **Dual-Track Architecture**:
1. **Track A (Paper-Faithful DCR-ReID)**: Rigorous reference implementation reproducing the exact IEEE TCSVT 2023 paper specifications (ImageNet-pretrained ResNet-50 backbone with layer4 stride-1, 384×192 input resolution, 2560-channel CRD feature decomposition, 4-block residual decoders $\psi^v$ with Tanh activation, stacked max+avg global pooling, 5120-D DAD feature assembly, 4096-D clothes-irrelevant inference, and two-stage loss optimization with adversarial target distributions $q(c)$ and $h(c)$).
2. **Track B (Lightweight DCR-ReID Edge Variant)**: Edge-optimized engineering variant (9.75M parameters, 3.00 GFLOPs at 256×128, dual BNNeck metric neck, 512-D L2-normalized embedding, 2,785.7 FPS batched throughput) designed for resource-constrained real-time surveillance hardware (NVIDIA Jetson, Intel NUC, smart cameras).

---

## 👥 Project Information & Authors
- **Course**: 23CSE373 - Computer Vision
- **Base Paper**: *DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification* (IEEE Transactions on Circuits and Systems for Video Technology, Vol. 33, No. 8, pp. 4172–4184, August 2023)
- **Official Paper Repository**: `https://github.com/PKU-ICST-MIPL/DCR-ReID_TCSVT2023`
- **Team Members**:
  - **G N Bhuvaneshwaran** (`CB.SC.U4CSE24218`)
  - **Sanjay MS** (`CB.SC.U4CSE24248`)
  - **Sanjay S** (`CB.SC.U4CSE24249`)

---

## 🌟 Architecture & Methodological Comparison

```
+---------------------------------------------------------------------------------------------------------+
|                                    INPUT IMAGE (384×192 or 256×128)                                     |
|                                                    ↓                                                    |
|                                       BACKBONE FEATURE EXTRACTION                                       |
|                  Track A: ResNet-50 (Stride-1 Stage 4)  |  Track B: Lightweight 4-Stage CNN              |
|                                                    ↓                                                    |
|                                  CRD: COMPONENT RECONSTRUCTION                                          |
|                Track A: 2560 channels (1024 ⊕ 512 ⊕ 1024) | Track B: 512 channels (204 bytes)           |
|                                                    ↓                                                    |
|         ┌──────────────────────────────────────────┼──────────────────────────────────────────┐         |
|         ↓                                          ↓                                          ↓         |
|   [PI Branch]                                [CR Branch]                                [CI Branch]     |
|   Person Identification                 Component Reconstruction                   Clothes Ident        |
|   - Stacked Pooling φ(P_i^v)            - 4-Block Residual Decoders ψ^v            - Clothes Head C_P   |
|   - Identity Classifier C_ID            - Tanh Activation [-1, 1]                  - Adv Loss L_ca      |
|   - Label Smooth CE L_ID                - L1 Reconstruction Loss L_R               - DAD Assembly G_i   |
|                                                    ↓                                                    |
|                                         INFERENCE REPRESENTATION                                        |
|   Track A: Discards F_i^-; Outputs L2-normalized F_i^+ ⊕ F_i^t (4096 dimensions)                       |
|   Track B: Invariant Inference via Dual BNNeck (512 dimensions)                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Key Differences Matrix

| Architectural Attribute | Track A: Paper-Faithful DCR-ReID | Track B: Lightweight DCR-ReID |
|---|---|---|
| **Backbone Network** | ResNet-50 (ImageNet-pretrained, layer4 stride-1) | Custom 4-stage Residual CNN (4.90M params) |
| **Input Resolution** | $384 \times 192$ (Aspect Ratio 2:1) | $256 \times 128$ (Aspect Ratio 2:1) |
| **CRD Channel Budget** | 2560 channels ($P_i^+: 1024, P_i^-: 512, P_i^t: 1024$) | 512 channels ($P_i^+: 204, P_i^-: 104, P_i^t: 204$) |
| **Component Decoders** | 3 independent 4-block residual decoders $\psi^v$ | 3 residual component decoders |
| **Decoder Activations** | Tanh activation $[-1, 1]$ matching binary masks | Tanh activation $[-1, 1]$ |
| **Feature Aggregation** | Stacked Max-Pooling + Avg-Pooling $\phi(P_i^v)$ | Dual Attention (CBAM Channel + Spatial) |
| **Feature Assembly ($G_i$)** | Concatenation $F_i^+ \oplus F_{ai}^- \oplus F_i^t \in \mathbb{R}^{5120}$ | Feature concatenation with attention modulation |
| **Inference Representation** | **4096 dimensions** ($F_i^+ \oplus F_i^t$, clothes discarded) | **512 dimensions** ($f_{id}$, clothes-invariant) |
| **Total Model Parameters** | **~31.6 M** | **9.75 M** (69.1% parameter reduction) |
| **Computational FLOPs** | **~8.50 GFLOPs** (at $384 \times 192$) | **3.00 GFLOPs** (at $256 \times 128$) |
| **Batched Throughput** | ~1,200 FPS on RTX 4060 GPU | **2,785.7 FPS** on RTX 4060 GPU |
| **Primary Use Case** | Server clusters, research verification, maximum accuracy | Real-time edge surveillance, embedded IoT devices |

---

## 🎯 Two-Stage Optimization Protocol

Both tracks implement the two-stage optimization methodology defined in base paper Section III.B.4:

### Stage 1: Disentanglement Initialization
Establishes baseline identity and clothing classification boundaries while training component decoders to reconstruct anatomical, apparel, and contour masks:
$$\mathcal{L}_{\text{stage1}} = \mathcal{L}_c + \mathcal{L}_{ID} + \mathcal{L}_R$$

### Stage 2: Adversarial & Assembled Disentanglement
Activates the clothes adversarial loss $\mathcal{L}_{ca}$ to penalize clothing correlation in identity features, along with assembled losses $\mathcal{L}_{ac}$ and $\mathcal{L}'_{ID}$ over shuffled clothing features:
$$\mathcal{L}_{\text{stage2}} = \mathcal{L}_{ID} + \mathcal{L}_c + \mathcal{L}_{ca} + \alpha \mathcal{L}_{ac} + \gamma \mathcal{L}'_{ID} + \mathcal{L}_R$$

Hyperparameters (verified paper defaults for PRCC):
- $\alpha = 0.05$ (Weight of assembled clothes loss)
- $\gamma = 1.0$ (Weight of assembled identity loss)
- $\epsilon = 0.1$ (Label smoothing parameter)

Adversarial target distributions:
- **$q(c)$ for $\mathcal{L}_{ca}$ (Eq. 13-14)**: Uniform over intra-identity clothes classes ($\frac{\epsilon}{K}$ for $c \neq c_i$, $1 - \epsilon + \frac{\epsilon}{K}$ for $c = c_i$).
- **$h(c)$ for $\mathcal{L}_{ac}$ (Eq. 17-18)**: Uniform distribution ($\frac{1}{K}$) over all clothing categories belonging to identity $i$.

---

## 📊 Benchmark Results

### 1. Published Paper Reported Benchmarks (IEEE TCSVT 2023 Table I)

| Benchmark Protocol | Condition | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) | mAP (%) |
|---|---|---:|---:|---:|---:|
| **`prcc_cc`** | PRCC Cloth-Changing (Cam A $\to$ Cam C) | **57.20** | **78.40** | **85.10** | **57.40** |
| **`prcc_sc`** | PRCC Same-Clothes (Cam A $\to$ Cam B) | **85.40** | **95.80** | **97.60** | **83.60** |
| **`prcc_all`** | PRCC Combined (Cam A $\to$ Cam B + C) | **72.80** | **88.50** | **92.30** | **68.50** |

### 2. Track B Empirical Benchmarks: Baseline Checkpoint vs. CUDA-Optimized

Evaluated across all 7 benchmark subsets under identical cosine distance metrics and Market-1501 junk exclusion rules:

| Dataset Subset | Surveillance Condition | Baseline Rank-1 | Baseline mAP | CUDA-Optimized Rank-1 | CUDA-Optimized mAP | Absolute Gain (Rank-1) | Absolute Gain (mAP) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`both_small`** | Combined Small Subset | 22.54% | 10.71% | **68.49%** | **47.20%** | **+45.95 pp** | **+36.49 pp** |
| **`with_bag`** | Bag / Accessory Occlusion | 13.84% | 5.31% | **60.69%** | **37.11%** | **+46.85 pp** | **+31.80 pp** |
| **`without_bag`** | Clean Body Silhouette | 45.81% | 26.55% | **90.32%** | **77.54%** | **+44.51 pp** | **+50.99 pp** |
| **`both_large`** | Full-Scale Surveillance | 25.72% | 7.82% | **76.29%** | **44.26%** | **+50.57 pp** | **+36.44 pp** |
| **`prcc_cc`** | **Cloth-Changing (Cam A → C)** | 4.70% | 4.79% | **7.36%** | **9.19%** | **+2.66 pp** | **+4.40 pp** |
| **`prcc_sc`** | **Same-Clothes (Cam A → B)** | 4.43% | 5.04% | **30.08%** | **24.73%** | **+25.65 pp** | **+19.69 pp** |
| **`prcc_all`** | **PRCC Combined (Cam A → B+C)** | 4.70% | 3.79% | **12.77%** | **10.13%** | **+8.07 pp** | **+6.34 pp** |

### 3. Track B FAST Diagnostic Mode (`evaluate.py --fast`)

Reproduced with deterministic stratified sampling (`seed=42`, `max_query=64`, `max_gallery=224`):

| Dataset Subset | Baseline Rank-1 | Baseline mAP | CUDA-Optimized Rank-1 | CUDA-Optimized mAP | Absolute Gain (Rank-1) | Absolute Gain (mAP) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`both_small`** | 18.18% | 22.24% | **70.91%** | **75.55%** | **+52.73 pp** | **+53.31 pp** |
| **`with_bag`** | 11.48% | 10.25% | **52.46%** | **52.51%** | **+40.98 pp** | **+42.26 pp** |
| **`without_bag`** | 29.82% | 30.71% | **87.72%** | **83.20%** | **+57.90 pp** | **+52.49 pp** |
| **`both_large`** | 22.41% | 26.87% | **68.97%** | **73.00%** | **+46.56 pp** | **+46.13 pp** |
| **`prcc_cc`** | 3.12% | 5.84% | **6.25%** | **13.40%** | **+3.13 pp** | **+7.56 pp** |
| **`prcc_sc`** | 4.69% | 7.96% | **18.75%** | **24.36%** | **+14.06 pp** | **+16.40 pp** |
| **`prcc_all`** | 3.12% | 5.67% | **7.81%** | **12.89%** | **+4.69 pp** | **+7.22 pp** |

---

## ⚡ Computational Efficiency & Hardware Profiling

Benchmarked on **NVIDIA GeForce RTX 4060 Laptop GPU** (8 GB VRAM) and host CPU:

| Metric | Track B: Lightweight DCR-ReID | Track A: Paper ResNet-50 | Efficiency Advantage |
|:---|:---:|:---:|:---|
| **Total Parameters** | **9,752,277 (9.75 M)** | ~31.6 M | **69.1% Parameter Reduction** |
| **Theoretical FLOPs** | **3.0015 GFLOPs** (at 256×128) | ~8.50 GFLOPs (at 384×192) | **64.7% FLOP Reduction** |
| **Theoretical MACs** | **1.5007 GMACs** | ~4.25 GMACs | **64.7% MAC Reduction** |
| **GPU Latency (BS=1)** | **8.01 ms** (Median) | ~18.2 ms | **56.0% Lower Latency** |
| **GPU Batched Throughput** | **2,785.7 FPS** (BS=32) | ~1,200 FPS (BS=32) | **2.32× Higher FPS** |
| **CPU Latency (BS=1)** | **31.44 ms** (31.8 FPS) | ~95.0 ms | **66.0% Lower Latency** |
| **Embedding Dimension** | **512-D** | **4096-D** | **87.5% Smaller Vector** |
| **Gallery Index Size (100k IDs)** | **0.20 GB** | **1.64 GB** | **87.8% Lower RAM Usage** |

---

## 📁 Repository Structure

```
Lightweight_Person_Re-Identification_Model/
├── checkpoints/
│   ├── best_model.pth                      # Original baseline checkpoint (preserved strictly immutable)
│   ├── paper_dcr_best.pth                  # Track A: Paper-Faithful best checkpoint
│   ├── paper_dcr_latest.pth                # Track A: Paper-Faithful latest checkpoint
│   └── experiments/
│       └── dcr_reid_cuda_v1/
│           └── best_model.pth              # Track B: Selected CUDA-optimized checkpoint
├── data/
│   └── dataset_loader.py                   # Data loader, PK sampler, transforms (adaptable size), PRCC mapping
├── models/
│   ├── paper_dcr_reid.py                   # Track A: Paper-Faithful DCR-ReID (ResNet-50, CRD 2560-D, DAD)
│   ├── paper_loss.py                       # Track A: Exact Two-Stage Loss (L_ID, L_c, L_R, L_ca, L_ac, L'_ID)
│   ├── lightweight_reid.py                 # Track B: Lightweight DCR-ReID (9.75M params, dual BNNeck)
│   └── loss.py                             # Track B: Combined multi-task loss
├── utils/
│   ├── metrics.py                          # Market-1501 junk exclusion, CMC Rank-k, mAP computation
│   ├── visualization.py                    # 4-panel visualizer (Body, Clothes, Contour, DAD)
│   └── reid_engine.py                      # Re-ID feature extraction engine
├── tests/
│   ├── test_paper_dcr.py                   # Track A: 12 automated unit tests (Architecture, Loss, Decoders, CUDA)
│   └── test_dcr_reid.py                    # Track B: 22 automated unit and regression tests
├── reports/
│   ├── baseline/
│   │   ├── baseline_metrics.json           # Baseline metrics snapshot
│   │   └── baseline_summary.md             # Baseline reproduction verification
│   ├── paper_compliance_audit.md           # Granular IEEE TCSVT 2023 paper compliance audit
│   ├── paper_reproduction_results.md       # Track A benchmark reproduction report
│   ├── lightweight_comparison.md           # Side-by-side Track A vs Track B architectural comparison
│   ├── evaluation_protocol.md              # Formal PRCC and Market-1501 evaluation protocol definitions
│   ├── reproducibility.md                  # Complete reproducibility runbook, commands, and hashes
│   ├── environment_audit.md                # CUDA environment & 8 acceptance tests
│   ├── evaluation_protocol_audit.md        # Dataset split integrity & junk filtering audit
│   ├── model_implementation_audit.md       # Architecture & loss function audit
│   ├── clothing_invariance_analysis.md     # PRCC clothing invariance & hyperspherical separation analysis
│   ├── efficiency_benchmark.md             # Latency, FLOPs, MACs, throughput, and VRAM audit
│   ├── final_metrics.json                  # Machine-readable final benchmark metrics
│   ├── experiment_comparison.csv           # Tabular comparison across all 7 subsets
│   └── test_results.md                     # Regression test execution report
├── requirements_cuda.txt                   # Reproducible CUDA dependencies (PyTorch 2.5.1+cu124)
├── evaluate.py                             # Dual-track evaluation benchmark script (FAST / FULL / Auto-detect)
├── train_paper_dcr.py                      # Track A: Paper-Faithful training script (Two-stage optimization)
├── train_cuda.py                           # Track B: CUDA training script (PK sampling, AMP, 2-stage loss)
├── verify_cuda.py                          # CUDA environment diagnostic tool
└── README.md                               # Project documentation
```

---

## 🚀 Environment Setup & Reproduction Guide

### 1. Environment Installation
For hardware GPU acceleration with PyTorch 2.5.1 and CUDA 12.4:
```powershell
# Create dedicated CUDA virtual environment
python -m venv .venv_cuda
.\.venv_cuda\Scripts\activate

# Install CUDA-enabled PyTorch and dependencies
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install numpy pillow
```

### 2. Verify CUDA Acceptance
Run the diagnostic script to verify GPU detection, tensor math, model forward/backward pass, and AMP:
```powershell
.\.venv_cuda\Scripts\python.exe verify_cuda.py
```

### 3. Run Complete Automated Unit Test Suites (34 Tests)
Run all 34 automated unit and regression tests across both tracks:
```powershell
.\.venv_cuda\Scripts\python.exe -m unittest tests/test_dcr_reid.py tests/test_paper_dcr.py
```

### 4. FAST Diagnostic Evaluation (Baseline Reproduction)
Reproduce the baseline checkpoint evaluation on CPU or CUDA in under 3 seconds:
```powershell
.\.venv_cuda\Scripts\python.exe evaluate.py --fast --device auto
```

### 5. FULL Parity Benchmark Across All 7 Subsets
Run full exhaustive evaluation on CUDA:
```powershell
.\.venv_cuda\Scripts\python.exe evaluate.py --full --device cuda
```

### 6. Train Track A (Paper-Faithful DCR-ReID)
Train with ResNet-50 backbone, 384×192 input, CRD module, and two-stage loss optimization on PRCC:
```powershell
.\.venv_cuda\Scripts\python.exe train_paper_dcr.py --epochs 60 --stage1-epochs 10 --batch-size 32 --device cuda
```

### 7. Train Track B (Lightweight DCR-ReID Edge Variant)
Train lightweight CNN with identity-balanced PK sampling and dual BNNeck:
```powershell
.\.venv_cuda\Scripts\python.exe train_cuda.py --dataset_type both_small --epochs 15 --batch_size 32 --use_pk_sampler --use_amp
```

---

## 🔐 Checkpoint Integrity & Security
All model checkpoints are secured with SHA-256 cryptographic hashes:
- Baseline Checkpoint (`checkpoints/best_model.pth`): `e5bfb5aa627fc3a10f27c01d03bdb54ba8cda06cb3453d24484cd8142e034730`
- The baseline checkpoint is preserved strictly immutable and is never overwritten during any training or evaluation workflows.
