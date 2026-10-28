# Lightweight Person Re-Identification Model (DCR-ReID)
### 23CSE373 - Computer Vision | Amrita School of Engineering

A high-performance, lightweight Computer Vision system implementing **Deep Component Reconstruction for Cloth-Changing and Accessory Person Re-Identification (DCR-ReID)** based on the IEEE TCSVT 2023 research paper. Designed for edge surveillance deployment, the framework disentangles identity-invariant biometric representations from transient clothes and accessory variations (clothing changes, handbags, backpacks).

---

## 👥 Project Information & Authors
- **Course**: 23CSE373 - Computer Vision
- **Base Paper**: *DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification* (IEEE Transactions on Circuits and Systems for Video Technology, 2023)
- **Team Members**:
  - **G N Bhuvaneshwaran** (`CB.SC.U4CSE24218`)
  - **Sanjay MS** (`CB.SC.U4CSE24248`)
  - **Sanjay S** (`CB.SC.U4CSE24249`)

---

## 🌟 Full DCR-ReID Architecture & Key Modules

The framework implements three coordinated branches formulated in Section III of the base paper:

```
                            Input Image (256×128)
                                     ↓
                    Lightweight Residual Backbone (P_i)
                                     ↓
        ┌────────────────────────────┼────────────────────────────┐
        ↓                            ↓                            ↓
[PI Branch]                   [CR Branch]                  [CI Branch]
Person Identification        Component Reconstruction     Clothes Identification
  ├── Global Avg Pooling       ├── Channel Decomp (Eq. 8)   ├── Clothes Classifier C_P
  ├── Dual BNNeck & Metric         P_i = P⁺ ⊕ P⁻ ⊕ Pᵗ       ├── Adversarial Loss L_ca
  └── Invariant Inference:     ├── 4-Block Decoders ψ^v     ├── DAD Feature Assembly:
      Discards F⁻ (clothes)    └── L1 Recon Loss L_R (Eq.10)    G_i = F⁺ ⊕ F_ai⁻ ⊕ Fᵗ
                                                            └── Dual Attention (CA + SA)
```

### 1. Person Identification (PI) Branch
- Deep spatial feature extraction $P_i \in \mathbb{R}^{B \times 512 \times 16 \times 8}$ via a lightweight residual backbone (~4.90M parameters) using stride-1 Stage 4 to preserve spatial resolution for component decoders.
- Metric learning with **Label-Smoothed Cross-Entropy Loss** ($\epsilon=0.1$, Eq. 5-6) and **Batch-Hard Triplet Ranking Loss** ($\text{margin}=0.3$).
- **Cloth-Changing Invariant Inference**: Discards transient clothes-relevant features $F_i^-$ during inference, relying strictly on body shape and contour cues ($F_i^+ \oplus F_i^t$) to prevent clothing bias.

### 2. Component Reconstruction (CR) Branch
- **Channel Decomposition** (Eq. 8):
  $$P_i = P_i^- \oplus P_i^+ \oplus P_i^t$$
  - $P_i^-$: Clothes-relevant representations (shirt, pants, apparel color, bag accessories)
  - $P_i^+$: Clothes-irrelevant representations (body shape, facial landmarks, anatomical proportions)
  - $P_i^t$: Human silhouette contour/boundary representations
- **4-Block Component Decoders** ($\psi^-, \psi^+, \psi^t$, Fig. 4):
  Normalizes channels via projection and decodes visual component masks $Y_i^-, Y_i^+, Y_i^t$ through 4 residual reconstruction blocks.
- **Component Reconstruction Loss** (Eq. 10):
  $$\mathcal{L}_R = \frac{1}{N} \sum_{i=1}^N \sum_{v \in \{-, +, t\}} |T_i^v - Y_i^v|_1$$

### 3. Clothes Identification (CI) Branch & DAD Module
- **Clothes Classifier** $C_P(\cdot | c)$ with fine-grained clothes classification loss $\mathcal{L}_c$ (Eq. 11-12).
- **Clothes Adversarial Loss** $\mathcal{L}_{ca}$ (Eq. 13-14) to penalize correlation between clothing and identity.
- **Deep Assembled Disentanglement (DAD) Module** (Eq. 15-18):
  Assembles feature vector $G_i = F_i^+ \oplus F_{ai}^- \oplus F_i^t$ where $F_{ai}^-$ is randomly permuted across identities in the batch, forcing the model to classify identity using clothing-invariant features.
- **Two-Stage Optimization Protocol**:
  - Stage 1: $\mathcal{L} = \mathcal{L}_{ID} + \mathcal{L}_{\text{triplet}} + 0.5 \mathcal{L}_c + 0.2 \mathcal{L}_R$
  - Stage 2: $\mathcal{L} = \mathcal{L}_{ID} + \mathcal{L}_{\text{triplet}} + 0.5 \mathcal{L}_c + 0.1 \mathcal{L}_{ca} + 0.2 \mathcal{L}_R$

---

## 📊 Empirical Benchmark Evaluation: Baseline vs. CUDA-Optimized

Both checkpoints were benchmarked under identical test manifests, 512-dimensional $L_2$-normalized embeddings, cosine distance calculation, and Market-1501 junk exclusion rules.

### 1. FULL Exhaustive Evaluation (Complete Valid Test Splits)

| Dataset Subset | Surveillance Condition | Query Count | Valid Queries | Gallery Count | Baseline Rank-1 | Baseline mAP | CUDA-Optimized Rank-1 | CUDA-Optimized mAP | Absolute Gain (Rank-1) | Absolute Gain (mAP) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`both_small`** | Combined Small Subset | 475 | 457 | 2,146 | 22.54% | 10.71% | **68.49%** | **47.20%** | **+45.95 pp** | **+36.49 pp** |
| **`with_bag`** | Bag / Accessory Occlusion | 322 | 318 | 1,131 | 13.84% | 5.31% | **60.69%** | **37.11%** | **+46.85 pp** | **+31.80 pp** |
| **`without_bag`** | Clean Body Silhouette | 179 | 155 | 986 | 45.81% | 26.55% | **90.32%** | **77.54%** | **+44.51 pp** | **+50.99 pp** |
| **`both_large`** | Full-Scale Surveillance | 1,265 | 1,244 | 10,048 | 25.72% | 7.82% | **76.29%** | **44.26%** | **+50.57 pp** | **+36.44 pp** |
| **`prcc_cc`** | **Cloth-Changing (Cam A → C)** | 3,384 | 3,384 | 3,543 | 4.70% | 4.79% | **7.36%** | **9.19%** | **+2.66 pp** | **+4.40 pp** |
| **`prcc_sc`** | **Same-Clothes (Cam A → B)** | 3,384 | 3,384 | 3,873 | 4.43% | 5.04% | **30.08%** | **24.73%** | **+25.65 pp** | **+19.69 pp** |
| **`prcc_all`** | **PRCC Combined (Cam A → B+C)** | 3,384 | 3,384 | 7,416 | 4.70% | 3.79% | **12.77%** | **10.13%** | **+8.07 pp** | **+6.34 pp** |

*All 7 subsets demonstrate verified positive gains in both Rank-1 and mAP.*

### 2. FAST Diagnostic Mode Evaluation (`evaluate.py --fast`)

Evaluated with deterministic stratified representative sampling (`seed=42`, `max_query=64`, `max_gallery=224`):

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

## ⚡ Computational Efficiency & Lightweight Edge Profiling

Measured on the **NVIDIA GeForce RTX 4060 Laptop GPU** (8 GB VRAM) and host CPU:

| Metric | Measurement / Value | ResNet-50 Reference | Efficiency Assessment |
|:---|:---:|:---:|:---|
| **Total Parameters** | **9,752,277 (9.75 M)** | 25.56 M | **61.8% Parameter Reduction** |
| **Backbone Parameters** | **4.90 M** | 25.56 M | **80.9% Backbone Reduction** |
| **Theoretical FLOPs** | **3.0015 GFLOPs** (at 256×128) | 8.20 GFLOPs | **63.4% FLOP Reduction** |
| **Theoretical MACs** | **1.5007 GMACs** ($1 \text{ MAC} = 2 \text{ FLOPs}$) | 4.10 GMACs | **63.4% MAC Reduction** |
| **GPU Latency (Batch-1, Synchronized)** | **11.07 ms** (Median) | ~35 ms | **3.16× Faster** |
| **GPU Batched Throughput (BS=32)** | **2,694.9 FPS** (0.371 ms / img) | ~400 FPS | **Real-Time Edge Capable** |
| **Peak GPU VRAM (Training)** | **488.9 MB** | ~2.5 GB | **< 6.1% VRAM Utilized** |
| **CPU Latency (Batch-1)** | **31.44 ms** (31.8 FPS) | ~78 ms | **2.48× Faster** |
| **Model Checkpoint Size** | **37.5 MB** | 98.0 MB | **61.7% Smaller Footprint** |

---

## 📁 Repository Structure

```
Lightweight_Person_Re-Identification_Model/
├── checkpoints/
│   ├── best_model.pth                      # Original baseline checkpoint (preserved strictly unchanged)
│   └── experiments/
│       └── dcr_reid_cuda_v1/
│           ├── best_model.pth              # Selected CUDA-optimized checkpoint
│           └── final_model.pth             # Final epoch checkpoint
├── data/
│   └── dataset_loader.py                   # Data loader, PK sampler, transforms, PRCC label mapping
├── models/
│   ├── lightweight_reid.py                 # Lightweight DCR-ReID (Stem, Stages 1-4, CRD, DAD, BNNeck)
│   └── loss.py                             # Label-smoothed CE, Batch-Hard Triplet, L1 Recon, Clothes Loss
├── utils/
│   ├── metrics.py                          # Market-1501 junk exclusion, CMC Rank-k, mAP computation
│   ├── visualization.py                    # 4-panel visualizer (Body, Clothes, Contour, DAD)
│   └── reid_engine.py                      # Re-ID feature extraction engine
├── tests/
│   └── test_dcr_reid.py                    # 22 automated regression & unit tests
├── reports/
│   ├── baseline/
│   │   ├── baseline_metrics.json           # Canonical baseline metrics snapshot
│   │   └── baseline_summary.md             # Baseline reproduction verification
│   ├── environment_audit.md                # CUDA environment & 8 acceptance tests
│   ├── evaluation_protocol_audit.md        # Dataset split integrity & junk filtering
│   ├── model_implementation_audit.md       # Architecture & loss function audit
│   ├── clothing_invariance_analysis.md     # PRCC clothing invariance & hyperspherical analysis
│   ├── efficiency_benchmark.md             # Latency, FLOPs, MACs, throughput, VRAM audit
│   ├── final_evaluation.md                 # Full parity benchmark report
│   ├── final_metrics.json                  # Machine-readable final benchmark metrics
│   ├── experiment_comparison.csv           # Tabular comparison across all 7 subsets
│   └── test_results.md                     # Regression test execution report
├── requirements_cuda.txt                   # Reproducible CUDA dependencies (PyTorch 2.5.1+cu124)
├── evaluate.py                             # Evaluation benchmark script (FAST / FULL / device selection)
├── run_full_parity_benchmark.py            # Automated 7-subset parity benchmark script
├── train_cuda.py                           # CUDA training script (PK sampling, AMP, 2-stage loss)
├── verify_cuda.py                          # CUDA environment diagnostic tool
└── README.md                               # Project documentation
```

---

## 🚀 Environment Setup & Reproduction Guide

### 1. Environment Installation
For hardware GPU acceleration with PyTorch 2.5.1 and CUDA 12.4:
```bash
# Create dedicated CUDA virtual environment
python -m venv .venv_cuda
.\.venv_cuda\Scripts\activate

# Install CUDA-enabled PyTorch and dependencies
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install numpy pillow
```

### 2. Verify CUDA Acceptance
Run the diagnostic script to verify GPU detection, tensor math, model forward/backward pass, and AMP:
```bash
python verify_cuda.py
```

### 3. Run Automated Regression Test Suite
Run all 22 automated unit and regression tests:
```bash
python -m unittest tests/test_dcr_reid.py
```

### 4. FAST Diagnostic Evaluation
Run FAST evaluation with device auto-detection:
```bash
python evaluate.py --fast
```
Or force specific devices:
```bash
python evaluate.py --fast --device cuda
python evaluate.py --fast --device cpu
```

### 5. FULL Parity Benchmark Across All 7 Subsets
Run full exhaustive evaluation comparing baseline and CUDA checkpoints:
```bash
python run_full_parity_benchmark.py
```

### 6. Train the Improved Model with CUDA Acceleration
Train with identity-balanced $P \times K$ sampling ($P=8, K=4$), Automatic Mixed Precision, and two-stage optimization:
```bash
python train_cuda.py --dataset_type both_small --epochs 15 --batch_size 32 --lr 0.00035 --use_pk_sampler --use_amp
```

---

## ⚠️ Known Limitations & Deployment Notes
1. **Bag / Accessory Occlusion**: Bulky backpacks and messenger bags occlude the torso region ($y \in [0.22, 0.85]$). Because DCR-ReID inference suppresses the clothes-relevant branch $F_i^-$ (where bag textures route), biometric signal is concentrated in the head and lower limbs.
2. **End-to-End Surveillance Latency**: The 11.07 ms single-image inference latency profiles neural network feature extraction. Real-world edge surveillance pipelines additionally include upstream RTSP video decoding (~2–4 ms), YOLO person detection (~8–15 ms), and SORT tracking (~1–3 ms).
3. **Cross-Dataset Generalization**: Checkpoints trained on Market-1501-style subsets (`both_small`) exhibit high Rank-1 on surveillance subsets (68.49% on `both_small`, 90.32% on `without_bag`), but cross-clothing generalization on PRCC requires dedicated cross-clothing pairs in training.
