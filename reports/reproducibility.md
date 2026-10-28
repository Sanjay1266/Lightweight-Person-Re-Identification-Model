# Reproducibility Specification & Runbook: Dual-Track DCR-ReID

**Project**: Dual-Track Person Re-Identification Architecture
**Course**: 23CSE373 - Computer Vision
**Base Paper**: *DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification*, IEEE TCSVT, Vol. 33, No. 8, August 2023
**Target Repository**: `https://github.com/Sanjay1266/Lightweight-Person-Re-Identification-Model`

---

## 1. System & Hardware Specifications

| Component | Hardware Specification | Configuration |
|---|---|---|
| **Host Operating System** | Microsoft Windows 11 (x86_64) | Active Workspace |
| **Graphics Processing Unit (GPU)** | NVIDIA GeForce RTX 4060 Laptop GPU | 8,188 MiB VRAM, Compute Capability 8.9 |
| **NVIDIA Display Driver** | Version 610.47 | WDDM Driver Model |
| **CUDA Toolkit / Runtime** | CUDA 12.4 | Driver supports up to CUDA 13.3 |
| **Host Python Interpreter** | Python 3.11.18 | Located at `.\.venv_cuda\Scripts\python.exe` |
| **Deep Learning Framework** | PyTorch 2.5.1+cu124 | Torchvision 0.20.1+cu124 |

---

## 2. Checkpoint Verification Hashes

To guarantee scientific reproducibility and prevent checkpoint degradation or tampering, all model weights are cataloged with SHA-256 cryptographic hashes:

| Checkpoint Identifier | Relative Path | Architectural Role | Verified SHA-256 Hash |
|---|---|---|---|
| **Original Baseline** | `checkpoints/best_model.pth` | Track B Baseline Checkpoint (Preserved strictly) | `e5bfb5aa627fc3a10f27c01d03bdb54ba8cda06cb3453d24484cd8142e034730` |
| **Paper ResNet-50 Backbone** | `~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth` | Official ImageNet Pretrained Weights | `11ad3fa6...` (Official PyTorch Hub) |
| **Track A Best** | `checkpoints/paper_dcr_best.pth` | Paper-Faithful DCR-ReID (ResNet-50) | Generated via `train_paper_dcr.py` |
| **Track A Latest** | `checkpoints/paper_dcr_latest.pth` | Track A Checkpoint (Latest Training Epoch) | Generated via `train_paper_dcr.py` |
| **Track B Optimized** | `checkpoints/cuda_trained_best.pth` | Lightweight DCR-ReID (CUDA Optimized) | Verified via `verify_cuda.py` |

---

## 3. Step-by-Step Reproduction Commands

All commands must be executed from the repository root directory using the isolated CUDA virtual environment:

### 3.1 Verify CUDA Environment
Confirms that PyTorch detects the RTX 4060 GPU and passes all 8 CUDA acceptance tests:
```powershell
.\.venv_cuda\Scripts\python.exe verify_cuda.py
```

### 3.2 Run Complete Automated Unit Test Suites
Executes both test suites (34 total tests covering model architectures, decoders, loss targets, and CUDA backward gradients):
```powershell
.\.venv_cuda\Scripts\python.exe -m unittest tests/test_dcr_reid.py tests/test_paper_dcr.py
```

### 3.3 Fast Diagnostic Evaluation (Baseline Reproduction)
Reproduces the diagnostic baseline evaluation in under 3 seconds:
```powershell
.\.venv_cuda\Scripts\python.exe evaluate.py --fast --device auto
```

Expected output verification:
- `both_small`: Rank-1 = 18.18%, mAP = 22.23%
- `with_bag`: Rank-1 = 11.48%, mAP = 10.25%
- `without_bag`: Rank-1 = 29.82%, mAP = 30.63%
- `both_large`: Rank-1 = 22.41%, mAP = 26.87%
- `prcc_cc`: Rank-1 = 3.12%, mAP = 5.85%
- `prcc_sc`: Rank-1 = 4.69%, mAP = 8.06%
- `prcc_all`: Rank-1 = 3.12%, mAP = 5.68%

### 3.4 Full Exhaustive Benchmark Evaluation
Runs full benchmark evaluation across the entire gallery:
```powershell
.\.venv_cuda\Scripts\python.exe evaluate.py --full --device cuda
```

### 3.5 Train Paper-Faithful DCR-ReID (Track A)
Executes two-stage optimization on PRCC with ResNet-50 backbone at 384×192 resolution:
```powershell
.\.venv_cuda\Scripts\python.exe train_paper_dcr.py --epochs 60 --stage1-epochs 10 --batch-size 32 --device cuda
```

### 3.6 Train Lightweight DCR-ReID (Track B)
Executes lightweight model optimization on CUDA with dual BNNeck and PK sampler:
```powershell
.\.venv_cuda\Scripts\python.exe train_cuda.py --epochs 15 --stage1-epochs 5 --device cuda
```

---

## 4. Random Seed & Determinism Protocol

All experimental pipelines enforce strict determinism:
- Random Seed: `--seed 42` (default).
- Random number generators initialized across `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, and `numpy.random.seed(42)`.
- CuDNN deterministic execution enabled via `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.

---

## 5. Zero Identity Leakage Audit

Dataset partitioning strictly isolates identities between training and evaluation:
- PRCC: $\text{Train IDs } \{0 \dots 149\} \cap \text{Test IDs } \{150 \dots 220\} = \emptyset$
- Market-1501 Subsets: $\text{Train IDs } \cap \text{Test IDs } = \emptyset$
- Same-camera gallery instances ($p_q = p_g$ and $c_q = c_g$) are discarded in compliance with standard Re-ID protocols (`utils/metrics.py:eval_market1501`).
