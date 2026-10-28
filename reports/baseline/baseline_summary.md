# Baseline Benchmark Snapshot & Verification Report

- **Date & Timestamp**: 2026-10-28 20:22:05 IST
- **Evaluation Command**: `python evaluate.py --fast`
- **Evaluation Protocol**: FAST Diagnostic Mode (`seed=42`, `max_query=64`, `max_gallery=224`)
- **Environment**:
  - Python: `3.14.3` (`C:\Program Files\PyManager\python.exe`)
  - PyTorch: `2.13.0+cpu` (CPU build)
  - Device Policy: `auto` -> Resolved: `cpu`
- **Checkpoint Verification**:
  - Target: `checkpoints/best_model.pth`
  - Canonical SHA-256: `e5bfb5aa627fc3a10f27c01d03bdb54ba8cda06cb3453d24484cd8142e034730`
  - Loaded Tensors: 162/162 (100% matched)
  - Identity Classes: 150
  - Clothing Classes: 100
- **Model Efficiency**:
  - Total Parameters: 9,752,277 (9.75 M)
  - Residual Backbone: 4.90 M
  - CRD Decoders ($\psi^-, \psi^+, \psi^t$): 4.17 M
  - DAD Dual Attention: 0.56 M
  - Dual BNNeck & Heads: 0.13 M
  - Computation: 3.0015 GFLOPs (1.5008 GMACs at 256×128)
  - CPU Single-Image Latency (BS=1): Median 33.01 ms | Mean 33.73 ms | p95 38.69 ms
  - CPU Sequential Throughput: 30.3 FPS
  - CPU Batched Throughput (BS=32): 43.5 FPS

---

## Empirical Baseline Metrics (FAST Diagnostic Mode)

| Subset | Query Count | Gallery Count | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) | mAP (%) | Convergence Target |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **both_small** | 64 | 224 | **18.18%** | 30.91% | 40.00% | **22.24%** | [==        ] 20.5% |
| **with_bag** | 64 | 224 | **11.48%** | 21.31% | 27.87% | **10.25%** | [=         ] 13.5% |
| **without_bag** | 64 | 224 | **29.82%** | 52.63% | 70.18% | **30.71%** | [===       ] 32.7% |
| **both_large** | 64 | 224 | **22.41%** | 41.38% | 48.28% | **26.87%** | [==        ] 25.5% |
| **prcc_cc** | 64 | 224 | **3.12%** | 9.38% | 14.06% | **5.84%** | [          ] 5.5% |
| **prcc_sc** | 64 | 224 | **4.69%** | 9.38% | 12.50% | **7.96%** | [          ] 5.5% |
| **prcc_all** | 64 | 224 | **3.12%** | 10.94% | 15.62% | **5.67%** | [          ] 4.3% |

---

## Verification & Integrity Assurance
1. **Preservation Guarantee**: `checkpoints/best_model.pth` is strictly preserved and will never be overwritten.
2. **Reproducibility**: The evaluation was executed with `seed=42`. Query and gallery sampling are stratified across identity PIDs.
3. **Parity**: The baseline metrics reproduced herein match the prompt's reference baseline exactly across all seven subsets.
