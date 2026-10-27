# Lightweight DCR-ReID: Checkpoint Integrity and Tensor Audit Report

### Course: 23CSE373 - Computer Vision | Date: 2026-10-27

---

## 1. Executive Summary

A deep audit was conducted on the existing model checkpoints (`best_model.pth` and `final_model.pth`). The audit resolves the root cause of the previously reported `161/162 tensors matched` message, validates checkpoint SHA-256 integrity, and confirms forward-pass numerical stability.

### Checkpoint: `checkpoints/best_model.pth`

- **SHA-256 Hash**: `e5bfb5aa627fc3a10f27c01d03bdb54ba8cda06cb3453d24484cd8142e034730`
- **File Size**: 37.3 MB
- **Total Checkpoint Tensors**: 162
- **Detected Training Classes (PIDs)**: 150 (PRCC Training Split)
- **Detected Clothes Categories**: 100
- **Exact Match Status**: **162 / 162** tensors matched (100% full weight compatibility when `num_clothes=100`).

#### Tensor Breakdown by Component:
- **Backbone**: 78 parameter/buffer tensors
- **Crd**: 63 parameter/buffer tensors
- **Dad**: 9 parameter/buffer tensors
- **Bnneck**: 10 parameter/buffer tensors
- **Heads**: 2 parameter/buffer tensors

#### Numerical & Forward-Pass Verification:
- **Input Shape**: `[2, 3, 256, 128]` (Batch=2, Channels=3, Height=256, Width=128)
- **Output Shape (Clothes Invariant)**: `[2, 512]` (512-dim L2 normalized)
- **NaN / Inf Detection**: None (ci_has_nan=False, ci_has_inf=False)
- **L2 Normalization**: Strictly Verified (Unit sphere Euclidean norm $\|f\|_2 = 1.0 \pm 10^{-5}$)

### Checkpoint: `checkpoints/final_model.pth`

- **SHA-256 Hash**: `fa5566bfe947f7bc3fd4771a46294b259556c05a94b28a11e2efb7381497c0f8`
- **File Size**: 37.31 MB
- **Total Checkpoint Tensors**: 162
- **Detected Training Classes (PIDs)**: 150 (PRCC Training Split)
- **Detected Clothes Categories**: 100
- **Exact Match Status**: **162 / 162** tensors matched (100% full weight compatibility when `num_clothes=100`).

#### Tensor Breakdown by Component:
- **Backbone**: 78 parameter/buffer tensors
- **Crd**: 63 parameter/buffer tensors
- **Dad**: 9 parameter/buffer tensors
- **Bnneck**: 10 parameter/buffer tensors
- **Heads**: 2 parameter/buffer tensors

#### Numerical & Forward-Pass Verification:
- **Input Shape**: `[2, 3, 256, 128]` (Batch=2, Channels=3, Height=256, Width=128)
- **Output Shape (Clothes Invariant)**: `[2, 512]` (512-dim L2 normalized)
- **NaN / Inf Detection**: None (ci_has_nan=False, ci_has_inf=False)
- **L2 Normalization**: Strictly Verified (Unit sphere Euclidean norm $\|f\|_2 = 1.0 \pm 10^{-5}$)

### Checkpoint: `checkpoints/experiments/dcr_reid_corrected_pk_v1.pth`

- **SHA-256 Hash**: `bcfd487f566d8478e338febf7a34b9fd615050e53c43e02400e163f6a3c8bf0c`
- **File Size**: 37.31 MB
- **Total Checkpoint Tensors**: 162
- **Detected Training Classes (PIDs)**: 150 (PRCC Training Split)
- **Detected Clothes Categories**: 100
- **Exact Match Status**: **162 / 162** tensors matched (100% full weight compatibility when `num_clothes=100`).
- **Training Pipeline**: Fine-tuned using validated PRCC camera-to-clothing condition mapping and identity-balanced `RandomIdentitySampler` ($P=4, K=4$).
- **Loss Progression**: Total loss converged from 8.7286 to 8.5108; Triplet loss reduced from 1.3749 to 0.7377; Reconstruction loss reduced from 0.1969 to 0.1460.

#### Tensor Breakdown by Component:
- **Backbone**: 78 parameter/buffer tensors
- **Crd**: 63 parameter/buffer tensors
- **Dad**: 9 parameter/buffer tensors
- **Bnneck**: 10 parameter/buffer tensors
- **Heads**: 2 parameter/buffer tensors

#### Numerical & Forward-Pass Verification:
- **Input Shape**: `[2, 3, 256, 128]` (Batch=2, Channels=3, Height=256, Width=128)
- **Output Shape (Clothes Invariant)**: `[2, 512]` (512-dim L2 normalized)
- **NaN / Inf Detection**: None (ci_has_nan=False, ci_has_inf=False)
- **L2 Normalization**: Strictly Verified (Unit sphere Euclidean norm $\|f\|_2 = 1.0 \pm 10^{-5}$)

---

## 2. Root Cause of the 161/162 Tensor Match Discrepancy

In the original evaluation script (`evaluate.py`), the model was instantiated with:
```python
model = LightweightDCRReID(num_classes=num_classes, feat_dim=512, inference_mode='clothes_invariant')
```
1. `LightweightDCRReID.__init__` has a default parameter `num_clothes=50`, creating `clothes_classifier.weight` of shape `[50, 512]`.
2. In the checkpoint header, `clothes_classifier.weight` has shape `[100, 512]` (trained with `num_clothes = max(20, min(100, num_classes * 2)) = 100` for 150 PRCC training identities).
3. Because `[100, 512] != [50, 512]`, the script skipped this single tensor, loading 161/162 tensors.
4. **Impact on Retrieval**: The `clothes_classifier` head is only used during training for clothes loss supervision; it is NOT invoked during evaluation/inference. Thus, the mismatch was functionally harmless to feature embeddings. However, by reading `num_clothes` directly from the checkpoint header (`sd['clothes_classifier.weight'].shape[0]`), the model loads **162/162 (100%)** tensors without skipping.
