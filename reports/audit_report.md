# Lightweight DCR-ReID: Comprehensive Repository and Architectural Audit Report

### Course: 23CSE373 - Computer Vision | Date: 2026-10-27

---

## 1. Executive Summary

A comprehensive repository audit was conducted on the Lightweight Person Re-Identification Model (DCR-ReID) codebase. All scripts, models, loss definitions, data loaders, checkpoints, and metrics were scrutinized line-by-line.

The audit identifies the exact technical root causes of:
1. The **FAST vs FULL evaluation discrepancy** (e.g. `prcc_sc` Rank-1 dropping from 59.38% to 4.40%).
2. The **161/162 checkpoint tensor match** warning.
3. The **silent exception fallback in FLOP counting**.
4. The **PRCC clothing label inversion** in training supervision.
5. The **lack of PK-sampling** hindering triplet metric learning.
6. The **untrained `bottleneck_irrel` branch** in dual BNNeck.

---

## 2. Confirmed Bugs & Technical Evidence

### Bug 1: FAST Mode Truncation Bias and Artificial Score Inflation
- **Location**: `evaluate.py`, lines 49–77 & 195–204.
- **Evidence**:
  ```python
  if fast:
      max_q, max_g = 60, 200
  ...
  q_count = 0
  for imgs, pids, cams, _ in query_loader:
      ...
      q_count += len(pids)
      if max_query and q_count >= max_query: break
  for imgs, pids, cams, _ in gallery_loader:
      ...
      g_count += len(pids)
      if max_gallery and g_count >= max_gallery: break
  ```
- **Root Cause**:
  1. `query_loader` and `gallery_loader` are instantiated with `shuffle=False`.
  2. Sequential iteration takes only the first 2 batches (64 images) of query images and first 7 batches (224 images) of gallery images.
  3. In PRCC, queries are grouped by identity: the first 64 queries belong strictly to PID 1 and PID 2. The first 224 gallery images belong strictly to PIDs 1, 2, 4, 5.
  4. In `utils/metrics.py`, any query without positive matches in the gallery is skipped (`if not np.any(orig_cmc): continue`).
  5. The retrieval task in FAST mode is reduced from a 71-identity search across 3,873 gallery images to a **4-identity search across 224 gallery images**. Distractors from 67 other identities are omitted.
  6. In FULL mode, all 3,384 queries are evaluated against all 3,873 gallery images across all 71 identities, revealing the true model accuracy (~4.40%).

### Bug 2: Duplicate Metric Duplication between `prcc_sc` and `prcc_all` in FAST Mode
- **Location**: `data/dataset_loader.py` lines 107–120, `evaluate.py` lines 67–77.
- **Evidence**:
  In `_load_prcc()`:
  - `prcc_sc` loads gallery images from Camera B (3,873 images).
  - `prcc_all` loads gallery images from Camera B, followed by Camera C.
- **Root Cause**: Because FAST mode terminates at `max_gallery = 200`, `prcc_all` never reaches Camera C images. Both `prcc_sc` and `prcc_all` evaluate against the exact same 224 images of Camera B. Consequently, their Rank-1 (59.38%) and mAP (71.67%) were identical bit-for-bit.

### Bug 3: Checkpoint Mismatch (161/162 Tensors)
- **Location**: `evaluate.py` line 173, `models/lightweight_reid.py` line 293.
- **Evidence**:
  `LightweightDCRReID` constructor defaults to `num_clothes=50` (`clothes_classifier.weight` shape: `[50, 512]`).
  In `checkpoints/best_model.pth`, `clothes_classifier.weight` has shape `[100, 512]`.
- **Root Cause**: `evaluate.py` detected `num_classes=150` from the checkpoint header, but did not detect `num_clothes=100`. The shape mismatch caused the clothing classifier tensor to be omitted during `load_state_dict`.
- **Impact**: Harmless during inference (since `clothes_classifier` is only evaluated during training), but causes unnecessary warning. Passing `num_clothes=100` achieves 162/162 (100%) tensor loading.

### Bug 4: FLOP Counter Mode Exception and Fallback
- **Location**: `evaluate.py`, lines 123–131.
- **Evidence**:
  ```python
  dummy_input = torch.randn(1, 3, 256, 128).to(device)
  with FlopCounterMode(display=False) as fcm:
      _ = model(dummy_input)
  ```
- **Root Cause**: `dummy_input` with `batch_size = 1` causes PyTorch's `FlopCounterMode` tracing to fail with `ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512])` inside `BatchNorm1d`. The exception was caught silently and fell back to hardcoded `flops_g = 3.001`.
- **Correction**: Providing `batch_size = 2` (`torch.randn(2, 3, 256, 128)`) and dividing total FLOPs by 2 allows `FlopCounterMode` to execute cleanly, dynamically returning `3.0015 GFLOPs` and `1.5007 GMACs`.

### Bug 5: Inverted PRCC Clothing Supervision in `train.py`
- **Location**: `train.py`, line 155.
- **Evidence**:
  `clothes_labels = (pids * 2 + (cams % 2)).long() % num_clothes`
- **Root Cause**:
  In PRCC:
  - Cam A (`camid = 0`): Clothes 1
  - Cam B (`camid = 1`): Clothes 1 (Same Clothes as Cam A)
  - Cam C (`camid = 2`): Clothes 2 (Cloth-Changing)
  Using `cams % 2`:
  - Cam A (`0 % 2 = 0`) and Cam C (`2 % 2 = 0`) received identical clothes label $0$.
  - Cam B (`1 % 2 = 1`) received clothes label $1$.
  The model was penalized for recognizing that Cam A and Cam B had identical clothing, and rewarded for treating Cam A and Cam C as identical clothing!
- **Correction**:
  `clothes_labels = (pids * 2 + (1 if cams == 2 else 0)).long() % num_clothes`.

### Bug 6: Ineffective Triplet Mining without Identity-Balanced (PK) Sampler
- **Location**: `train.py` line 108, `data/dataset_loader.py` line 228.
- **Evidence**: `train_loader` uses `shuffle=True` random batching.
- **Root Cause**: Across 150 training identities with batch size 16, randomly sampled batches rarely contain multiple instances of the same person identity. Positive pairs are missing from the distance matrix (`dist_ap = 0`), disabling the positive pull of `loss_triplet`.
- **Correction**: Implement `RandomIdentitySampler` providing $P$ identities with $K$ instances each ($P \times K = 16$).

### Bug 7: Untrained `bottleneck_irrel` in Dual BNNeck
- **Location**: `models/lightweight_reid.py` lines 386–404, `models/loss.py`.
- **Evidence**: `feat_irrel_normed` was computed in `forward()` but omitted from the training dictionary. `bottleneck_irrel.weight` retained its initialization value of `1.0` with zero gradient updates.

---

## 3. Summary of Files Audited & Status

| File | Audit Findings | Planned / Applied Action |
| :--- | :--- | :--- |
| `evaluate.py` | FAST truncation bug, FLOP batch size exception, header class/clothes detection | Fixed with representative sampling, dynamic FLOP measurement, 162/162 loading |
| `data/dataset_loader.py` | Valid Market/PRCC formats, missing PK sampler | Added `RandomIdentitySampler` for $P \times K$ batching |
| `models/lightweight_reid.py` | Solid DCR-ReID backbone, dual BNNeck gradient flow | Preserved architecture; verified exact FLOPs & inference branches |
| `models/loss.py` | Loss functions align with Eq. 5–18; backward compatibility intact | Verified two-stage losses (CE Label Smooth, Triplet, Recon L1, Clothes CE/Adv) |
| `train.py` | Inverted clothes label formula, un-batched triplet pairs | Corrected PRCC clothing labels, integrated PK sampler, verified gradient updates |
| `utils/metrics.py` | Standard Market-1501 evaluation mathematically correct | Validated CMC Rank-k and mAP; added unit tests |
