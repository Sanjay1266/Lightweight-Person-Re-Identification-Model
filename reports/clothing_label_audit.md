# Clothing Label & Supervision Audit Report: PRCC Protocol

### Course: 23CSE373 - Computer Vision | Date: 2026-10-27
**Project**: Lightweight Person Re-Identification Model (DCR-ReID)  
**Root Repository**: `S:\5th Sem\CV\Lightweight_Person_Re-Identification_Model`

---

## 1. Executive Summary

This report documents the discovery, theoretical proof, and verified correction of the clothing-label generation bug in the DCR-ReID training pipeline.

The previous formula:
```python
clothes_labels = (pids * 2 + (cams % 2)).long() % num_clothes
```
inverted the clothing supervision by mapping Camera A and Camera C to the same clothing label, while isolating Camera B as different clothing. This directly inverted the multi-task supervision for clothing-invariant disentanglement.

---

## 2. Theoretical Analysis & Mathematical Proof

### 1. Ground Truth Protocol of the PRCC Benchmark
In the PRCC benchmark (Yang et al., IEEE TPAMI 2021; Cui et al., IEEE TCSVT 2023):
- **Camera A (`camid = 0`)**: Person photographed in Outfit 1.
- **Camera B (`camid = 1`)**: Same person photographed in **Outfit 1 (Same-Clothes condition)**.
- **Camera C (`camid = 2`)**: Same person photographed in **Outfit 2 (Cloth-Changing condition)**.

### 2. The Parity Modulo-2 Defect
Evaluating the expression across cameras:
$$\text{Cam A } (c=0): 0 \pmod 2 = 0 \implies \text{label} = (\text{pid} \times 2 + 0) \pmod{\text{num\_clothes}}$$
$$\text{Cam B } (c=1): 1 \pmod 2 = 1 \implies \text{label} = (\text{pid} \times 2 + 1) \pmod{\text{num\_clothes}}$$
$$\text{Cam C } (c=2): 2 \pmod 2 = 0 \implies \text{label} = (\text{pid} \times 2 + 0) \pmod{\text{num\_clothes}}$$

### 3. Destruction of Multi-Task Learning
1. **Adversarial Inversion**: The clothes adversarial loss ($\mathcal{L}_{ca}$) penalizes the network when clothes features correlate with identity. Because Cam A and Cam C were assigned the same clothing label, the model was actively penalized for recognizing that clothing changed in Cam C.
2. **Classification Corruption**: In Cam B (same clothes), the model was penalized for assigning the same clothing label as Cam A, forcing it to artificially fabricate a clothing distinction between identical outfits.

---

## 3. Verified Implementation in Code

The corrected function `generate_clothing_labels` was implemented in `data/dataset_loader.py` and integrated into `train.py`:

```python
def generate_clothing_labels(pids, cams, is_prcc=True, num_clothes=100):
    if is_prcc:
        # Strict validation: reject any invalid camera IDs
        invalid_mask = (cams < 0) | (cams > 2)
        if invalid_mask.any():
            invalid_cams = cams[invalid_mask].tolist()
            raise ValueError(f"Invalid PRCC camera ID detected: {invalid_cams}. PRCC only supports cameras 0 (A), 1 (B), and 2 (C).")
        
        # Explicit mapping: condition 0 for Cams A & B; condition 1 for Cam C
        clothes_condition = (cams == 2).long()
        clothes_labels = (pids * 2 + clothes_condition) % num_clothes
    else:
        # Dataset-specific mapping for Market-1501
        clothes_labels = (pids * 2 + (cams % 2)).long() % num_clothes
        
    return clothes_labels
```

### Verified Properties:
1. **Camera A ($c=0$)**: $\text{condition} = 0 \implies \text{label} = \text{pid} \times 2 + 0$.
2. **Camera B ($c=1$)**: $\text{condition} = 0 \implies \text{label} = \text{pid} \times 2 + 0$ (**Identical to Cam A**).
3. **Camera C ($c=2$)**: $\text{condition} = 1 \implies \text{label} = \text{pid} \times 2 + 1$ (**Distinct Cloth-Changing**).
4. **Range Bound**: All labels strictly lie within $[0, \text{num\_clothes} - 1]$, perfectly matching the output dimension of `clothes_classifier.weight` ($[100, 512]$).
5. **Safety**: Invalid camera IDs (e.g. $c \notin \{0, 1, 2\}$) raise an immediate `ValueError`.

---

## 4. Empirical Validation on Live Dataset Batches

Live inspection on training batches loaded from `Data_set/prcc/rgb/train` yielded:
- **Identity 42**:
  - Cam 0 $\rightarrow$ Label `84`
  - Cam 0 $\rightarrow$ Label `84`
  - Cam 2 $\rightarrow$ Label `85`
  - Cam 2 $\rightarrow$ Label `85`
- **Identity 103**:
  - Cam 1 $\rightarrow$ Label `6`
  - Cam 0 $\rightarrow$ Label `6` (Matches Cam 1!)
  - Cam 2 $\rightarrow$ Label `7` (Distinguishes Cam 2!)

Unit test `test_17_prcc_clothing_label_mapping` in `tests/test_dcr_reid.py` passes deterministically.
