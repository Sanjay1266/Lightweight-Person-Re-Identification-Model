# Audit Report: PRCC Clothing-Label Mapping and Supervision Inversion

### Course: 23CSE373 - Computer Vision | Date: 2026-10-27

---

## 1. Executive Summary

A critical training bug was uncovered in the clothing-label generation logic used in `train.py` for training the Lightweight DCR-ReID model on the PRCC (Person Re-Identification under Cloth-Changing) dataset.

The expression previously used:
```python
clothes_labels = (pids * 2 + (cams % 2)).long() % num_clothes
```
inverted the clothing conditions by mapping Camera A and Camera C to the same clothing index, while isolating Camera B as a different clothing condition.

---

## 2. Root Cause & Mathematical Proof

### PRCC Dataset Camera Protocol
Under the documented PRCC dataset protocol (Yang et al., IEEE TPAMI 2021; Cui et al., IEEE TCSVT 2023):
- **Camera A (`camid = 0`)**: Person captured wearing Outfit 1.
- **Camera B (`camid = 1`)**: Same person captured wearing **Outfit 1 (Same-Clothes condition)**.
- **Camera C (`camid = 2`)**: Same person captured wearing **Outfit 2 (Cloth-Changing condition)**.

### The Inversion Error
Evaluating the modulo-2 formula across cameras:
$$\text{Cam A } (c=0): 0 \pmod 2 = 0 \implies \text{Clothes Label } = \text{pid} \times 2 + 0$$
$$\text{Cam B } (c=1): 1 \pmod 2 = 1 \implies \text{Clothes Label } = \text{pid} \times 2 + 1$$
$$\text{Cam C } (c=2): 2 \pmod 2 = 0 \implies \text{Clothes Label } = \text{pid} \times 2 + 0$$

### Consequences on Model Supervision:
1. **False Equivalence**: The network was penalized when it distinguished person $i$ in Cam C from person $i$ in Cam A, because both were assigned clothing label $\text{pid} \times 2 + 0$.
2. **False Disentanglement**: The network was penalized for recognizing that person $i$ in Cam A and Cam B wore the same clothes, because Cam B was assigned clothing label $\text{pid} \times 2 + 1$.
3. **Adversarial Loss Corruption**: The clothes adversarial loss $\mathcal{L}_{ca}$ and clothes classification loss $\mathcal{L}_c$ actively forced the model to treat the cloth-changed camera as identical clothing, destroying clothing-invariant feature learning.

---

## 3. Implemented Correction

The correction introduces an explicit, validated camera-to-clothing-condition mapping in `data/dataset_loader.py`:

```python
def generate_clothing_labels(pids, cams, is_prcc=True, num_clothes=100):
    if is_prcc:
        # Validate camera IDs strictly against {0, 1, 2}
        invalid_mask = (cams < 0) | (cams > 2)
        if invalid_mask.any():
            invalid_cams = cams[invalid_mask].tolist()
            raise ValueError(f"Invalid PRCC camera ID detected: {invalid_cams}. PRCC only supports cameras 0 (A), 1 (B), and 2 (C).")
        
        # Explicit mapping: 0 for Cam A (0) and B (1); 1 for Cam C (2)
        clothes_condition = (cams == 2).long()
        clothes_labels = (pids * 2 + clothes_condition) % num_clothes
    else:
        clothes_labels = (pids * 2 + (cams % 2)).long() % num_clothes
        
    return clothes_labels
```

### Corrected Behavior:
$$\text{Cam A } (c=0): \text{clothes\_condition} = 0 \implies \text{label} = \text{pid} \times 2 + 0$$
$$\text{Cam B } (c=1): \text{clothes\_condition} = 0 \implies \text{label} = \text{pid} \times 2 + 0 \quad (\text{Identical to Cam A!})$$
$$\text{Cam C } (c=2): \text{clothes\_condition} = 1 \implies \text{label} = \text{pid} \times 2 + 1 \quad (\text{Distinct Cloth-Changed!})$$

---

## 4. Verification and Automated Testing

Automated tests in `tests/test_dcr_reid.py` (`test_17_prcc_clothing_label_mapping`) verify:
1. `Cam A` and `Cam B` for the same person produce identical clothing labels (`labels[0] == labels[1]`).
2. `Cam A` and `Cam C` for the same person produce distinct clothing labels (`labels[0] != labels[1]`).
3. Different person identities in the same camera produce distinct clothing labels (`labels[0] != labels[1]`).
4. Camera IDs outside $\{0, 1, 2\}$ explicitly raise `ValueError`.
5. Non-PRCC datasets safely fall back to camera-conditioned parity without error.

**Status: VERIFIED & PASSED**
