# Evaluation Parity & Metric Protocol Audit Report

### Course: 23CSE373 - Computer Vision | Date: 2026-10-27
**Project**: Lightweight Person Re-Identification Model (DCR-ReID)  
**Root Repository**: `S:\5th Sem\CV\Lightweight_Person_Re-Identification_Model`

---

## 1. Executive Summary

This report establishes the evaluation parity framework for benchmark comparisons between the original baseline checkpoint (`checkpoints/best_model.pth`) and the fine-tuned checkpoint (`checkpoints/experiments/dcr_reid_corrected_pk_v1.pth`).

To ensure absolute scientific fairness:
1. Both models are evaluated using the **exact same code paths, distance computations, and metric functions**.
2. Both models are evaluated against **identical dataset manifests**.
3. FAST mode uses **deterministic, stratified representative sampling** rather than sequential truncation.
4. FULL mode evaluates the **100% exhaustive dataset population**.

---

## 2. Root Cause of Historical Evaluation Anomalies

### 1. The FAST Mode Inflation and Duplication Bug
In the original evaluation implementation:
```python
for batch in loader:
    ...
    count += len(b_pids)
    if max_items and count >= max_items:
        break
```
- **Consequence 1 (Identity Collapse)**: Because `DataLoader(shuffle=False)` reads directories sorted by PID (`001`, `002`...), the first 64 queries selected ONLY 2 identities (PID 1 and 2). The first 224 gallery images selected ONLY 4 identities (PID 1, 2, 4, 5). The retrieval task was collapsed into an artificial 4-class toy classification problem, inflating FAST Rank-1 to 59.38%.
- **Consequence 2 (`prcc_sc` vs `prcc_all` Duplication)**: In `prcc_all`, the gallery loader concatenated Camera B (3,873 images) and Camera C (3,543 images). Sequential truncation at 224 images stopped within Camera B; **Camera C was never loaded**. Thus, `prcc_all` evaluated against the identical gallery as `prcc_sc`, generating 100% duplicate metrics.

### 2. The Stratified FAST Resolution
The evaluation suite in `evaluate.py` was upgraded with **stratified representative sampling**:
1. **Query Sampling**: Draws 64 queries uniformly across all 71 available test identities.
2. **Gallery Sampling**: Guarantees cross-camera positive matches for each sampled query identity, then fills remaining slots with random distractors spanning 70 distinct identities.
3. **PRCC Combined (`prcc_all`)**: Draws gallery images proportionally from both Camera B (same-clothes) and Camera C (cloth-changing).
4. **Reproducibility**: Governed by deterministic seed `--seed 42`.

---

## 3. Mathematical Metric Formulation

### 1. Cosine Distance Matrix
Inference embeddings are strictly $L_2$ normalized on the unit hypersphere:
$$\|f\|_2 = 1.0 \implies D(q_i, g_j) = 1.0 - \langle q_i, g_j \rangle$$
Sorting indices in ascending order yields nearest neighbors at Rank 1.

### 2. Market-1501 Junk Filtering
For query $(y_q, c_q)$ and gallery candidate $(y_g, c_g)$:
$$\text{Exclude if: } y_g == y_q \land c_g == c_q$$
This removes trivial self-matches from identical surveillance camera frames.

### 3. Cumulative Matching Characteristics (CMC@K)
$$\text{cmc}_i[k] = \begin{cases} 1, & \text{if } \sum_{j=1}^k \text{matches}_i[j] \ge 1 \\ 0, & \text{otherwise} \end{cases}$$
$$\text{Rank-}k = \frac{1}{N_{\text{valid}}} \sum_{i=1}^{N_{\text{valid}}} \text{cmc}_i[k]$$

### 4. Mean Average Precision (mAP)
For query $i$ with $R_i$ total relevant positive gallery images:
$$AP_i = \frac{1}{R_i} \sum_{k=1}^{N_{\text{gallery}}} P(k) \times \text{matches}_i[k], \quad P(k) = \frac{\sum_{j=1}^k \text{matches}_i[j]}{k}$$
$$mAP = \frac{1}{N_{\text{valid}}} \sum_{i=1}^{N_{\text{valid}}} AP_i$$

---

## 4. Evaluation Parity Matrix

| Feature / Setting | Original Baseline Checkpoint | CUDA-Trained Best Checkpoint | Parity Status |
| :--- | :---: | :---: | :---: |
| **Model Weights File** | `checkpoints/best_model.pth` | `checkpoints/experiments/dcr_reid_cuda_v1/best_model.pth` | Verified Separate |
| **SHA-256 Checkpoint Hash** | `e5bfb5aa627fc3a10f27c01d03bdb54ba8cda06cb3453d24484cd8142e034730` | `4352d72c75c81db69ce06ad743190830ac7e74b5ce46fc5493c7572505966121` | Verified Distinct |
| **Loaded Tensors** | 162 / 162 (100%) | 162 / 162 (100%) | **Identical** |
| **Embedding Dimension**| 512 ($L_2$ Normalized, norm = 1.0) | 512 ($L_2$ Normalized, norm = 1.0) | **Identical** |
| **Inference Mode** | `clothes_invariant` | `clothes_invariant` | **Identical** |
| **Input Resolution** | $256 \times 128 \times 3$ | $256 \times 128 \times 3$ | **Identical** |
| **Evaluation Device** | CUDA / CPU | CUDA / CPU | **Identical** |
| **Metric Implementation**| `utils/metrics.py` (Market-1501) | `utils/metrics.py` (Market-1501) | **Identical** |
| **Junk Exclusion Rules**| Same-camera positive exclusion | Same-camera positive exclusion | **Identical** |
| **Evaluation Code Path**| `run_full_parity_benchmark.py` | `run_full_parity_benchmark.py` | **Identical** |

---

## 5. Seven-Subset FULL Manifest & Population Audit

Both models were benchmarked across all 7 evaluation subsets under 100% full un-truncated manifests:

| Subset Name | Total Queries | Valid Queries | Excluded Queries | Gallery Images | Distinct Identities | Camera Setup & Evaluation Protocol |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **both_small** | 475 | 457 | 18 (no cross-cam match) | 2,146 | 250 | Multi-camera campus surveillance; mixed accessories |
| **with_bag** | 322 | 318 | 4 (no cross-cam match) | 1,131 | 82 | Backpack/handbag/accessory occlusion subset |
| **without_bag** | 179 | 155 | 24 (no cross-cam match) | 986 | 167 | Clean surveillance without occluding bags |
| **both_large** | 1,265 | 1,244 | 21 (no cross-cam match) | 10,048 | 250 | Large-scale distractor gallery benchmark |
| **prcc_cc** | 3,384 | 3,384 | 0 | 3,543 | 71 | Query: Cam A; Gallery: Cam C (Cloth-Changing) |
| **prcc_sc** | 3,384 | 3,384 | 0 | 3,873 | 71 | Query: Cam A; Gallery: Cam B (Same-Clothes) |
| **prcc_all** | 3,384 | 3,384 | 0 | 7,416 | 71 | Query: Cam A; Gallery: Combined Cams B + C |

*Verification*: Every single query and gallery sample follows the identical loader generator and indexing between baseline and CUDA checkpoints. Valid query counts are mathematically determined by $\sum_{j} \mathbb{I}(y_g == y_q \land c_g \neq c_q) > 0$.

