# Evaluation Protocol Specification: DCR-ReID & PRCC Benchmarking

**Project**: Lightweight & Paper-Faithful Person Re-Identification
**Reference Paper**: *DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification*, IEEE TCSVT, Vol. 33, No. 8, August 2023
**Target Repository**: `https://github.com/Sanjay1266/Lightweight-Person-Re-Identification-Model`

---

## 1. Executive Summary

This document specifies the exact, reproducible evaluation protocols implemented across the project for evaluating Person Re-Identification (Re-ID) performance under cloth-changing and standard surveillance conditions. It formalizes query/gallery partitioning, same-camera filtering, Cumulative Matching Characteristics (CMC), and mean Average Precision (mAP) computation.

Two distinct evaluation tracks are supported:
1. **Track A (Paper-Faithful DCR-ReID)**: ResNet-50 backbone, 384×192 resolution, stacked pooling embedding (4096 dimensions), evaluating clothes-irrelevant representations ($F_i^+ \oplus F_i^t$).
2. **Track B (Lightweight DCR-ReID)**: Custom 4-stage residual backbone, 256×128 resolution, dual BNNeck embedding (512 dimensions), evaluating clothes-invariant representations ($f_{id}$).

---

## 2. Dataset Evaluation Protocols

### 2.1 PRCC (Person Re-ID Under Clothes Change) Protocol

The PRCC dataset consists of 33,698 images of 221 identities captured across three non-overlapping cameras (A, B, C):
- **Camera A**: Frontal view, daytime, room lighting, Clothes Set 1.
- **Camera B**: Oblique/side view, same lighting, Clothes Set 1 (Same Clothes as Camera A).
- **Camera C**: Frontal/oblique view, distinct room lighting, Clothes Set 2 (Different Clothes from Camera A and B).

#### Partitioning and Identity Split
- **Training Set**: 150 identities (17,896 images), IDs 0–149.
- **Testing Set**: 71 identities (10,800 images), IDs 150–220.
- **Leakage Invariant**: $\text{Train IDs} \cap \text{Test IDs} = \emptyset$ (Strict zero identity leakage verified).

#### Benchmark Configurations

| Benchmark Subset | Query Set | Gallery Set | Clothes Relationship | Evaluates |
|---|---|---|---|---|
| `prcc_cc` | Camera A (3,384 images, 71 IDs) | Camera C (3,543 images, 71 IDs) | **Cloth-Changing** ($c_q \neq c_g$) | Robustness to drastic clothing changes |
| `prcc_sc` | Camera A (3,384 images, 71 IDs) | Camera B (3,873 images, 71 IDs) | **Same Clothes** ($c_q = c_g$) | Standard cross-view feature discrimination |
| `prcc_all` | Camera A (3,384 images, 71 IDs) | Camera B + C (7,416 images, 71 IDs) | **Mixed Condition** | Real-world hybrid surveillance gallery |

### 2.2 Market-1501 Surveillance Protocol

For general surveillance evaluation, four standardized subsets are constructed:
- **`both_small`**: Combined small subset (475 query / 2,146 gallery images).
- **`with_bag`**: Query individuals carrying backpacks, handbags, or accessories (322 query / 1,131 gallery images).
- **`without_bag`**: Query individuals with unencumbered clothing (179 query / 986 gallery images).
- **`both_large`**: Full unconstrained gallery benchmark (1,265 query / 10,048 gallery images).

---

## 3. Mathematical Evaluation Metrics

### 3.1 Distance Metric
Cosine similarity is converted to distance for ranked retrieval:

$$D(q_i, g_j) = 1 - \frac{q_i \cdot g_j}{\|q_i\|_2 \|g_j\|_2}$$

Because all model representations are strictly $L_2$-normalized prior to distance computation ($\|q_i\|_2 = \|g_j\|_2 = 1.0$), this simplifies to:

$$D(q_i, g_j) = 1 - (q_i \cdot g_j)$$

### 3.2 Same-Camera Filtering (Junk Image Removal)
Following the standard Market-1501 evaluation protocol (`utils/metrics.py:eval_market1501`):
For a query $q_i$ with identity $p_i$ and camera $c_i$:
- Gallery samples $g_j$ where $p_j = p_i$ AND $c_j = c_i$ are marked as **junk** (same person under the exact same camera view is discarded from the ranking).
- Gallery samples with junk identity ($p_j = -1$ or unassigned) are discarded.
- Only gallery samples where $c_j \neq c_i$ are retained as valid positive or negative candidates.

### 3.3 Cumulative Matching Characteristics (CMC Rank-k)
For each valid query $q_i$, gallery items are sorted in ascending order of distance:

$$\text{CMC}(k) = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}\left( \min_{j \in \text{Pos}_i} \text{rank}(g_j) \le k \right)$$

Reported metrics include **Rank-1**, **Rank-5**, and **Rank-10**.

### 3.4 mean Average Precision (mAP)
Average Precision ($AP$) evaluates the precision across all recall levels for each query:

$$AP_i = \sum_{k=1}^{N_g} P(k) \times \Delta R(k) = \frac{1}{|\text{Pos}_i|} \sum_{k=1}^{N_g} P(k) \cdot \mathbb{I}(g_k \in \text{Pos}_i)$$

where $P(k)$ is the precision at rank $k$. mean Average Precision is the mean across all valid queries:

$$mAP = \frac{1}{|Q|} \sum_{i=1}^{|Q|} AP_i$$

---

## 4. Evaluation Execution Modes

The evaluation suite (`evaluate.py`) supports two distinct operational modes:

### 4.1 Fast Diagnostic Mode (`--fast`)
- **Purpose**: Rapid model sanity checking and regression detection during active development.
- **Sampling**: Stratified sampling across all query identities (64 query images, 224 gallery images) with random seed control (`--seed 42`).
- **Reproducibility Guarantee**: Exactly reproduces baseline checkpoint scores in ~2 seconds.

```powershell
python evaluate.py --fast --device auto
```

### 4.2 Full Exhaustive Mode (`--full`)
- **Purpose**: Exhaustive scientific benchmarking over the complete gallery and query sets.
- **Coverage**: Evaluates all 3,384 queries against 3,543 gallery images for PRCC CC, 3,873 images for PRCC SC, and 7,416 images for PRCC All.
- **Caching**: Employs query-feature memoization across sub-evaluations to eliminate redundant backbone forward passes.

```powershell
python evaluate.py --full --device cuda
```

---

## 5. Verification Checklist

| Criterion | Requirement | Status | Verification Reference |
|---|---|---|---|
| Query/Gallery Separation | Zero sample overlap between query and gallery sets | Verified | `tests/test_dcr_reid.py:test_07` |
| Identity Leakage | Train identities strictly disjoint from Test identities | Verified | `reports/evaluation_protocol_audit.md` |
| Same-Camera Discard | Same ID + Same Camera excluded from CMC/mAP calculation | Verified | `utils/metrics.py:eval_market1501` |
| Feature Normalization | Embeddings unit-normalized ($\|f\|_2 = 1.0$) prior to cosine distance | Verified | `tests/test_paper_dcr.py:test_06` |
| Checkpoint Preservation | Original baseline checkpoint hash immutable | Verified | `reports/baseline/baseline_metrics.json` |
