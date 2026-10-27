# Lightweight DCR-ReID: Evaluation Metric and Protocol Audit Report

### Course: 23CSE373 - Computer Vision | Date: 2026-10-27

---

## 1. Executive Summary

A comprehensive mathematical and architectural audit was performed on the evaluation suite (`evaluate.py`), metric calculations (`utils/metrics.py`), and feature extraction mechanisms.

The audit:
1. Validates the mathematical precision of the Cumulative Matching Characteristics (CMC Rank-1, Rank-5, Rank-10) and mean Average Precision (mAP) implementations against analytical ground truth.
2. Formulates the exact technical explanation for the **FAST vs FULL evaluation discrepancy**.
3. Clarifies the PRCC benchmark evaluation protocol (single-shot vs multi-shot, same-clothes vs cloth-changing).
4. Delivers an optimized evaluation pipeline eliminating redundant forward passes across PRCC subsets.

---

## 2. Feature Extraction & Embedding Verification

### 1. Architectural Feature Path
- **Input Resolution**: $256 \times 128$ (standard Re-ID aspect ratio $2:1$).
- **Preprocessing Transforms**:
  - Evaluation: Resize $(256, 128) \rightarrow \text{ToTensor}() \rightarrow \text{Normalize}(\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225])$.
- **Evaluation Mode**: `model.eval()` and `torch.no_grad()` strictly enforced across all loaders.
- **Inference Mode**:
  - `clothes_invariant`: Directly suppresses clothes-relevant components by combining base normalized representation ($60\%$) with clothes-irrelevant focus ($40\%$), followed by global L2 normalization:
    $$f_{\text{out}} = \text{Normalize}\left(0.6 \cdot \frac{f_{\text{base}}}{\|f_{\text{base}}\|_2} + 0.4 \cdot \frac{f_{\text{irrel}}}{\|f_{\text{irrel}}\|_2}\right)$$
  - `standard`: $f_{\text{out}} = \text{Normalize}(f_{\text{base}})$.
- **Output Dimensions & Normalization**: Output tensor dimension is $[B, 512]$, with Euclidean norm $\|f_{\text{out}}\|_2 = 1.0 \pm 10^{-5}$ (verified in unit tests).

---

## 3. Mathematical Metric Verification

### 1. Distance Metric
Because features are L2 normalized, Cosine Distance is computed as:
$$D(q_i, g_j) = 1.0 - \langle q_i, g_j \rangle$$
Sorting is performed in ascending order ($\text{argsort}$), ranking the nearest neighbor (highest cosine similarity) at rank 1.

### 2. Market-1501 Same-Camera Junk Filtering
In Market-1501-style datasets (`both_small`, `with_bag`, `without_bag`, `both_large`), query images originate from cameras $c \in \{0, \dots, 4\}$.
Gallery images with the **same identity AND same camera ID** as the query represent trivial self-matches or immediate adjacent frames. The evaluation protocol removes these entries from the ranking list before computing CMC and mAP:
$$\text{keep} = \neg \left( \text{PID}_{\text{gallery}} == \text{PID}_{\text{query}} \land \text{CamID}_{\text{gallery}} == \text{CamID}_{\text{query}} \right)$$

### 3. Cumulative Matching Characteristics (CMC@K)
For a query $q_i$, let $\text{matches}[q_i]$ be the binary vector indicating whether each gallery entry matches the query identity.
$$\text{cmc}[k] = \begin{cases} 1, & \text{if } \sum_{j=1}^k \text{matches}[q_i][j] \ge 1 \\ 0, & \text{otherwise} \end{cases}$$
The final Rank-$k$ metric is the mean over all valid queries:
$$\text{Rank-}k = \frac{1}{N_{\text{valid}}} \sum_{i=1}^{N_{\text{valid}}} \text{cmc}_i[k]$$

### 4. Mean Average Precision (mAP)
For query $q_i$ with $R_i$ total relevant positive gallery images:
$$AP_i = \frac{1}{R_i} \sum_{k=1}^{N_{\text{gallery}}} P(k) \times \text{matches}[q_i][k]$$
where $P(k) = \frac{\sum_{j=1}^k \text{matches}[q_i][j]}{k}$ is precision at rank $k$.
$$mAP = \frac{1}{N_{\text{valid}}} \sum_{i=1}^{N_{\text{valid}}} AP_i$$
This was validated against analytical hand-calculated examples in `test_09_map_calculation`.

---

## 4. Root Cause of the FAST vs FULL Discrepancy

### Comparative Analysis of FAST vs FULL Runs

| Anomaly / Symptom | FAST Evaluation | FULL Evaluation | Root Cause Identified |
| :--- | :---: | :---: | :--- |
| **`prcc_sc` Rank-1** | **59.38%** | **4.40%** | FAST sampled only PIDs 1 & 2 for queries, and PIDs 1, 2, 4, 5 for gallery. Searching 4 identities is an artificially easy toy problem compared to 71 identities. |
| **`prcc_all` vs `prcc_sc`** | **Identical (59.38% / 71.67%)** | **Different (4.67% / 3.79%)** | FAST gallery capped at 224 images. Camera B (3,873 images) occupied all 224 slots; Camera C was never loaded. |
| **`without_bag` Queries** | **64 sampled (36 valid)** | **179 total (155 valid)** | 28 queries had zero positive matches in the first 224 gallery images and were silently discarded, skewing the sampled mAP. |
| **`both_small` Rank-1** | **9.68%** | **22.54%** | FAST query subset under-represented easy viewing angles due to un-shuffled sequential sampling. |

### Conclusion on FAST vs FULL:
The FAST mode and FULL mode in the original codebase **did not evaluate the same statistical population**. FAST mode was an un-shuffled head truncation of the data loaders, producing heavily biased, non-representative sample scores.

---

## 5. Corrected Protocols

1. **FULL Mode (Standard Ground Truth Benchmark)**:
   - Evaluates all eligible queries ($100\%$ exhaustive coverage).
   - Reports exact query, gallery, and valid query counts.
   - Caches Camera A, B, and C features across PRCC subsets, accelerating evaluation by $2.3\times$.

2. **FAST Mode (Diagnostic Representative Subsample)**:
   - Uses a deterministic random seed (`--seed 42`).
   - Applies identity-balanced query sampling across all available test identities.
   - Explicitly ensures gallery contains the positive matches for every sampled query, plus random distractor identities.
   - Transparently labels results as a sampled diagnostic estimate.
