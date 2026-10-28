# Comprehensive Evaluation Protocol & Dataset Audit Report

**Project**: Lightweight Person Re-Identification Model (DCR-ReID)  
**Date**: 2026-10-28 20:28:30 IST  
**Audit Scope**: Dataset integrity, train/test split isolation, clothing-condition supervision, Market-1501 junk filtering, FAST vs FULL evaluation protocols.

---

## 1. Dataset Partition & Split Verification

Every subset under `Data_set/` was audited for directory existence, image counts, identity IDs, and cross-split leakage.

| Dataset Subset | Split Name | Image Count | Unique Identities | Camera Indices | Cross-Split Identity Overlap ($\text{Train} \cap \text{Test}$) | Leakage Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **both_small** | Train | 2,919 | 250 | $c \in [0, 5]$ | 0 | **ZERO LEAKAGE** |
| | Gallery | 2,146 | 250 | $c \in [0, 5]$ | - | - |
| | Query | 475 (457 valid) | 250 | $c \in [0, 5]$ | - | - |
| **with_bag** | Train | 1,157 | 82 | $c \in [0, 5]$ | 0 | **ZERO LEAKAGE** |
| | Gallery | 1,131 | 82 | $c \in [0, 5]$ | - | - |
| | Query | 322 (318 valid) | 82 | $c \in [0, 5]$ | - | - |
| **without_bag** | Train | 1,770 | 169 | $c \in [0, 5]$ | 0 | **ZERO LEAKAGE** |
| | Gallery | 986 | 167 | $c \in [0, 5]$ | - | - |
| | Query | 179 (155 valid) | 167 | $c \in [0, 5]$ | - | - |
| **both_large** | Train | 11,719 | 250 | $c \in [0, 5]$ | 0 | **ZERO LEAKAGE** |
| | Gallery | 10,048 | 250 | $c \in [0, 5]$ | - | - |
| | Query | 1,265 (1,244 valid)| 250 | $c \in [0, 5]$ | - | - |
| **PRCC (prcc_cc)** | Train | 17,896 | 150 | A(0), B(1), C(2) | 0 | **ZERO LEAKAGE** |
| | Gallery (C) | 3,543 | 71 | C(2) | - | - |
| | Query (A) | 3,384 | 71 | A(0) | - | - |
| **PRCC (prcc_sc)** | Gallery (B) | 3,873 | 71 | B(1) | - | - |
| | Query (A) | 3,384 | 71 | A(0) | - | - |
| **PRCC (prcc_all)**| Gallery (B+C)| 7,416 | 71 | B(1), C(2) | - | - |
| | Query (A) | 3,384 | 71 | A(0) | - | - |

> [!IMPORTANT]
> **Identity Disjointness Proof**:
> - PRCC Train PIDs: 150 directories (`'092'`, `'095'`, ..., `'332'`).
> - PRCC Test PIDs: 71 directories (`'001'`, `'002'`, ..., `'328'`).
> - Mathematical set intersection: $\text{Train} \cap \text{Test} = \emptyset$ (Exactly 0 identities shared).
> - Market-1501 subsets similarly have zero identity overlap between `bounding_box_train` and test/query splits.

---

## 2. PRCC Camera Conditions & Clothing Supervision Mapping

The PRCC benchmark defines three distinct camera recording setups:
- **Camera A (`camid = 0`)**: Query view, subject in Outfit 1.
- **Camera B (`camid = 1`)**: Gallery view, subject in **Outfit 1 (Same Clothes)**.
- **Camera C (`camid = 2`)**: Gallery view, subject in **Outfit 2 (Cloth-Changing)**.

### Audited Supervision Bug & Fix:
- **Historical Defect**: `clothes_labels = (pids * 2 + (cams % 2)) % num_clothes`.
  Under this formula, $\text{cams}=0$ and $\text{cams}=2$ both yielded $0 \pmod 2$. Consequently, Camera A and Camera C were assigned identical clothing labels, while Camera B was isolated. This inverted the cloth-changing training objective.
- **Corrected Formulation**:
  ```python
  clothes_condition = (cams == 2).long()  # 0 for Cam A & B, 1 for Cam C
  clothes_labels = (pids * 2 + clothes_condition) % num_clothes
  ```
  Now, Camera A and Camera B share clothing label $\text{pid} \times 2 + 0$, while Camera C is assigned $\text{pid} \times 2 + 1$.

---

## 3. Market-1501 Evaluation Protocol & Junk Filtering

For each query image $q$ with identity $y_q$ and camera $c_q$:
1. **Gallery Sorting**: Compute cosine distances $D(q, g_j) = 1.0 - \langle f_q, f_{g_j} \rangle$ and sort gallery indices in ascending order.
2. **Junk Removal**: Exclude any gallery image where $y_{g_j} == y_q \land c_{g_j} == c_q$ (self-frame match from same camera view).
3. **Invalid Query Handling**: If a query has no remaining cross-camera matching positive gallery samples ($y_{g_j} == y_q \land c_{g_j} \neq c_q$), the query is treated as invalid and excluded from CMC cumulative sum and average precision averaging, avoiding division by zero.
4. **Valid Query Counts Verified**:
   - `both_small`: 457 valid / 475 total (18 queries have all matches in identical camera).
   - `with_bag`: 318 valid / 322 total (4 queries excluded).
   - `without_bag`: 155 valid / 179 total (24 queries excluded).
   - `both_large`: 1,244 valid / 1,265 total (21 queries excluded).
   - `prcc_cc`, `prcc_sc`, `prcc_all`: 3,384 valid / 3,384 total (100% valid; all queries have cross-camera targets in B or C).

---

## 4. FAST Diagnostic Mode vs FULL Benchmark Mode

| Dimension | FAST Diagnostic Evaluation | FULL Benchmark Evaluation |
|:---|:---|:---|
| **Intended Purpose** | Rapid sanity check during development | Final benchmark verification & paper comparison |
| **Query Selection** | Stratified sampling across all identities (max 64) | 100% complete exhaustive query split |
| **Gallery Selection** | Matched positive instances + random distractors (max 224) | 100% complete exhaustive gallery split |
| **Reproducibility** | Governed by deterministic seed (`--seed 42`) | Deterministic (all images included in dataset order) |
| **Comparability** | Baseline FAST vs Improved FAST | Baseline FULL vs Improved FULL |

> [!WARNING]
> Scores from FAST diagnostic mode must NEVER be compared against FULL benchmark mode, because gallery distractor sizes directly affect Rank-1 and mAP retrieval difficulty. Both baseline and improved checkpoints must be evaluated under identical evaluation modes.

