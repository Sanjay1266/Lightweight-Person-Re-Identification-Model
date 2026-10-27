# Lightweight DCR-ReID: Training Pipeline & Gradient Audit Report

### Course: 23CSE373 - Computer Vision | Date: 2026-10-27

---

## 1. Executive Summary

A comprehensive architectural and algorithmic audit was conducted on the DCR-ReID training pipeline (`train.py`, `models/loss.py`, `data/dataset_loader.py`).

The audit reveals why the existing checkpoint (`checkpoints/best_model.pth`) remained at an early validation state (~4.4% - 4.7% Rank-1 on full PRCC benchmark), identifies two critical bugs in the loss supervision and batch sampling, and provides the corrected training methodology.

---

## 2. Training Pipeline Component Breakdown

### 1. Training / Validation / Test Separation
- **Train Split**: 150 unique person identities (IDs 92 to 332), 17,896 images across Cameras A, B, and C.
- **Test Split**: 71 unique person identities (IDs 1 to 328), Camera A (query: 3,384 images), Camera B (same-clothes gallery: 3,873 images), Camera C (cloth-changing gallery: 3,543 images).
- **Leakage Check**: Exactly **0 identities overlap** between train and test sets ($\text{Train} \cap \text{Test} = \emptyset$). Data partition integrity is fully verified.

### 2. Multi-Task Loss Formulation
The training objective implements the paper's two-stage loss formulation:
$$\mathcal{L} = \mathcal{L}_{ID} + \mathcal{L}_{\text{triplet}} + \lambda_1 \mathcal{L}_R + \lambda_2 \mathcal{L}_c + \lambda_3 \mathcal{L}_{ca}$$

| Loss Term | Formulation | Implementation & Status |
| :--- | :--- | :--- |
| $\mathcal{L}_{ID}$ | Label-smoothed Cross Entropy ($\epsilon=0.1$) | Implemented in `CrossEntropyLabelSmooth`, verified |
| $\mathcal{L}_{\text{triplet}}$ | Batch hard-mining Triplet Loss ($\text{margin}=0.3$) | Implemented in `TripletLoss`, requires PK sampler |
| $\mathcal{L}_R$ | 3-way L1 Component Reconstruction | Supervises $\psi^+, \psi^-, \psi^t$ against pseudo masks |
| $\mathcal{L}_c$ | Clothes Classification Cross Entropy | Implemented in `clothes_loss`, affected by label formula bug |
| $\mathcal{L}_{ca}$ | Clothes Adversarial Loss (Stage 2) | Smooths clothes logits to disentangle clothing bias |

### 3. Critical Training Flaws Discovered

#### Flaw 1: Inverted PRCC Clothing Labels
- **Code in `train.py` line 155**:
  `clothes_labels = (pids * 2 + (cams % 2)).long() % num_clothes`
- **Mechanism**:
  - Cam A (`camid=0`) and Cam C (`camid=2`) both evaluate to $0 \pmod 2$.
  - Cam B (`camid=1`) evaluates to $1 \pmod 2$.
- **Consequence**: The model was supervised to believe that Cam A and Cam C share the same clothes, while Cam B has different clothes. In reality, Cam A and Cam B are identical clothing, and Cam C is the cloth-changing condition. This directly corrupted the clothing classification head and adversarial loss.
- **Correction**:
  `clothes_labels = (pids * 2 + (1 if camid == 2 else 0)).long() % num_clothes`

#### Flaw 2: Batch Sampling Strategy (Absence of PK Sampler)
- **Code in `data/dataset_loader.py`**:
  `DataLoader(..., shuffle=True, drop_last=True)`
- **Mechanism**: Standard uniform shuffling with batch size 16 across 150 identities and 17,896 images results in virtually all batches having at most 1 image per identity.
- **Consequence**: Hard triplet loss cannot find positive pairs ($P(i)$), defaulting to self-distance ($0.0$). The triplet loss can only push negatives, but cannot pull intraclass features together.
- **Correction**: Implement `RandomIdentitySampler` providing $P=4$ identities with $K=4$ instances each ($P \times K = 16$).

#### Flaw 3: Untrained `bottleneck_irrel` in Dual BNNeck
- **Mechanism**: `feat_irrel_normed` was created during forward pass but omitted from the dictionary returned to `train.py`.
- **Consequence**: `bottleneck_irrel.weight` stayed at $1.0$ without receiving gradient backpropagation.
- **Correction**: Include `feat_irrel_normed` in loss calculation or derive inference representation through the attention-refined `feat_normed`.

#### Flaw 4: Early Stopping at Validation Smoke Test
- In `documentation/TRAINING_AND_VALIDATION.md`, the logged training run was a single 0.31-minute epoch test. The initial cross-entropy loss was $7.84$ ($-\ln(1/150) \approx 5.01$ for identity classifier + clothes loss). The model was saved because $2.68\% > 0.0\%$, leaving the weights in an early, unconverged state.

---

## 4. Hardware and Convergence Strategy

1. **CPU Execution Speed**: On local 4-core AMD CPU, 1 full epoch of 17,896 images takes ~18-20 minutes. A full 60-epoch convergence run on CPU requires ~18-20 hours.
2. **Controlled Fine-Tuning Execution**:
   - Stage B (Smoke test): Verify forward, backward, loss, optimizer, and checkpoint reload in a controlled mini-run.
   - Stage C & D (Controlled Fine-Tuning): Run a dedicated fine-tuning experiment with the corrected PK sampler and corrected clothing labels to observe gradient descent and validate that retrieval metrics improve.
