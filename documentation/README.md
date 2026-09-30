# Lightweight DCR-ReID: Project Documentation Index
### 23CSE373 - Computer Vision | Amrita School of Engineering

Welcome to the comprehensive technical documentation for the **Lightweight Person Re-Identification Model (DCR-ReID)**.

---

## 👥 Authors & Course Information
- **Course**: 23CSE373 - Computer Vision
- **Base Paper**: *DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification* (IEEE Transactions on Circuits and Systems for Video Technology, 2023)
- **Team Members**:
  - **G N Bhuvaneshwaran** (`CB.SC.U4CSE24218`)
  - **Sanjay MS** (`CB.SC.U4CSE24248`)
  - **Sanjay S** (`CB.SC.U4CSE24249`)

---

## 📚 Documentation Table of Contents

| Document | Description |
| :--- | :--- |
| **[1. System Architecture](file:///s:/5th%20Sem/CV/Lightweight_Person_Re-Identification_Model/documentation/ARCHITECTURE.md)** | Deep dive into the 3-branch DCR-ReID framework (PI, CR, CI), CRD 3-way channel decomposition, DAD feature assembly, 4-block decoders, and clothing-invariant inference. |
| **[2. Mathematical Formulation](file:///s:/5th%20Sem/CV/Lightweight_Person_Re-Identification_Model/documentation/MATHEMATICAL_FORMULATION.md)** | Full paper mathematical definitions, equations (Eq. 1 to Eq. 20), component reconstruction loss, clothes classification, adversarial smoothing, and two-stage optimization. |
| **[3. Code Changes & Git Diff](file:///s:/5th%20Sem/CV/Lightweight_Person_Re-Identification_Model/documentation/CODE_CHANGES_AND_DIFF.md)** | Detailed audit of all modifications between the previous baseline and the updated implementation across all 10 modified files. |
| **[4. Benchmarks & Convergence Tracking](file:///s:/5th%20Sem/CV/Lightweight_Person_Re-Identification_Model/documentation/BENCHMARK_AND_CONVERGENCE.md)** | Cross-condition evaluation results (CMC Rank-1/5/10, mAP), current checkpoint accuracy vs final paper convergence targets, and edge complexity vs ResNet-50. |
| **[5. User & Deployment Guide](file:///s:/5th%20Sem/CV/Lightweight_Person_Re-Identification_Model/documentation/USER_GUIDE.md)** | Step-by-step instructions for running benchmark evaluation, CLI query search, two-stage PyTorch model training, and the Flask Glassmorphism Web Dashboard. |

---

## 🌟 Executive Summary

Cloth-changing person re-identification (CC-ReID) is a major open challenge in long-term surveillance. Traditional Re-ID models overfit to clothing colors and patterns; when suspects change clothes, add jackets, or carry backpacks, conventional models fail.

The **Lightweight DCR-ReID** system solves this by:
1. **Disentangling** representations at the convolutional channel level ($P_i = P_i^- \oplus P_i^+ \oplus P_i^t$).
2. **Reconstructing** human body components and silhouette contours to strictly regularize clothes-irrelevant features.
3. **Assembling** features ($G_i = F_i^+ \oplus F_{ai}^- \oplus F_i^t$) with permuted clothes to enforce invariant identity classification.
4. **Stripping** clothes-relevant features ($F_i^-$) during inference, achieving pure clothing-invariant metric matching.
5. **Compressing** the model to 3.24M parameters (87.3% smaller than ResNet-50) for 54+ FPS real-time edge execution.
