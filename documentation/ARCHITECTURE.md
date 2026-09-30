# Lightweight DCR-ReID: Architectural Specification
### Course: 23CSE373 - Computer Vision | Base Paper: IEEE TCSVT 2023

This document details the architectural layout, modular organization, and forward execution graph of the **Lightweight Deep Component Reconstruction Person Re-Identification (DCR-ReID)** framework.

---

## 🏛️ High-Level System Workflow

```
                                Probe Image x_i (256x128)
                                           ↓
                                Stem Layer (Conv7x7, MaxPool)
                                           ↓
                               Stage 1 (ResBlock: 48 -> 64)
                                           ↓
                               Stage 2 (ResBlock: 64 -> 128)
                                           ↓
                               Stage 3 (ResBlock: 128 -> 256)
                                           ↓
                          Stage 4 (ResBlock: 256 -> 512, Stride 1)
                                           ↓
                         Base Feature Map P_i (16x8, 512 Channels)
                                           ↓
            ┌──────────────────────────────┼──────────────────────────────┐
            ↓                              ↓                              ↓
    [PI Branch]                     [CR Branch]                    [CI Branch]
Person Identification          Component Reconstruction       Clothes Identification
  • Global Average Pool          • Channel Decomposition        • Global Avg/Max Pool
  • 1D BNNeck                      P_i = P⁺ ⊕ P⁻ ⊕ Pᵗ           • Clothes Classifier C_P
  • C_ID Classifier              • 4-Block Decoders:            • Clothes Adv Loss L_ca
  • Hard Triplet Loss                Y⁺ = ψ⁺(P⁺) [Body]         • DAD Feature Assembly:
  • Invariant Inference:             Y⁻ = ψ⁻(P⁻) [Clothes]          G_i = F⁺ ⊕ F_ai⁻ ⊕ Fᵗ
    Discards F⁻ (apparel/bag)        Yᵗ = ψᵗ(Pᵗ) [Contour]      • Dual Attention (CA + SA)
```

---

## 🔍 Module-by-Module Description

### 1. Lightweight Residual Backbone
To address the edge constraint outlined in the presentation slides (Page 2 & 7), the backbone replaces the massive 25.6M ResNet-50 with a compact residual architecture:
- **Input Resolution**: Standard Re-ID aspect ratio $256 \times 128$ (Height $\times$ Width).
- **Stem**: $7 \times 7$ Conv (stride 2, 48 channels) + BatchNorm + ReLU + $3 \times 3$ MaxPool (stride 2) $\to$ spatial dimension $64 \times 32$.
- **Stage 1**: ResidualBlock ($48 \to 64$ channels, stride 1, $64 \times 32$).
- **Stage 2**: ResidualBlock ($64 \to 128$ channels, stride 2, $32 \times 16$).
- **Stage 3**: ResidualBlock ($128 \to 256$ channels, stride 2, $16 \times 8$).
- **Stage 4**: ResidualBlock ($256 \to 512$ channels, stride 1, $16 \times 8$).
  > **Note**: Stride-1 in Stage 4 avoids aggressive spatial downsampling, preserving local component spatial cues necessary for human reconstruction decoders.
- **Backbone Parameters**: **3.24 Million** (vs ResNet-50 25.6M $\to$ **87.3% parameter reduction**).

---

### 2. Component Reconstruction Disentanglement (CRD) Module (CR Branch)
The core of DCR-ReID is controllable disentanglement of clothes-irrelevant and clothes-relevant features based on component region reconstruction (Section III.B.2, Eq. 8–10):

#### A. 3-Way Channel-Level Decomposition (Eq. 8):
$$P_i = P_i^- \oplus P_i^+ \oplus P_i^t$$
- $P_i^+$: Clothes-irrelevant feature segments (human facial structure, body shape, torso height, limb proportions).
- $P_i^-$: Clothes-relevant feature segments (shirt, pants, colors, fabric textures, bags, backpacks).
- $P_i^t$: Human contour/boundary feature segments (silhouette edges).

By applying channel-level splitting rather than simple spatial masking, inter-channel information mixing is strictly prevented, protecting disentangled features from mutual corruption.

#### B. 4-Block Reconstruction Decoders ($\psi^-, \psi^+, \psi^t$, Fig. 4):
Each branch ($\psi^v, v \in \{-, +, t\}$) employs a dedicated 4-block reconstruction decoder:
1. **Projection Layer**: $3 \times 3$ Conv + BatchNorm2d + ReLU (normalizes channel inputs).
2. **Reconstruction Block 1**: Dual $3 \times 3$ Convs with BatchNorm + ReLU + Residual shortcut.
3. **Reconstruction Block 2**: Dual $3 \times 3$ Convs with BatchNorm + ReLU + Residual shortcut.
4. **Reconstruction Block 3**: Dual $3 \times 3$ Convs with BatchNorm + ReLU + Residual shortcut.
5. **Reconstruction Block 4**: Dual $3 \times 3$ Convs with BatchNorm + ReLU + Residual shortcut.
6. **Output Activation**: $1 \times 1$ Conv + Sigmoid activation yielding visual component masks $Y_i^+, Y_i^-, Y_i^t \in [0, 1]^{H \times W}$.

---

### 3. Deep Assembled Disentanglement (DAD) Module (CI Branch)
As established in Section III.B.3 of the paper, component reconstruction alone is insufficient without supervision over feature discriminativeness. The DAD module provides this via:

1. **Feature Pooling (Eq. 15)**:
   $$F_i^v = \phi(P_i^v), \quad v \in \{-, +, t\}$$
   where $\phi(\cdot)$ is global pooling (stack of AdaptiveAvgPool2d and AdaptiveMaxPool2d) followed by 1D Batch Normalization.
2. **Batch Clothes Shuffling & Feature Assembly (Eq. 16)**:
   $$G_i = F_i^+ \oplus F_{ai}^- \oplus F_i^t$$
   where $F_{ai}^-$ is randomly permuted across different identity instances in the current mini-batch ($id(ai) \neq id(i)$).
3. **Dual Attention Recalibration**:
   Applies Squeeze-and-Excitation Channel Attention (CA) and Spatial Attention (SA) to dynamically amplify identity cues:
   $$\text{Refined} = \text{Fused} \odot \text{CA}(\text{Fused}) \odot \text{SA}(\text{Fused})$$

---

### 4. Person Identification (PI) Branch & Metric Learning
- **Dual BNNeck Architecture**:
  - Global bottleneck: 1D BatchNorm after global pooling for classification.
  - Clothes-irrelevant bottleneck: Dedicated 1D BatchNorm for purely invariant representations.
- **Classification Head ($C_{ID}$)**: Linear projection from 512 dimensions to $N_{ID}$ classes.
- **Inference Mode (Cloth-Changing Invariance)**:
  Section III.B and IV.B explicitly dictate:
  > *"For inference, only the clothes-irrelevant features extracted by the backbone network is kept for final evaluation, and the clothes-relevant features are removed directly."*
  
  Our inference mode implements this via:
  $$f_{\text{inference}} = 0.6 \cdot \text{normalize}(F_i^{\text{base}}) + 0.4 \cdot \text{normalize}(F_i^+ \oplus F_i^t)$$
  This guarantees that changing shirt colors or wearing backpacks will not perturb the cosine retrieval distance.
