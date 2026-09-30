# Lightweight DCR-ReID: Mathematical Formulation
### Course: 23CSE373 - Computer Vision | Base Paper: IEEE TCSVT 2023

This document catalogs every mathematical equation, loss formulation, and optimization protocol defined in the base research paper:
> **Zhenyu Cui, Jiahuan Zhou, Yuxin Peng, Shiliang Zhang, Yaowei Wang**, *"DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification"*, IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol. 33, No. 8, August 2023.

---

## 1. Problem Formulation & Preliminary

### General Person Re-ID (Eq. 1)
Given gallery set $\{g_i\}_{i=1}^N$ with $L$ identities and probe query $\{q_j\}_{j=1}^M$, person Re-ID determines:
$$i^* = \arg\min_{i=1, \dots, N} d(\phi(q_j; \theta), \phi(g_i; \theta)) \tag{1}$$
where $d(\cdot, \cdot)$ is cosine or Euclidean feature distance.

### The Cloth-Changing Challenge (Eq. 2–4)
In cloth-changing scenarios (CC-ReID), individuals with different identities may wear similar clothing:
$$d(w(q_{j'}), w(g_i)) \ll d(w(q_j), w(g_i)) \tag{2}$$
where $w(\cdot)$ is the clothes representation, $q_j$ and $g_i$ share identity ($id(q_j) = id(g_i)$), but $q_{j'}$ is a different identity ($id(q_{j'}) \neq id(g_i)$).

This leads to the breakdown where intra-identity distance exceeds inter-identity distance:
$$\| d(\phi(q_j; \theta), \phi(g_i; \theta)) - d(\phi(q_{j'}; \theta), \phi(g_i; \theta)) \| \gg 0 \tag{3}$$

DCR-ReID optimizes:
$$\min_{\theta} \sum_{j=1}^M \sum_{j'=1}^M \left( d(\phi(q_j; \theta), \phi(g_i; \theta)) - d(\phi(q_{j'}; \theta), \phi(g_i; \theta)) \right) \tag{4}$$

---

## 2. Person Identification (PI) Branch Formulation

### Identity Classification with Label Smoothing (Eq. 5–6)
Given instance $x_i$ with identity label $l_i$, the identification loss $L_{ID}$ is:
$$L_{ID} = - \sum_{i=1}^N \log \left( \frac{y(x_i, l_i)}{\sum_{j=1}^{N_{ID}} y(x_i, l_j)} \right) \tag{5}$$
$$y(x_i, l) = C_{ID}(\phi(x_i; \theta) \mid l) \tag{6}$$

With label smoothing ($\epsilon = 0.1$):
$$q(l \mid x_i) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{N_{ID}}, & l = l_i \\ \frac{\epsilon}{N_{ID}}, & l \neq l_i \end{cases}$$

### Hard-Mining Triplet Loss
$$L_{\text{triplet}} = \frac{1}{N} \sum_{i=1}^N \left[ \max_{p \in \mathcal{P}(i)} d(f_i, f_p) - \min_{n \in \mathcal{N}(i)} d(f_i, f_n) + m \right]_+$$
where $m = 0.3$ is the margin.

---

## 3. Component Reconstruction (CR) Branch Formulation

### Convolutional Feature Decomposition (Eq. 7–8)
Deep feature map from backbone $E(\cdot; \theta)$:
$$P_i = E(x_i; \theta) \tag{7}$$

Decomposed via channel-level splitting into 3 disjoint sub-tensors:
$$P_i = P_i^- \oplus P_i^+ \oplus P_i^t \tag{8}$$
where:
- $P_i^-$: Clothing region features (clothes-relevant)
- $P_i^+$: Non-clothing body region features (clothes-irrelevant)
- $P_i^t$: Human contour/boundary features
- $\oplus$: Channel-wise concatenation

### Visual Reconstruction (Eq. 9)
Reconstructed masks from dedicated decoders $\psi^v$:
$$Y_i^v = \psi^v(P_i^v), \quad v \in \{-, +, t\} \tag{9}$$

### Component Reconstruction Loss $L_R$ (Eq. 10)
Supervised using target masks $T_i^v$ (ground truth or anatomical pseudo-masks):
$$L_R = \frac{1}{N} \sum_{i=1}^N \sum_{v \in \{-, +, t\}} \ell_1(T_i^v, Y_i^v) \tag{10}$$

---

## 4. Clothes Identification (CI) Branch & DAD Module

### Clothes Classification Loss $L_c$ (Eq. 11–12)
$$L_c = - \sum_{i=1}^N \log \left( \frac{u(x_i, c_i)}{\sum_{j=1}^{N_U} u(x_i, c_j)} \right) \tag{11}$$
$$u(x_i, c) = C_P(\phi(x_i; \theta) \mid c) \tag{12}$$
where $c_i$ is the fine-grained clothes category and $N_U$ is the number of clothes classes.

### Clothes Adversarial Loss $L_{ca}$ (Eq. 13–14)
Forces the model to learn clothes-irrelevant features by smoothing probabilities across clothes classes under the same identity:
$$L_{ca} = - \sum_{i=1}^N \sum_{c=1}^{N_U} q(c) \log \left( \frac{u(x_i, c)}{u(x_i, c) + \sum_{id(j) \neq id(i)} u(x_i, j)} \right) \tag{13}$$
$$q(c) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{K}, & c = c_i \\ \frac{\epsilon}{K}, & c \neq c_i \text{ and } id(c) = id(i) \\ 0, & id(c) \neq id(i) \end{cases} \tag{14}$$
where $K$ is the number of clothes categories belonging to identity $id(i)$, and $\epsilon = 0.1$.

### Feature Assembly (Eq. 15–16)
Global pooling with batch normalization:
$$F_i^v = \phi(P_i^v), \quad v \in \{-, +, t\} \tag{15}$$

Batch-shuffled assembled feature vector:
$$G_i = F_i^+ \oplus F_{ai}^- \oplus F_i^t \tag{16}$$
where $F_{ai}^-$ is randomly permuted from a different identity $id(ai) \neq id(i)$.

### Assembled Clothes Loss $L_{ac}$ (Eq. 17–18)
$$L_{ac} = - \sum_{i=1}^N \sum_{c=1}^{N_U} h(c) \log \left( \frac{C_P(G_i \mid c)}{C_P(G_i \mid c) + \sum_{id(j) \neq id(i)} C_P(G_i \mid j)} \right) \tag{17}$$
$$h(c) = \begin{cases} \frac{1}{K}, & id(c) = id(i) \\ 0, & id(c) \neq id(i) \end{cases} \tag{18}$$

### CI Branch Total Loss (Eq. 19)
$$L_C = L_c + L_{ca} + \alpha L_{ac} + \gamma L_{ID}' \tag{19}$$
where $\alpha = 0.05$, $\gamma = 1.0$, and $L_{ID}'$ is the identity loss evaluated on the assembled feature $G_i$.

---

## 5. Total Unified Loss & Two-Stage Optimization (Eq. 20)

$$L = L_{ID} + L_{\text{triplet}} + L_C + L_R \tag{20}$$

### Two-Stage Optimization Schedule:
- **Stage 1 (Disentanglement Initialization)**:
  $$L_{\text{Stage 1}} = L_{ID} + L_{\text{triplet}} + L_c + L_R$$
  Trains a well-grounded clothes classifier and establishes stable component reconstruction decoders.
- **Stage 2 (Full Adversarial Optimization)**:
  $$L_{\text{Stage 2}} = L_{ID} + L_{\text{triplet}} + L_c + L_{ca} + \alpha L_{ac} + \gamma L_{ID}' + L_R$$
  Activates the DAD feature assembly and clothes adversarial loss to force identity classification to rely solely on clothing-invariant cues.
