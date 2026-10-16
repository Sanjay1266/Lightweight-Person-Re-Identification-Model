# Lightweight DCR-ReID: Model Outputs & Technical Specification
### Course: 23CSE373 - Computer Vision | Base Paper: IEEE TCSVT 2023
**Model Class**: `LightweightDCRReID` (`models/lightweight_reid.py`)

---

## 📌 Executive Summary of Outputs

The **Lightweight DCR-ReID** model operates under two distinct operational modes (Inference vs. Training) and provides dedicated methods for interpretability and feature extraction.

```
                           Input Image Crop: (B, 3, 256, 128)
                                           │
                                           ▼
                         Lightweight Backbone (4.90M Params | 9.78M Total)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          [INFERENCE MODE]                                [TRAINING MODE]
           model.eval()                                    model.train()
                    │                                             │
      Normalized Embedding: (B, 512)                     11-Key Output Dictionary:
      • Clothes-Invariant (0.6 Base + 0.4 Irrel)         • cls_score, clothes_score
      • Standard (Base Normalized)                       • global_feat, feat_normed
                                                         • f_irrel, f_rel, f_contour
                                                         • recon_irrel, recon_rel, recon_contour
                                                         • assembled_vector
```

---

## 1. 🔍 Inference Mode Outputs (`model.eval()`)

When `model.eval()` is set, invoking `model(x)` returns an **L2-normalized feature embedding vector** used for cosine distance matching and identity retrieval in surveillance galleries.

* **Input Tensor Shape**: `(B, 3, 256, 128)` (Batch Size $B$, 3 RGB channels, Height 256, Width 128)
* **Output Tensor Shape**: `(B, 512)`
* **Data Type**: `torch.FloatTensor` (L2-normalized along dimension 1, $\|\mathbf{f}\|_2 = 1.0$)

### Operational Inference Modes:

| Inference Mode | Output Formulation | Output Dimension | Description & Purpose |
| :--- | :--- | :---: | :--- |
| **`clothes_invariant`** *(Default)* | $\mathbf{f} = \text{L2\_Norm}(0.6 \cdot \hat{\mathbf{f}}_{\text{base}} + 0.4 \cdot \hat{\mathbf{f}}_{\text{irrel}})$ | `(B, 512)` | **Strips transient clothing and bag features ($F^-$)**. Retains body proportions, facial geometry, and silhouette contours ($F^+ \oplus F^t$) to ensure robust matching even when subjects change clothes or wear bags. |
| **`standard`** | $\mathbf{f} = \text{L2\_Norm}(\hat{\mathbf{f}}_{\text{base}})$ | `(B, 512)` | Standard global identity embedding from the primary BNNeck without clothes disentanglement filtering. |

---

## 2. 🏋️ Training Mode Outputs (`model.train()`)

During training, `model(x, shuffle_clothes=True/False)` returns a **dictionary containing 11 tensor outputs** required to compute the two-stage multi-task loss ($L_{ID}, L_{\text{triplet}}, L_R, L_c, L_{ca}, L_{ac}$):

```python
outputs = model(imgs, shuffle_clothes=True)
```

### Complete Tensor Output Dictionary:

| Key Name | Tensor Shape | Value Range | Branch | Loss / Mathematical Function |
| :--- | :---: | :---: | :---: | :--- |
| **`cls_score`** | `(B, num_classes)` | $(-\infty, +\infty)$ | Person ID (PI) | Identity classification logits passed to Label-Smoothed Cross-Entropy Loss ($L_{ID}$, Eq. 5–6). |
| **`clothes_score`** | `(B, num_clothes)` | $(-\infty, +\infty)$ | Clothes ID (CI) | Predicted clothes class logits from $C_P$ passed to Clothes Cross-Entropy ($L_c$) and Adversarial Loss ($L_{ca}$, Eq. 11–14). |
| **`global_feat`** | `(B, 512)` | $(-\infty, +\infty)$ | PI Branch | Raw global pooled feature vector before Batch Normalization, used for **Hard-Mining Triplet Loss** ($L_{\text{triplet}}$ with margin 0.3). |
| **`feat_normed`** | `(B, 512)` | $(-\infty, +\infty)$ | PI Branch | Feature vector after BNNeck layer, used as input for classifier $C_{ID}$. |
| **`f_irrel`** ($P^+$) | `(B, 512, 16, 8)` | $[0, +\infty)$ | Component Reconstruction (CR) | **Clothes-irrelevant feature map**: captures invariant body shape, limbs, and facial geometry. |
| **`f_rel`** ($P^-$) | `(B, 512, 16, 8)` | $[0, +\infty)$ | CR Branch | **Clothes-relevant feature map**: captures shirt, pants, colors, and bag accessory patterns. |
| **`f_contour`** ($P^t$) | `(B, 512, 16, 8)` | $[0, +\infty)$ | CR Branch | **Contour feature map**: captures silhouette boundary lines and human posture contours. |
| **`recon_irrel`** ($Y^+$) | `(B, 1, 16, 8)` | $[0.0, 1.0]$ | 4-Block Decoder $\psi^+$ | **Reconstructed body & face mask** decoded through 4 residual blocks for L1 Reconstruction Loss ($L_R$, Eq. 10). |
| **`recon_rel`** ($Y^-$) | `(B, 1, 16, 8)` | $[0.0, 1.0]$ | 4-Block Decoder $\psi^-$ | **Reconstructed apparel & accessory mask** decoded for L1 Reconstruction Loss ($L_R$, Eq. 10). |
| **`recon_contour`** ($Y^t$) | `(B, 1, 16, 8)` | $[0.0, 1.0]$ | 4-Block Decoder $\psi^t$ | **Reconstructed silhouette edge mask** decoded for L1 Reconstruction Loss ($L_R$, Eq. 10). |
| **`assembled_vector`** ($G_i$) | `(B, 1536)` | $(-\infty, +\infty)$ | DAD Module | Pooled assembled vector $G_i = F_i^+ \oplus F_{ai}^- \oplus F_i^t$ with batch-permuted clothes features, used for Assembled Clothes Loss ($L_{ac}$, Eq. 17–18). |

---

## 3. 🎨 Disentanglement Visualization Outputs

Extracted using `model.extract_disentangled_maps(x)` (implemented in `models/lightweight_reid.py` and visualized in `utils/visualization.py`):

```python
maps = model.extract_disentangled_maps(img_tensor)
```

| Output Key | Tensor Shape | Visualization Description |
| :--- | :---: | :--- |
| **`recon_irrel`** | `(B, 1, 16, 8)` | **Body Shape & Face Geometry ($Y_i^+$)**: Activates exclusively on non-clothing regions (head, face, arms, legs, anatomical frame). |
| **`recon_rel`** | `(B, 1, 16, 8)` | **Apparel & Bag Region ($Y_i^-$)**: Activates on upper-body clothes, trousers, backpacks, handbags, and shoes. |
| **`recon_contour`** | `(B, 1, 16, 8)` | **Human Contour Boundary ($Y_i^t$)**: Outlines the person's physical silhouette edge, separating foreground person from background noise. |
| **`assembled_feat`** | `(B, 512, 16, 8)` | **DAD Assembled Activation Map**: Dual attention-refined feature map (Channel + Spatial Attention) highlighting robust identity-bearing cues. |

---

## 4. 🚀 Downstream System & Pipeline Outputs

Beyond the raw neural network forward pass, the model powers several downstream modules:

### A. Re-Identification Search Engine (`utils/reid_engine.py`)
* **Method**: `rank_query(query_img_path, gallery_samples, top_k=10)`
* **Output Format**:
  ```python
  [
      {
          "rank": 1,
          "similarity": 95.37,          # Percentage cosine similarity
          "distance": 0.0463,           # Cosine distance (1 - similarity)
          "pid": 104,                   # Matched Person ID
          "camid": 0,                   # Camera Viewpoint ID
          "path": "./Data_set/both_small/bounding_box_test/0104_c1_f12.jpg"
      },
      ...
  ]
  ```

### B. Evaluation Suite (`evaluate.py`)
* **Method**: `evaluate_subset(model, data_dir, dataset_type, device)`
* **Output Metrics**:
  * **Rank-1 Accuracy (%)**: First-place true match rate.
  * **Rank-5 / Rank-10 Accuracy (%)**: Top-5 and Top-10 cumulative recall.
  * **mAP (%)**: Mean Average Precision under Market-1501 single-query protocol.
  * **Latency & Throughput**: Average single-image CPU latency in milliseconds and real-time FPS throughput.
  * **Convergence Progress (%)**: Empirical checkpoint progress relative to IEEE TCSVT 2023 paper benchmarks.

### C. Web Application & Dashboard (`app.py`)
* **Endpoint**: `/api/search`
* **Outputs**:
  * JSON payload with matched gallery person cards, similarity scores, and person IDs.
  * **4-Panel Disentanglement Heatmap**: Synthesized visual comparison showing Input Probe, Body Shape Map ($Y^+$), Apparel Map ($Y^-$), and DAD Activation Overlay.
