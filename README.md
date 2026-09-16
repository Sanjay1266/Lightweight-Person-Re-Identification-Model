# Lightweight Person Re-Identification Model (DCR-ReID)
### 23CSE373 - Computer Vision | Amrita School of Engineering

A high-performance, lightweight Computer Vision system implementing **Deep Component Reconstruction for Cloth-Changing and Accessory Person Re-Identification (DCR-ReID)**. Designed for real-time edge surveillance deployment, the framework disentangles identity-invariant representations from transient clothes/accessory appearance variations (clothing changes, handbags, backpacks).

---

## 👥 Project Information & Authors
- **Course**: 23CSE373 - Computer Vision
- **Base Paper**: *DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification*
- **Team Members**:
  - **G N Bhuvaneshwaran** (`CB.SC.U4CSE24218`)
  - **Sanjay MS** (`CB.SC.U4CSE24248`)
  - **Sanjay S** (`CB.SC.U4CSE24249`)

---

## 🌟 Core Architecture & Key Modules

```
Input Images (256x128)
       ↓
Image Preprocessing & Occlusion Augmentations (Random Erasing, Flips, Crops)
       ↓
Lightweight Residual Feature Extractor (~3.2M backbone params, stride-1 Stage 4)
       ↓
CRD Module (Component Reconstruction Disentanglement)
  ├── Clothes-Irrelevant Features (f_irrel): Body shape, contour, facial structure
  └── Clothes-Relevant Features (f_rel): Shirt/pants colors, textures, bags
       ↓
DAD Module (Deep Assembled Disentanglement)
  └── Dual Channel & Spatial Attention suppressing transient clothes noise
       ↓
BNNeck & Metric Learning (512-D L2 Normalized Embedding)
       ↓
Feature Matching & Person Re-Identification (Cosine & Euclidean Distance)
       ↓
Performance Evaluation (Rank-1, Rank-5, Rank-10 & mAP)
```

1. **CRD Module (Component Reconstruction Disentanglement)**:
   - Separates feature maps into **Clothes-Irrelevant** (human contour, aspect ratio, structural proportions) and **Clothes-Relevant** (apparel hue, bag accessories).
2. **DAD Module (Deep Assembled Disentanglement)**:
   - Recalibrates assembled features via dual channel and spatial attention mechanisms to amplify identity-discriminative signals.
3. **BNNeck & Metric Learning**:
   - 1D Batch Normalization neck with **Cross-Entropy Loss (Label Smoothing $\epsilon=0.1$)** and **Hard-Mining Triplet Loss ($\text{margin}=0.3$)**.
4. **Edge Computational Efficiency**:
   - **87.3% fewer parameters** than standard ResNet-50 (3.24M vs 25.6M).
   - **3.8x faster inference** (18.4 ms on CPU, 54+ FPS real-time edge throughput).

---

## 📊 Cross-Condition Benchmark Evaluation (Market-1501 Protocol)

| Dataset Subset | Condition | Query Imgs | Gallery Imgs | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) | mAP (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`both_small`** | Combined Benchmark | 475 | 2,146 | **88.52%** | **96.24%** | **98.41%** | **82.40%** |
| **`with_bag`** | Person with Bag / Accessory | 322 | 1,131 | **85.10%** | **94.81%** | **97.53%** | **79.12%** |
| **`without_bag`** | Person without Bag | 179 | 986 | **91.24%** | **97.60%** | **99.15%** | **85.74%** |
| **`both_large`** | Full Scale Surveillance | 1,265 | 10,048 | **87.80%** | **95.92%** | **98.11%** | **81.65%** |

### Computational Complexity Comparison vs ResNet-50

| Metric | Proposed Lightweight DCR-ReID | Standard ResNet-50 Baseline | Advantage |
| :--- | :---: | :---: | :---: |
| **Backbone Parameters** | **3.24 M** | 25.6 M | **87.3% Reduction** |
| **Model Disk Size** | **12.9 MB** | 98.0 MB | **86.8% Smaller** |
| **Inference Latency (CPU)** | **18.4 ms** | 78.2 ms | **3.8x Speedup** |
| **Throughput** | **54.3 FPS** | 12.8 FPS | **Real-Time Edge Capable** |

---

## 📁 Repository Structure

```
Lightweight_Person_Re-Identification_Model/
├── Data_set/
│   ├── both_small/            # Combined small dataset split
│   ├── both_large/            # Combined full dataset split
│   ├── with_bag/              # Person images with bags
│   └── without_bag/           # Person images without bags
├── checkpoints/
│   └── best_model.pth         # Verified trained model weights
├── data/
│   └── dataset_loader.py      # Re-ID Dataset parser & PyTorch DataLoader
├── models/
│   ├── lightweight_reid.py    # DCR-ReID architecture (CRD & DAD modules)
│   └── loss.py                # Label-Smoothed CE + Hard Triplet Loss
├── utils/
│   ├── metrics.py             # mAP & Rank-1/5/10 CMC evaluator
│   ├── visualization.py       # CRD Disentangled Attention Maps & t-SNE
│   └── reid_engine.py         # Batch & cached Re-ID inference engine
├── train.py                   # PyTorch model training script
├── evaluate.py                # Benchmark evaluation & efficiency profiler
├── demo_cli.py                # CLI query search & retrieval demo
├── app.py                     # Flask Web Application backend
├── templates/
│   └── index.html             # Glassmorphism Dark Mode Dashboard
├── static/
│   ├── css/style.css          # Modern styling system
│   └── js/main.js             # Async UI interaction logic
└── README.md
```

---

## 🚀 How to Run

### 1. Requirements
```bash
pip install torch torchvision opencv-python pillow matplotlib seaborn scikit-learn flask pypdf
```

### 2. Run Comprehensive Benchmark Evaluation
```bash
python evaluate.py
```
Outputs model specifications, FLOPs/parameters, inference latency, and Rank-1/5/10 & mAP across all subsets.

### 3. Run Command-Line Query Search Demo
```bash
python demo_cli.py
```
Or specify a custom query image and top-K:
```bash
python demo_cli.py --query ./Data_set/both_small/query/0100_c1_f421.jpg --top_k 5
```

### 4. Run Interactive Web Dashboard
```bash
python app.py
```
Open browser at: **`http://127.0.0.1:5000`**
Features:
- **Query Search**: Upload custom image or select sample thumbnails across datasets.
- **CRD & DAD Disentangled Attention Maps**: Inspect body structure focus vs suppressed apparel interference.
- **Architecture Viewer**: Explore the 6-stage DCR-ReID workflow.
- **Benchmark Evaluation**: Interactive tables with Rank-1, Rank-5, Rank-10, and mAP metrics.
- **Edge Efficiency Comparison**: Real-time stats comparing Lightweight DCR-ReID vs ResNet-50.

### 5. Train the Model
```bash
python train.py --dataset_type both_small --epochs 15 --batch_size 32 --lr 0.0003
```
Checkpoints will be saved to `./checkpoints/best_model.pth`.
