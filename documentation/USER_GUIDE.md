# Lightweight DCR-ReID: User & Execution Guide
### Course: 23CSE373 - Computer Vision | Base Paper: IEEE TCSVT 2023

This guide details how to install dependencies, run benchmark evaluations, perform CLI query searches, train the model, and launch the interactive web dashboard.

---

## 🛠️ 1. Environment Setup

### Prerequisites
- Python 3.10+
- PyTorch & Torchvision
- OpenCV, Pillow, Matplotlib, Scikit-Learn, Flask

```bash
pip install torch torchvision opencv-python pillow matplotlib seaborn scikit-learn flask pypdf
```

---

## 📊 2. Run Comprehensive Benchmark Evaluation

### Fast Mode (Sampled Queries & Gallery)
Runs in ~30 seconds, providing immediate feedback on all 4 dataset splits:
```bash
python evaluate.py --fast
```

### Full Mode (Exhaustive Evaluation Across All Images)
Evaluates every query against every gallery image in the dataset:
```bash
python evaluate.py --full
```

### Custom Evaluation Options:
```bash
python evaluate.py --data_dir ./Data_set --model_path ./checkpoints/best_model.pth
```

**Expected Terminal Output:**
- Device & latency specs (CPU/GPU latency in ms, FPS throughput).
- Section 1: Empirical Model Checkpoint Evaluation (Rank-1, Rank-5, Rank-10, mAP, convergence progress bar).
- Section 2: DCR-ReID Paper Target Benchmarks.

---

## 🔍 3. Run Command-Line Query Search (demo_cli.py)

Search the surveillance gallery for matching identities using a query image:

```bash
# Auto-selects first available query image from both_small
python demo_cli.py

# Custom query image with Top-10 ranked results
python demo_cli.py --query ./Data_set/both_small/query/0100_c1_f421.jpg --top_k 10 --dataset_type both_small
```

**Output Format:**
```text
Rank   | Similarity   | Distance   | Person ID  | Camera ID  | Gallery Image Path
------------------------------------------------------------------------------------------
#1     | 95.37      % | 0.0463     | 104        | 0          | ./Data_set\both_small\bounding_box_test\0104_c1_f12.jpg
#2     | 95.28      % | 0.0472     | 114        | 0          | ./Data_set\both_small\bounding_box_test\0114_c1_f336.jpg
...
```

---

## 🌐 4. Run Interactive Web Application (app.py)

Launch the Flask dark-mode glassmorphism dashboard:

```bash
python app.py
```

Open your browser at: **`http://127.0.0.1:5000`**

### Dashboard Capabilities:
1. **🔍 Query Search Tab**:
   - Choose dataset subset (`both_small`, `with_bag`, `without_bag`, `both_large`).
   - Pick from pre-loaded sample thumbnails or upload a custom person probe.
   - Adjust Top-K retrieval rank (5, 10, 15, 20).
   - Click **Run Lightweight DCR-ReID Matching**.
   - Displays matched gallery identity cards with percentage similarity.
   - Generates the **4-Panel Component Disentanglement Visualizer**:
     - Panel 1: Input Query Probe
     - Panel 2: Body Shape & Face Landmark Map ($Y^+$)
     - Panel 3: Clothing & Accessory Map ($Y^-$)
     - Panel 4: DAD Assembled Identity Feature Activation Overlay
2. **🧩 DCR-ReID Architecture Tab**:
   - Explores the 5-step workflow.
   - Reviews the mathematical equations box (Eq. 8, 10, 16, 20).
   - Features comparison table (clothes-irrelevant vs clothes-relevant).
3. **📊 Benchmark & Convergence Tab**:
   - Training Convergence Notice Banner.
   - Live progress bars for each dataset subset showing checkpoint progress towards target.
   - Summary cards displaying Target Rank-1, Rank-5, Rank-10, and mAP.
4. **⚡ Edge Efficiency Tab**:
   - Direct comparison against standard ResNet-50 baseline.
   - Highlights 87.3% parameter reduction and 3.8x CPU inference speedup.

---

## 🏋️ 5. Train the Model (train.py)

Train the lightweight DCR-ReID model using the two-stage optimization protocol:

```bash
# Train on small combined benchmark
python train.py --dataset_type both_small --epochs 15 --batch_size 16 --lr 0.0003

# Train on person with bag subset
python train.py --dataset_type with_bag --epochs 20 --batch_size 16

# Train on person without bag subset
python train.py --dataset_type without_bag --epochs 20 --batch_size 16
```

### Training Highlights:
- **Stage 1 (Epochs 1 to 5)**: Optimizes $L_{ID} + L_{\text{triplet}} + L_c + L_R$ to initialize the clothes classifier and component decoders.
- **Stage 2 (Epochs 6 to 15)**: Optimizes the full loss with batch clothes feature assembly ($L_{ac}$) and clothes adversarial smoothing ($L_{ca}$).
- Best weights are automatically saved to `./checkpoints/best_model.pth`.
- Final weights saved to `./checkpoints/final_model.pth`.
