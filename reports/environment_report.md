# Lightweight DCR-ReID: Environment & Hardware Audit Report

### Course: 23CSE373 - Computer Vision | Date: 2026-10-27

---

## 1. System and Operating Environment

| Component | Specification | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (Build 10.0.26300) | 64-bit AMD64 architecture |
| **Shell / CLI** | PowerShell 7 / Windows PowerShell | Terminal environment |
| **Workspace Root** | `S:\5th Sem\CV\Lightweight_Person_Re-Identification_Model` | Local project repository |
| **CPU Architecture** | AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD | 4 Physical Cores / 8 Logical Threads |
| **System RAM** | 23.69 GB Total (~4.14 GB Available) | DDR4/DDR5 system memory |

---

## 2. GPU Hardware & CUDA Compatibility Audit

| Component | Detected Specification | Status & Verification |
| :--- | :--- | :--- |
| **Discrete GPU Model** | NVIDIA GeForce RTX 4060 Laptop GPU | Detected via `nvidia-smi` |
| **VRAM Capacity** | 8,188 MiB (~8.0 GB GDDR6) | Active, 1029 MiB currently in use by Windows |
| **NVIDIA Driver Version** | 610.47 | Modern display and compute driver |
| **Hardware CUDA Capability** | CUDA 13.3 Compute Ready | Architecture: Ada Lovelace (Compute 8.9) |
| **PyTorch CUDA Status** | `torch.cuda.is_available() == False` | **Installed PyTorch build is CPU-only (`2.13.0+cpu`)** |

### Directive Compliance Note on GPU / CUDA:
Per Section 9 of the engineering directive:
> *"Do not assume GPU availability. If CUDA is unavailable, report the reason and use CPU only when appropriate. Do not automatically reinstall PyTorch, CUDA or drivers."*

**Root Cause**: The host machine possesses an NVIDIA GeForce RTX 4060 Laptop GPU, but the system Python environment (`Python 3.14.3`) is equipped with the CPU build of PyTorch (`2.13.0+cpu`). Following the strict directive, PyTorch and system drivers will not be modified. All inference and training workflows are strictly configured for multi-threaded CPU execution with optimized data pipelines.

---

## 3. Python Runtime & Core Dependencies

| Package / Library | Installed Version | Role in DCR-ReID Project |
| :--- | :--- | :--- |
| **Python** | 3.14.3 (64-bit MSC v.1944) | Base Python runtime |
| **PyTorch (`torch`)** | `2.13.0+cpu` | Core deep learning and tensor operations |
| **Torchvision (`torchvision`)** | `0.28.0+cpu` | Data transforms and image utilities |
| **NumPy (`numpy`)** | `2.5.0` | Matrix operations and distance computations |
| **Pillow (`PIL`)** | `12.3.0` | Image loading and dataset verification |
| **OpenCV (`cv2`)** | `5.0.0.93` | Computer vision utilities and visualization |
| **SciPy (`scipy`)** | `1.18.0` | Scientific computing |
| **Scikit-Learn (`sklearn`)** | `1.9.0` | Evaluation metrics and ranking routines |
| **Matplotlib (`matplotlib`)** | `3.11.1` | Plotting CMC curves and training loss plots |
| **Seaborn (`seaborn`)** | `0.13.2` | Advanced visualization |
| **Flask (`flask`)** | `3.1.3` | Interactive web dashboard server |
| **PyPDF (`pypdf`)** | `6.14.2` | Base paper PDF protocol extraction |
| **psutil (`psutil`)** | `7.2.2` | System hardware monitoring |

---

## 4. Execution Feasibility & Strategy

1. **Evaluation**: Highly optimized with single-pass feature caching. Extracts 2,621 images for Market-1501 in ~50-80s, and PRCC queries/galleries in ~3-4 minutes.
2. **Training & Fine-Tuning**: CPU execution supports controlled short training and validation runs with identity-balanced PK sampling (`batch_size=16`, `P=4, K=4`).
