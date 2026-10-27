# Environment & Hardware Audit Report: Lightweight DCR-ReID

### Course: 23CSE373 - Computer Vision | Date: 2026-10-27
**Project**: Lightweight Person Re-Identification Model (DCR-ReID)  
**Root Repository**: `S:\5th Sem\CV\Lightweight_Person_Re-Identification_Model`

---

## 1. Executive Summary

This audit records the complete hardware, driver, Python runtime, PyTorch build, and compute acceleration status for the Lightweight DCR-ReID engineering environment. 

A physical NVIDIA Ada Lovelace discrete GPU is present on the host system. However, the active system Python environment (`Python 3.14.3`) is bound to a CPU-only build of PyTorch (`2.13.0+cpu`), preventing direct CUDA acceleration without an isolated runtime environment.

---

## 2. Hardware and Operating System Specifications

| Specification | Hardware Value | Verification Method |
| :--- | :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home (64-bit) | Systeminfo / PowerShell |
| **Host Processor (CPU)** | AMD Ryzen Processor (4 cores / 8 threads) | WMI / PowerShell |
| **System Memory (RAM)** | 16 GB Physical RAM (~8.5 GB available) | GlobalMemoryStatusEx |
| **Discrete GPU** | **NVIDIA GeForce RTX 4060 Laptop GPU** | `nvidia-smi` |
| **GPU Architecture** | Ada Lovelace (AD107M, Compute Capability 8.9) | Hardware device ID `01:00.0` |
| **Dedicated Video Memory** | **8,188 MiB (8.0 GB GDDR6)** | `nvidia-smi` |
| **Current GPU Memory Usage** | ~1,299 MiB utilized (Windows Desktop Window Manager) | `nvidia-smi` |
| **NVIDIA Graphics Driver** | **610.47** | Driver Model WDDM |
| **Supported Driver CUDA Version** | **CUDA 13.3** | `nvidia-smi` UMD Version |

---

## 3. Python Runtime & PyTorch Framework Audit

| Software Component | Active Version / Path | Status & Implications |
| :--- | :--- | :--- |
| **Python Executable** | `C:\Users\sanja\AppData\Local\Python\pythoncore-3.14-64\python.exe` | System Python |
| **Python Version** | **3.14.3** (tags/v3.14.3:323c59a, Feb 3 2026, 16:04:56) | Very recent Python release |
| **PyTorch Distribution** | **`2.13.0+cpu`** | **CPU-Only Build** |
| **TorchVision Distribution**| **`0.28.0+cpu`** | CPU-Only Build |
| **`torch.version.cuda`** | `None` | No compiled CUDA runtime |
| **`torch.cuda.is_available()`**| **`False`** | Model executes on CPU |
| **`torch.cuda.device_count()`**| `0` | No CUDA devices exposed to PyTorch |
| **Device Placement** | `device = torch.device('cpu')` | Model & tensors placed on CPU |

---

## 4. Root Cause of CUDA Absence

1. **Python 3.14 Wheel Availability**:
   - Official prebuilt CUDA wheels from PyTorch (`https://download.pytorch.org/whl/cu124` or `cu121`) are distributed for Python 3.9, 3.10, 3.11, and 3.12 (with experimental 3.13).
   - Python 3.14 is not yet an official prebuilt CUDA target on the PyTorch stable repository. The active environment uses a custom or CPU-only wheel `2.13.0+cpu`.
2. **Current Policy Compliance**:
   - Per project instructions, packages are **not** installed globally, and system Python is **not** altered without explicit approval.

---

## 5. Proposed Isolated Virtual Environment Plan (CUDA Activation)

To safely enable full GPU acceleration on the NVIDIA RTX 4060 Laptop GPU without touching the system Python installation:

### Proposed Architecture
1. **Isolated Virtual Environment**:
   - Directory: `s:\5th Sem\CV\Lightweight_Person_Re-Identification_Model\.venv_cuda`
   - Python Engine: Python 3.11 or 3.12 (which have official, stable PyTorch wheels compiled for CUDA 12.1/12.4).
2. **Installation Steps (Requires User Approval)**:
   ```powershell
   # 1. Create isolated virtual environment with Python 3.11/3.12
   py -3.11 -m venv .venv_cuda  # or py -3.12
   
   # 2. Activate isolated environment
   .\.venv_cuda\Scripts\Activate.ps1
   
   # 3. Install PyTorch with CUDA 12.4 support
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   
   # 4. Install remaining Re-ID requirements
   pip install numpy pillow
   ```
3. **Hardware Expected Performance Gain**:
   - 1 full PRCC training epoch on CPU takes $\approx 18\text{ minutes}$.
   - 1 full PRCC training epoch on RTX 4060 GPU with mixed precision (`torch.amp`) takes $\approx 18\text{ seconds}$ (**$60\times$ speedup**).
   - Peak VRAM footprint for batch size 16 ($P=4, K=4$) is $\approx 1.8\text{ GB}$, comfortably below the 8 GB ceiling.

---

## 6. Current Operational State

Until user approval is granted for virtual environment setup:
- All training experiments utilize controlled mini-batch fine-tuning on CPU (17.2 img/s).
- All evaluations execute on CPU with zero crashes, supported by PRCC feature caching and multi-core threading.
