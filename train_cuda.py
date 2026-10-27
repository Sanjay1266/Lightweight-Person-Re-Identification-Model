"""
CUDA-Accelerated Training and Controlled Optimization for Lightweight DCR-ReID
Base Paper: "DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification"
IEEE TCSVT 2023 | Course: 23CSE373 - Computer Vision

Features:
- Hardware CUDA execution on NVIDIA RTX 4060 Laptop GPU (8GB VRAM)
- Automatic Mixed Precision (AMP) via torch.amp with GradScaler
- Identity-balanced PK Sampling (P distinct IDs, K instances per ID)
- Ground-truth PRCC clothing condition supervision (Cam A = Cam B != Cam C)
- Dual-stage optimization protocol (Stage 1: Disentanglement, Stage 2: Adversarial & Assembly)
- Warmup + Cosine Annealing learning rate schedule with decoupled weight decay parameter groups
- Validation-driven checkpoint selection (best_model.pth & final_model.pth)
- Epoch-by-epoch telemetry logging to reports/cuda_training_log.csv
"""

import os
import sys
import time
import json
import csv
import hashlib
import argparse
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

sys.path.insert(0, os.path.abspath('.'))

from data.dataset_loader import get_dataloaders, generate_clothing_labels
from models.lightweight_reid import LightweightDCRReID
from models.loss import DCRReIDCombinedLoss
from utils.metrics import compute_distance_matrix, eval_market1501


def get_sha256(file_path):
    if not file_path or not os.path.exists(file_path):
        return None
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_pseudo_component_masks(imgs):
    """
    Synthesizes ground-truth component masks directly on GPU:
    - target_contour (T_i^t): Sobel gradient contour map
    - target_irrel (T_i^+): Non-clothing anatomy (head/face & lower legs)
    - target_rel (T_i^-): Clothing/apparel geometry (torso & middle body)
    """
    b, c, h, w = imgs.shape
    device = imgs.device

    # 1. Edge detector for contour map T_i^t (Eq. 9-10)
    gray = 0.299 * imgs[:, 0:1] + 0.587 * imgs[:, 1:2] + 0.114 * imgs[:, 2:3]
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    target_contour = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    target_contour = torch.sigmoid(target_contour * 3.0)

    # 2. Structural human anatomical geometry
    y_coords = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
    target_irrel = ((y_coords < 0.22) | (y_coords > 0.85)).float()
    target_rel = ((y_coords >= 0.22) & (y_coords <= 0.85)).float()

    return target_irrel, target_rel, target_contour


def evaluate_model(model, query_loader, gallery_loader, device, max_eval_q=150, max_eval_g=600):
    model.eval()
    q_feats, q_pids, q_cams = [], [], []
    g_feats, g_pids, g_cams = [], [], []

    with torch.no_grad():
        q_count = 0
        for imgs, pids, cams, _ in query_loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            q_feats.append(feats.cpu().numpy())
            q_pids.extend(pids.numpy())
            q_cams.extend(cams.numpy())
            q_count += len(pids)
            if max_eval_q and q_count >= max_eval_q:
                break

        g_count = 0
        for imgs, pids, cams, _ in gallery_loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            g_feats.append(feats.cpu().numpy())
            g_pids.extend(pids.numpy())
            g_cams.extend(cams.numpy())
            g_count += len(pids)
            if max_eval_g and g_count >= max_eval_g:
                break

    q_feats = np.vstack(q_feats)
    g_feats = np.vstack(g_feats)
    q_pids = np.array(q_pids)
    g_pids = np.array(g_pids)
    q_cams = np.array(q_cams)
    g_cams = np.array(g_cams)

    distmat = compute_distance_matrix(q_feats, g_feats, metric='cosine')
    cmc, mAP = eval_market1501(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=10)

    rank1 = float(cmc[0] * 100) if len(cmc) > 0 else 0.0
    rank5 = float(cmc[4] * 100) if len(cmc) >= 5 else 0.0
    rank10 = float(cmc[9] * 100) if len(cmc) >= 10 else 0.0
    mAP_pct = float(mAP * 100)

    return rank1, rank5, rank10, mAP_pct


def train_cuda(
    data_dir='./Data_set',
    dataset_type='both_small',
    epochs=25,
    batch_size=32,
    lr=0.00035,
    weight_decay=5e-4,
    save_dir='./checkpoints/experiments/dcr_reid_cuda_v1',
    resume=None,
    max_batches=None,
    use_pk_sampler=True,
    num_instances=4,
    use_amp=True,
    clip_grad=5.0,
    seed=42,
    experiment_id="EXP_CUDA_V1"
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 68)
    print(f"  LIGHTWEIGHT DCR-ReID CONTROLLED CUDA TRAINING PIPELINE")
    print("=" * 68)
    print(f"[+] Execution Device: {device}")
    if device.type == 'cuda':
        print(f"[+] GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"[+] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"[+] Automatic Mixed Precision (AMP): {use_amp}")
        torch.cuda.reset_peak_memory_stats()

    # Dataloader
    dataset, train_loader, query_loader, gallery_loader = get_dataloaders(
        data_dir=data_dir, dataset_type=dataset_type, batch_size=batch_size,
        use_pk_sampler=use_pk_sampler, num_instances=num_instances
    )
    dataset.print_dataset_summary()

    num_classes = dataset.num_train_pids
    num_clothes = max(20, min(100, num_classes * 2))
    p_ids = batch_size // num_instances
    print(f"[+] Identity Classes: {num_classes} | Clothing Classes: {num_clothes}")
    if use_pk_sampler:
        print(f"[+] PK Sampler: P={p_ids} identities, K={num_instances} instances/identity (Batch Size={batch_size})")

    # Instantiate Lightweight DCR-ReID model
    model = LightweightDCRReID(num_classes=num_classes, num_clothes=num_clothes, feat_dim=512).to(device)

    starting_ckpt_hash = None
    if resume and os.path.exists(resume):
        starting_ckpt_hash = get_sha256(resume)
        state_dict = torch.load(resume, map_location=device)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"[+] Loaded {len(filtered_dict)}/{len(model_dict)} tensors from checkpoint: '{resume}'")
        print(f"    SHA-256: {starting_ckpt_hash}")

    criterion = DCRReIDCombinedLoss(num_classes=num_classes, num_clothes=num_clothes, margin=0.3, epsilon=0.1).to(device)

    # Decoupled parameter groups: decay weights, do not decay bias or batchnorm weights
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'bn' in name or 'bottleneck' in name or 'downsample.1' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.Adam([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr)

    # Learning rate schedule: Warmup (5 epochs) + Cosine Annealing
    warmup_epochs = min(5, max(1, epochs // 5))
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == 'cuda'))

    stage_switch_epoch = max(1, epochs // 3)
    best_rank1 = 0.0
    best_mAP = 0.0
    best_epoch = 0

    log_csv = 'reports/cuda_training_log.csv'
    csv_exists = os.path.exists(log_csv)

    print("\n--- Starting Controlled CUDA Training ---")
    training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        if epoch <= stage_switch_epoch:
            criterion.stage = 1
            stage_name = "Stage 1 (Disentangle Init)"
        else:
            criterion.stage = 2
            stage_name = "Stage 2 (Full DCR & Adversarial)"

        model.train()
        running_loss = 0.0
        running_id = 0.0
        running_trip = 0.0
        running_recon = 0.0
        running_clothes = 0.0
        batches_processed = 0
        total_images_processed = 0

        for batch_idx, (imgs, pids, cams, _) in enumerate(train_loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            imgs = imgs.to(device, non_blocking=True)
            pids = pids.to(device, non_blocking=True)
            cams = cams.to(device, non_blocking=True)

            clothes_labels = generate_clothing_labels(pids, cams, is_prcc=dataset.is_prcc, num_clothes=num_clothes)
            target_irrel, target_rel, target_contour = generate_pseudo_component_masks(imgs)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                outputs = model(imgs, shuffle_clothes=(criterion.stage == 2))
                loss_dict = criterion(
                    outputs, pids, clothes_labels=clothes_labels,
                    target_irrel=target_irrel, target_rel=target_rel, target_contour=target_contour
                )
                total_loss = loss_dict['total_loss']

            scaler.scale(total_loss).backward()

            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()
            running_id += loss_dict['loss_id'].item()
            running_trip += loss_dict['loss_triplet'].item()
            running_recon += loss_dict['loss_recon'].item()
            running_clothes += loss_dict['loss_clothes'].item()
            batches_processed += 1
            total_images_processed += imgs.size(0)

        scheduler.step()

        n_batches = max(1, batches_processed)
        avg_loss = running_loss / n_batches
        avg_id = running_id / n_batches
        avg_trip = running_trip / n_batches
        avg_recon = running_recon / n_batches
        avg_clothes = running_clothes / n_batches
        epoch_duration = time.time() - epoch_start_time
        img_per_sec = total_images_processed / max(0.01, epoch_duration)
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0.0

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{epochs:02d}] {stage_name} | Loss: {avg_loss:.4f} (ID: {avg_id:.3f}, Trip: {avg_trip:.3f}, Recon: {avg_recon:.4f}, Clothes: {avg_clothes:.3f}) | Speed: {img_per_sec:.1f} img/s | VRAM: {peak_vram_mb:.1f} MB | LR: {current_lr:.6f}")

        # Validation after every epoch
        rank1, rank5, rank10, mAP = evaluate_model(model, query_loader, gallery_loader, device)
        print(f"   => Val Epoch {epoch:02d} | Rank-1: {rank1:.2f}% | Rank-5: {rank5:.2f}% | Rank-10: {rank10:.2f}% | mAP: {mAP:.2f}%")

        is_best = False
        if rank1 > best_rank1 or (rank1 == best_rank1 and mAP > best_mAP):
            best_rank1 = rank1
            best_mAP = mAP
            best_epoch = epoch
            is_best = True
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"   [+] New best validation checkpoint saved to '{best_model_path}'")

        # Log to reports/cuda_training_log.csv
        with open(log_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists and epoch == 1 and f.tell() == 0:
                writer.writerow([
                    'experiment_id', 'epoch', 'stage', 'total_loss', 'loss_id',
                    'loss_triplet', 'loss_recon', 'loss_clothes', 'rank1', 'rank5',
                    'rank10', 'mAP', 'lr', 'img_per_sec', 'epoch_duration_sec',
                    'peak_vram_mb', 'device', 'starting_checkpoint_hash', 'dataset_type'
                ])
                csv_exists = True
            writer.writerow([
                experiment_id, epoch, criterion.stage, round(avg_loss, 4),
                round(avg_id, 4), round(avg_trip, 4), round(avg_recon, 4),
                round(avg_clothes, 4), round(rank1, 2), round(rank5, 2),
                round(rank10, 2), round(mAP, 2), round(current_lr, 6),
                round(img_per_sec, 1), round(epoch_duration, 2), round(peak_vram_mb, 1),
                str(device), starting_ckpt_hash or "Scratch", dataset_type
            ])

    total_training_time = time.time() - training_start_time
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)

    print("\n" + "=" * 68)
    print(f"  TRAINING COMPLETED: {total_training_time/60:.2f} min ({total_training_time:.1f}s)")
    print(f"  Best Checkpoint: Epoch {best_epoch:02d} | Rank-1: {best_rank1:.2f}% | mAP: {best_mAP:.2f}%")
    print(f"  Best Model Saved: {os.path.join(save_dir, 'best_model.pth')}")
    print(f"  Final Model Saved: {final_model_path}")
    print("=" * 68)

    # Save experiment config
    config_dict = {
        'experiment_id': experiment_id,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'device': str(device),
        'gpu_model': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A',
        'dataset_type': dataset_type,
        'data_dir': data_dir,
        'epochs': epochs,
        'batch_size': batch_size,
        'p_ids': p_ids,
        'k_instances': num_instances,
        'base_lr': lr,
        'weight_decay': weight_decay,
        'use_amp': use_amp,
        'clip_grad': clip_grad,
        'seed': seed,
        'starting_checkpoint': resume,
        'starting_checkpoint_hash': starting_ckpt_hash,
        'save_dir': save_dir,
        'best_epoch': best_epoch,
        'best_rank1': best_rank1,
        'best_mAP': best_mAP,
        'total_time_seconds': round(total_training_time, 2)
    }
    with open('reports/cuda_experiment_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    return config_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Controlled CUDA Training for Lightweight DCR-ReID")
    parser.add_argument('--data_dir', type=str, default='./Data_set')
    parser.add_argument('--dataset_type', type=str, default='both_small',
                        choices=['both_small', 'with_bag', 'without_bag', 'both_large', 'prcc', 'prcc_cc', 'prcc_sc', 'prcc_all'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/experiments/dcr_reid_cuda_v1')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment_id', type=str, default='EXP_CUDA_V1')

    args = parser.parse_args()

    if args.smoke_test:
        print("[!] Running GPU Smoke Test (1 epoch, 5 batches)...")
        args.epochs = 1
        args.max_batches = 5
        args.experiment_id = 'CUDA_SMOKE_TEST'

    train_cuda(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        resume=args.resume,
        max_batches=args.max_batches,
        use_amp=not args.no_amp,
        seed=args.seed,
        experiment_id=args.experiment_id
    )
