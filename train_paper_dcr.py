"""
Paper-Faithful DCR-ReID Training Pipeline (Track A)
Reference: Zhenyu Cui, Jiahuan Zhou, Yuxin Peng, Shiliang Zhang, Yaowei Wang,
"DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification",
IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol. 33, No. 8, August 2023.

Methodology Specification:
- Architecture: ResNet-50 backbone (ImageNet-pretrained, layer4 stride-1)
- CRD Module: Channels P_i^+ (1024), P_i^- (512), P_i^t (1024) [2560 total]
- Component Decoders: 4-block residual decoders ψ^+, ψ^-, ψ^t with Tanh activation
- Pooling: Stacked max-pooling and average-pooling φ(P_i^v)
- DAD Module: Feature assembly G_i = F_i^+ ⊕ F_ai^- ⊕ F_i^t (5120-D)
- Loss Objectives:
  * Stage 1: L_stage1 = L_c + L_ID + L_R
  * Stage 2: L_stage2 = L_ID + L_c + L_ca + α * L_ac + γ * L'_ID + L_R
    (PRCC default: α = 0.05, γ = 1.0)
- Optimizer: Adam (lr = 3.5e-4, weight_decay = 5e-4)
- Learning Rate Schedule: Decay by 0.1 every 20 epochs
- Input Resolution: 384 x 192
- Hardware Acceleration: CUDA on NVIDIA RTX 4060 Laptop GPU with AMP
- Checkpoint Policy: Preserves baseline checkpoints/best_model.pth strictly;
  saves to checkpoints/paper_dcr_best.pth and checkpoints/paper_dcr_latest.pth.
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
from torch.optim.lr_scheduler import MultiStepLR

sys.path.insert(0, os.path.abspath('.'))

from data.dataset_loader import get_dataloaders, generate_clothing_labels
from models.paper_dcr_reid import PaperDCRReID
from models.paper_loss import PaperDCRReIDCombinedLoss
from utils.metrics import compute_distance_matrix, eval_market1501


def get_sha256(file_path):
    if not file_path or not os.path.exists(file_path):
        return None
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_paper_component_masks(imgs):
    """
    Synthesizes binary ground-truth component masks T_i^v directly on GPU (Eq. 9-10):
    - T_i^t (Contour): Sobel edge gradient map
    - T_i^+ (Clothes-irrelevant): Anatomical non-clothing body regions (head, face, feet/shoes)
    - T_i^- (Clothes-relevant): Apparel geometry (torso, middle body)
    Values returned in range [-1, 1] matching Tanh decoder activations.
    """
    b, _, h, w = imgs.shape
    device = imgs.device

    # 1. Edge detector for contour map T_i^t
    gray = 0.299 * imgs[:, 0:1] + 0.587 * imgs[:, 1:2] + 0.114 * imgs[:, 2:3]
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    target_contour = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    target_contour = torch.sigmoid(target_contour * 3.0) * 2.0 - 1.0

    # 2. Structural anatomical geometry
    y_coords = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
    mask_irrel = ((y_coords < 0.22) | (y_coords > 0.85)).float()
    mask_rel = ((y_coords >= 0.22) & (y_coords <= 0.85)).float()

    target_irrel = mask_irrel * 2.0 - 1.0
    target_rel = mask_rel * 2.0 - 1.0

    return target_irrel, target_rel, target_contour


def evaluate_paper_model(model, query_loader, gallery_loader, device):
    """
    Evaluates PaperDCRReID on query and gallery sets using official CMC and mAP metrics.
    Inference removes F_i^- and uses L2-normalized F_i^+ ⊕ F_i^t (4096 dimensions).
    """
    model.eval()
    q_feats, q_pids, q_cams = [], [], []
    g_feats, g_pids, g_cams = [], [], []

    with torch.no_grad():
        for batch in query_loader:
            imgs, pids, cams = batch[0], batch[1], batch[2]
            imgs = imgs.to(device, non_blocking=True)
            feats = model(imgs)
            q_feats.append(feats.cpu().numpy())
            q_pids.append(pids.numpy())
            q_cams.append(cams.numpy())

        for batch in gallery_loader:
            imgs, pids, cams = batch[0], batch[1], batch[2]
            imgs = imgs.to(device, non_blocking=True)
            feats = model(imgs)
            g_feats.append(feats.cpu().numpy())
            g_pids.append(pids.numpy())
            g_cams.append(cams.numpy())

    q_feats = np.vstack(q_feats)
    q_pids = np.concatenate(q_pids)
    q_cams = np.concatenate(q_cams)

    g_feats = np.vstack(g_feats)
    g_pids = np.concatenate(g_pids)
    g_cams = np.concatenate(g_cams)

    distmat = compute_distance_matrix(q_feats, g_feats, metric='cosine')
    cmc, m_ap = eval_market1501(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=50)

    rank1 = cmc[0] * 100.0 if len(cmc) > 0 else 0.0
    rank5 = cmc[4] * 100.0 if len(cmc) > 4 else 0.0
    rank10 = cmc[9] * 100.0 if len(cmc) > 9 else 0.0
    map_pct = m_ap * 100.0

    return rank1, rank5, rank10, map_pct


def train_paper_dcr(
    data_dir='./Data_set',
    dataset_type='prcc_cc',
    epochs=60,
    stage1_epochs=10,
    batch_size=32,
    lr=3.5e-4,
    weight_decay=5e-4,
    alpha=0.05,
    gamma=1.0,
    img_height=384,
    img_width=192,
    num_classes=150,
    num_clothes=300,
    num_instances=4,
    accum_steps=1,
    save_dir='./checkpoints',
    log_dir='./reports',
    eval_freq=5,
    device_name='cuda',
    seed=42
):
    # 1. Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"\n==========================================================================================")
    print(f"  PAPER-FAITHFUL DCR-ReID TRAINING PIPELINE (TRACK A)")
    print(f"==========================================================================================")
    print(f"[*] Architecture               : ResNet-50 (Pretrained, layer4 stride-1) + CRD (2560-D) + DAD")
    print(f"[*] Input Resolution           : {img_height} x {img_width}")
    print(f"[*] Resolved Device            : {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"[*] Dataset Partition          : {dataset_type}")
    print(f"[*] Identities / Clothes       : {num_classes} / {num_clothes}")
    print(f"[*] Optimization Protocol      : Two-Stage (Stage 1: 1-{stage1_epochs}, Stage 2: {stage1_epochs+1}-{epochs})")
    print(f"[*] Hyperparameters            : lr={lr}, weight_decay={weight_decay}, alpha={alpha}, gamma={gamma}")
    print(f"[*] Batch Size / Eval Freq     : {batch_size} / Every {eval_freq} epochs")
    print(f"==========================================================================================\n")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Dataloaders at 384x192
    dataset, train_loader, query_loader, gallery_loader = get_dataloaders(
        data_dir=data_dir,
        dataset_type=dataset_type,
        batch_size=batch_size,
        num_workers=0,
        use_pk_sampler=True,
        num_instances=num_instances,
        height=img_height,
        width=img_width
    )
    print(f"=> Loaded {len(dataset.train_samples)} training samples ({len(train_loader)} batches/epoch)")
    print(f"=> Evaluation gallery: {len(dataset.gallery_samples)} samples, query: {len(dataset.query_samples)} samples\n")

    # Build intra-identity clothing map
    pid_to_clothes_map = {}
    for _, pid, cam in dataset.train_samples:
        cond = 1 if cam == 2 else 0
        cid = (pid * 2 + cond) if num_clothes >= 300 else ((pid * 2 + cond) % num_clothes)
        if pid not in pid_to_clothes_map:
            pid_to_clothes_map[pid] = set()
        pid_to_clothes_map[pid].add(cid)
    pid_to_clothes_map = {k: list(v) for k, v in pid_to_clothes_map.items()}

    # 3. Model & Loss Formulation
    model = PaperDCRReID(num_classes=num_classes, num_clothes=num_clothes, pretrained=True).to(device)
    loss_fn = PaperDCRReIDCombinedLoss(
        num_classes=num_classes,
        num_clothes=num_clothes,
        alpha=alpha,
        gamma=gamma,
        epsilon=0.1
    ).to(device)

    # 4. Optimizer & MultiStep LR Decay
    # Paper Section IV.B: "We use the Adam optimizer to train our model... The learning rate is set to 3.5e-4
    # and is decayed by 0.1 every 20 epochs."
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # CSV Telemetry Log
    log_csv_path = os.path.join(log_dir, 'paper_training_log.csv')
    csv_file = open(log_csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch', 'stage', 'lr', 'loss_total', 'loss_id', 'loss_clothes',
        'loss_ca', 'loss_ac', 'loss_id_prime', 'loss_recon',
        'rank1', 'rank5', 'rank10', 'mAP', 'epoch_time_s'
    ])

    best_rank1 = 0.0
    best_map = 0.0
    best_epoch = 0

    best_ckpt_path = os.path.join(save_dir, 'paper_dcr_best.pth')
    latest_ckpt_path = os.path.join(save_dir, 'paper_dcr_latest.pth')

    print(f"{'Epoch':<6} | {'Stage':<6} | {'Total Loss':<11} | {'L_ID':<8} | {'L_c':<8} | {'L_ca':<8} | {'L_ac':<8} | {'L_R':<8} | {'Rank-1 (%)':<10} | {'mAP (%)':<8} | {'Time'}")
    print("-" * 105)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        stage = 1 if epoch <= stage1_epochs else 2
        model.train()

        running_loss = 0.0
        running_id = 0.0
        running_c = 0.0
        running_ca = 0.0
        running_ac = 0.0
        running_id_prime = 0.0
        running_r = 0.0

        optimizer.zero_grad()
        for batch_idx, (imgs, pids, cams, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            pids = pids.to(device, non_blocking=True)
            cams = cams.to(device, non_blocking=True)

            clothes_labels = generate_clothing_labels(pids, cams, is_prcc=dataset.is_prcc, num_clothes=num_clothes).to(device)
            target_plus, target_minus, target_contour = generate_paper_component_masks(imgs)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(imgs, pids=pids, shuffle_clothes=(stage == 2))
                loss_dict = loss_fn(
                    outputs, pids, clothes_labels,
                    target_plus=target_plus,
                    target_minus=target_minus,
                    target_contour=target_contour,
                    pid_to_clothes_map=pid_to_clothes_map,
                    stage=stage
                )
                loss = loss_dict['total_loss']
                if accum_steps > 1:
                    loss = loss / accum_steps

            if device.type == 'cuda':
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss_dict['total_loss'].item()
            running_id += loss_dict['l_id'].item()
            running_c += loss_dict['l_c'].item()
            running_ca += loss_dict['l_ca'].item()
            running_ac += loss_dict['l_ac'].item()
            running_id_prime += loss_dict['l_id_prime'].item()
            running_r += loss_dict['l_r'].item()

        scheduler.step()
        num_batches = len(train_loader)
        avg_loss = running_loss / num_batches
        avg_id = running_id / num_batches
        avg_c = running_c / num_batches
        avg_ca = running_ca / num_batches
        avg_ac = running_ac / num_batches
        avg_id_prime = running_id_prime / num_batches
        avg_r = running_r / num_batches

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Periodic Evaluation
        r1, r5, r10, map_val = 0.0, 0.0, 0.0, 0.0
        if epoch % eval_freq == 0 or epoch == epochs:
            r1, r5, r10, map_val = evaluate_paper_model(model, query_loader, gallery_loader, device)

            is_best = r1 > best_rank1 or (r1 == best_rank1 and map_val > best_map)
            if is_best:
                best_rank1 = r1
                best_map = map_val
                best_epoch = epoch

                # Save best checkpoint safely
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'rank1': r1,
                    'mAP': map_val,
                    'num_classes': num_classes,
                    'num_clothes': num_clothes,
                    'architecture': 'PaperDCRReID',
                    'backbone': 'ResNet-50-layer4-stride1'
                }, best_ckpt_path)

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'rank1': r1,
            'mAP': map_val,
            'num_classes': num_classes,
            'num_clothes': num_clothes,
            'architecture': 'PaperDCRReID',
            'backbone': 'ResNet-50-layer4-stride1'
        }, latest_ckpt_path)

        eval_str_r1 = f"{r1:.2f}%" if (epoch % eval_freq == 0 or epoch == epochs) else "---"
        eval_str_map = f"{map_val:.2f}%" if (epoch % eval_freq == 0 or epoch == epochs) else "---"

        print(
            f"{epoch:<6} | {stage:<6} | {avg_loss:<11.4f} | {avg_id:<8.4f} | {avg_c:<8.4f} | "
            f"{avg_ca:<8.4f} | {avg_ac:<8.4f} | {avg_r:<8.4f} | {eval_str_r1:<10} | {eval_str_map:<8} | {epoch_time:.1f}s",
            flush=True
        )

        csv_writer.writerow([
            epoch, stage, current_lr, avg_loss, avg_id, avg_c,
            avg_ca, avg_ac, avg_id_prime, avg_r,
            r1, r5, r10, map_val, epoch_time
        ])
        csv_file.flush()

    csv_file.close()
    total_time = time.time() - start_time

    best_sha256 = get_sha256(best_ckpt_path)
    latest_sha256 = get_sha256(latest_ckpt_path)

    print("\n==========================================================================================")
    print("  PAPER DCR-ReID TRAINING COMPLETE")
    print("==========================================================================================")
    print(f"[*] Total Training Duration    : {total_time / 60:.2f} minutes")
    print(f"[*] Peak Convergence Epoch    : Epoch {best_epoch}")
    print(f"[*] Peak Rank-1 / mAP          : {best_rank1:.2f}% / {best_map:.2f}%")
    print(f"[*] Best Checkpoint Saved To   : {best_ckpt_path} (SHA-256: {best_sha256})")
    print(f"[*] Latest Checkpoint Saved To : {latest_ckpt_path} (SHA-256: {latest_sha256})")
    print(f"[*] Telemetry CSV Log          : {log_csv_path}")
    print("==========================================================================================\n")

    return {
        'best_epoch': best_epoch,
        'best_rank1': best_rank1,
        'best_map': best_map,
        'best_ckpt_path': best_ckpt_path,
        'best_sha256': best_sha256,
        'total_time_s': total_time
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Paper-Faithful DCR-ReID (Track A)")
    parser.add_argument('--data-dir', '--data_dir', type=str, default='./Data_set', dest='data_dir')
    parser.add_argument('--dataset-type', '--dataset_type', type=str, default='prcc_cc', dest='dataset_type')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--stage1-epochs', '--stage1_epochs', type=int, default=10, dest='stage1_epochs')
    parser.add_argument('--batch-size', '--batch_size', type=int, default=32, dest='batch_size')
    parser.add_argument('--lr', type=float, default=3.5e-4)
    parser.add_argument('--weight-decay', '--weight_decay', type=float, default=5e-4, dest='weight_decay')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--img-height', '--img_height', type=int, default=384, dest='img_height')
    parser.add_argument('--img-width', '--img_width', type=int, default=192, dest='img_width')
    parser.add_argument('--num-classes', '--num_classes', type=int, default=150, dest='num_classes')
    parser.add_argument('--num-clothes', '--num_clothes', type=int, default=300, dest='num_clothes')
    parser.add_argument('--num-instances', '--num_instances', type=int, default=4, dest='num_instances')
    parser.add_argument('--accum-steps', '--accum_steps', type=int, default=1, dest='accum_steps')
    parser.add_argument('--save-dir', '--save_dir', type=str, default='./checkpoints', dest='save_dir')
    parser.add_argument('--log-dir', '--log_dir', type=str, default='./reports', dest='log_dir')
    parser.add_argument('--eval-freq', '--eval_freq', type=int, default=5, dest='eval_freq')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    train_paper_dcr(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        epochs=args.epochs,
        stage1_epochs=args.stage1_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        gamma=args.gamma,
        img_height=args.img_height,
        img_width=args.img_width,
        num_classes=args.num_classes,
        num_clothes=args.num_clothes,
        num_instances=args.num_instances,
        accum_steps=args.accum_steps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        device_name=args.device,
        seed=args.seed
    )

