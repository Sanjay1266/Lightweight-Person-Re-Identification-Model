"""
Training Pipeline for Lightweight DCR-ReID
Base Paper: "DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification"
IEEE TCSVT 2023 | Course: 23CSE373 - Computer Vision

Implements Two-Stage Optimization Protocol:
Stage 1: Disentanglement initialization (L_ID + L_triplet + L_c + L_R)
Stage 2: Full DCR-ReID optimization (L_ID + L_triplet + L_c + L_ca + alpha * L_ac + gamma * L_ID' + L_R)
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import time
import argparse
import csv
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    Generates pseudo ground-truth component masks (T_i^+, T_i^-, T_i^t) for component reconstruction:
    - T_i^t: Human silhouette contour map (Sobel gradient filter)
    - T_i^+: Non-clothing body geometry (head/face & limb structural proportions)
    - T_i^-: Clothing / apparel & accessory regions (torso & middle body)
    """
    b, c, h, w = imgs.shape
    device = imgs.device

    # 1. Edge detector for contour map T_i^t (Eq. 9-10)
    gray = 0.299 * imgs[:, 0:1] + 0.587 * imgs[:, 1:2] + 0.114 * imgs[:, 2:3]
    # Sobel filters
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    target_contour = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    target_contour = torch.sigmoid(target_contour * 3.0)

    # 2. Structural masks based on human anatomy geometry
    y_coords = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
    
    # Non-clothing body regions (head: y < 0.22, lower legs: y > 0.85)
    target_irrel = ((y_coords < 0.22) | (y_coords > 0.85)).float()
    
    # Clothing / accessory regions (torso / pants: 0.22 <= y <= 0.85)
    target_rel = ((y_coords >= 0.22) & (y_coords <= 0.85)).float()

    return target_irrel, target_rel, target_contour


def evaluate_model(model, query_loader, gallery_loader, device, max_eval_q=120, max_eval_g=500):
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


def train(data_dir='./Data_set', dataset_type='both_small', epochs=15, batch_size=16, lr=0.0003,
          save_dir='./checkpoints/finetuned', resume=None, max_batches=None, use_pk_sampler=True,
          experiment_id="EXP_RUN"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=> DCR-ReID Training Initialized on Device: {device}")

    dataset, train_loader, query_loader, gallery_loader = get_dataloaders(
        data_dir=data_dir, dataset_type=dataset_type, batch_size=batch_size,
        use_pk_sampler=use_pk_sampler, num_instances=4
    )
    dataset.print_dataset_summary()

    num_classes = dataset.num_train_pids
    num_clothes = max(20, min(100, num_classes * 2))
    p_ids = batch_size // 4
    k_inst = 4
    print(f"=> Training Model for {num_classes} Person Identities with {num_clothes} Clothes Classes...")
    if use_pk_sampler:
        print(f"=> PK Sampler Active: P={p_ids} distinct identities, K={k_inst} instances/identity (Batch Size={batch_size})")

    model = LightweightDCRReID(num_classes=num_classes, num_clothes=num_clothes, feat_dim=512).to(device)

    # Resume from checkpoint if provided
    starting_ckpt_hash = None
    if resume and os.path.exists(resume):
        starting_ckpt_hash = get_sha256(resume)
        state_dict = torch.load(resume, map_location=device)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"=> Loaded {len(filtered_dict)}/{len(model_dict)} tensors from starting checkpoint: '{resume}' (SHA-256: {starting_ckpt_hash[:12]}...)")

    criterion = DCRReIDCombinedLoss(num_classes=num_classes, num_clothes=num_clothes, margin=0.3, epsilon=0.1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    best_rank1 = 0.0
    best_mAP = 0.0

    print("\n--- Starting DCR-ReID Training Loop (Two-Stage Optimization) ---")
    start_time = time.time()

    stage_switch_epoch = max(1, epochs // 3)

    log_csv = 'reports/experiment_log.csv'
    csv_exists = os.path.exists(log_csv)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        # Two-stage schedule per Section IV.B
        if epoch <= stage_switch_epoch:
            criterion.stage = 1
            stage_name = "Stage 1 (Disentanglement Init)"
        else:
            criterion.stage = 2
            stage_name = "Stage 2 (Full DCR-ReID & Adversarial)"

        model.train()
        running_loss = 0.0
        running_id = 0.0
        running_triplet = 0.0
        running_recon = 0.0
        running_clothes = 0.0
        batches_processed = 0
        total_images_processed = 0

        for batch_idx, (imgs, pids, cams, _) in enumerate(train_loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            imgs = imgs.to(device)
            pids = pids.to(device)
            cams = cams.to(device)

            # Validated clothing label synthesis:
            # For PRCC: Camera A (0) and B (1) share clothes (0), Cam C (2) has different clothes (1)
            clothes_labels = generate_clothing_labels(pids, cams, is_prcc=dataset.is_prcc, num_clothes=num_clothes)

            # Generate pseudo component target masks
            target_irrel, target_rel, target_contour = generate_pseudo_component_masks(imgs)

            optimizer.zero_grad()
            outputs = model(imgs, shuffle_clothes=(criterion.stage == 2))

            loss_dict = criterion(
                outputs, pids, clothes_labels=clothes_labels,
                target_irrel=target_irrel, target_rel=target_rel, target_contour=target_contour
            )

            total_loss = loss_dict['total_loss']
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_id += loss_dict['loss_id'].item()
            running_triplet += loss_dict['loss_triplet'].item()
            running_recon += loss_dict['loss_recon'].item()
            running_clothes += loss_dict['loss_clothes'].item()
            batches_processed += 1
            total_images_processed += imgs.size(0)

        scheduler.step()

        n_batches = max(1, batches_processed)
        avg_loss = running_loss / n_batches
        avg_id = running_id / n_batches
        avg_trip = running_triplet / n_batches
        avg_recon = running_recon / n_batches
        avg_clothes = running_clothes / n_batches
        epoch_duration = time.time() - epoch_start_time
        img_per_sec = total_images_processed / max(0.01, epoch_duration)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] {stage_name} | Loss: {avg_loss:.4f} (ID: {avg_id:.3f}, Trip: {avg_trip:.3f}, Recon: {avg_recon:.4f}, Clothes: {avg_clothes:.3f}) | Speed: {img_per_sec:.1f} img/s | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Periodic Evaluation
        rank1, rank5, rank10, mAP = 0.0, 0.0, 0.0, 0.0
        if epoch % 5 == 0 or epoch == epochs or max_batches is not None:
            rank1, rank5, rank10, mAP = evaluate_model(model, query_loader, gallery_loader, device)
            print(f"   => Eval Epoch {epoch:02d} | Rank-1: {rank1:.2f}% | Rank-5: {rank5:.2f}% | Rank-10: {rank10:.2f}% | mAP: {mAP:.2f}%")

            if rank1 >= best_rank1:
                best_rank1 = rank1
                best_mAP = mAP
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"   [+] Saved new checkpoint to '{best_model_path}'")

        # Append to CSV log
        with open(log_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists and epoch == 1 and f.tell() == 0:
                writer.writerow([
                    'experiment_id', 'epoch', 'stage', 'total_loss', 'loss_id',
                    'loss_triplet', 'loss_recon', 'loss_clothes', 'rank1', 'rank5',
                    'rank10', 'mAP', 'lr', 'img_per_sec', 'epoch_duration_sec',
                    'device', 'starting_checkpoint_hash', 'dataset_type'
                ])
                csv_exists = True
            writer.writerow([
                experiment_id, epoch, criterion.stage, round(avg_loss, 4),
                round(avg_id, 4), round(avg_trip, 4), round(avg_recon, 4),
                round(avg_clothes, 4), round(rank1, 2), round(rank5, 2),
                round(rank10, 2), round(mAP, 2), round(scheduler.get_last_lr()[0], 6),
                round(img_per_sec, 1), round(epoch_duration, 2), str(device),
                starting_ckpt_hash or "Scratch", dataset_type
            ])

    total_time = time.time() - start_time
    print(f"\n=> Training Completed in {total_time/60:.2f} minutes.")
    print(f"=> Best Validation Metric: Rank-1 = {best_rank1:.2f}% | mAP = {best_mAP:.2f}%")

    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"=> Saved final model checkpoint to '{final_model_path}'")

    return {
        'best_rank1': best_rank1,
        'best_mAP': best_mAP,
        'total_time_sec': total_time,
        'best_model_path': os.path.join(save_dir, 'best_model.pth'),
        'final_model_path': final_model_path
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Lightweight DCR-ReID Model")
    parser.add_argument('--data_dir', type=str, default='./Data_set')
    parser.add_argument('--dataset_type', type=str, default='both_small', 
                        choices=['both_small', 'with_bag', 'without_bag', 'both_large', 'prcc', 'prcc_cc', 'prcc_sc', 'prcc_all'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/finetuned')
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to fine-tune from")
    parser.add_argument('--max_batches', type=int, default=None, 
                        help="Maximum training batches per epoch (useful for validation/quick testing)")
    parser.add_argument('--experiment_id', type=str, default='EXP_RUN')

    args = parser.parse_args()
    train(data_dir=args.data_dir, dataset_type=args.dataset_type, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, save_dir=args.save_dir,
          resume=args.resume, max_batches=args.max_batches, experiment_id=args.experiment_id)
