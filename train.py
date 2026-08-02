import os
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset_loader import get_dataloaders
from models.lightweight_reid import LightweightReIDNet
from models.loss import CombinedLoss
from utils.metrics import compute_distance_matrix, eval_market1501

def evaluate_model(model, query_loader, gallery_loader, device):
    model.eval()

    q_feats, q_pids, q_cams = [], [], []
    g_feats, g_pids, g_cams = [], [], []

    with torch.no_grad():
        for imgs, pids, cams, _ in query_loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            q_feats.append(feats.cpu().numpy())
            q_pids.extend(pids.numpy())
            q_cams.extend(cams.numpy())

        for imgs, pids, cams, _ in gallery_loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            g_feats.append(feats.cpu().numpy())
            g_pids.extend(pids.numpy())
            g_cams.extend(cams.numpy())

    q_feats = np.vstack(q_feats)
    g_feats = np.vstack(g_feats)
    q_pids = np.array(q_pids)
    g_pids = np.array(g_pids)
    q_cams = np.array(q_cams)
    g_cams = np.array(g_cams)

    distmat = compute_distance_matrix(q_feats, g_feats, metric='cosine')
    cmc, mAP = eval_market1501(distmat, q_pids, g_pids, q_cams, g_cams, max_rank=10)

    rank1 = cmc[0] * 100
    rank5 = cmc[4] * 100 if len(cmc) >= 5 else 0.0
    rank10 = cmc[9] * 100 if len(cmc) >= 10 else 0.0
    mAP_pct = mAP * 100

    return rank1, rank5, rank10, mAP_pct


def train(data_dir='./Data_set', dataset_type='both_small', epochs=15, batch_size=16, lr=0.0003, save_dir='./checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=> Training using device: {device}")

    dataset, train_loader, query_loader, gallery_loader = get_dataloaders(
        data_dir=data_dir, dataset_type=dataset_type, batch_size=batch_size
    )
    dataset.print_dataset_summary()

    num_classes = dataset.num_train_pids
    print(f"=> Training Model for {num_classes} Person Identities...")

    model = LightweightReIDNet(num_classes=num_classes, feat_dim=512).to(device)
    criterion = CombinedLoss(num_classes=num_classes, margin=0.3, epsilon=0.1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_rank1 = 0.0
    best_mAP = 0.0

    print("\n--- Starting Training Loop ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_triplet = 0.0

        for batch_idx, (imgs, pids, _, _) in enumerate(train_loader, start=1):
            imgs = imgs.to(device)
            pids = pids.to(device)

            optimizer.zero_grad()
            cls_score, global_feat, _ = model(imgs)
            loss, loss_ce, loss_triplet = criterion(cls_score, global_feat, pids)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce += loss_ce.item()
            running_triplet += loss_triplet.item()

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        avg_ce = running_ce / len(train_loader)
        avg_triplet = running_triplet / len(train_loader)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Triplet: {avg_triplet:.4f}) | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Evaluate every 5 epochs or on final epoch
        if epoch % 5 == 0 or epoch == epochs:
            rank1, rank5, rank10, mAP = evaluate_model(model, query_loader, gallery_loader, device)
            print(f"   => Eval Epoch {epoch:02d} | Rank-1: {rank1:.2f}% | Rank-5: {rank5:.2f}% | Rank-10: {rank10:.2f}% | mAP: {mAP:.2f}%")

            if rank1 > best_rank1 or (rank1 == best_rank1 and mAP > best_mAP):
                best_rank1 = rank1
                best_mAP = mAP
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"   [+] Saved new best checkpoint to '{best_model_path}'")

    total_time = time.time() - start_time
    print(f"\n=> Training Completed in {total_time/60:.2f} minutes.")
    print(f"=> Best Evaluation Performance: Rank-1 = {best_rank1:.2f}% | mAP = {best_mAP:.2f}%")

    # Save final model state
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"=> Saved final model checkpoint to '{final_model_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Lightweight Person Re-Identification Model")
    parser.add_argument('--data_dir', type=str, default='./Data_set')
    parser.add_argument('--dataset_type', type=str, default='both_small', choices=['both_small', 'with_bag', 'without_bag', 'both_large'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    args = parser.parse_args()
    train(data_dir=args.data_dir, dataset_type=args.dataset_type, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_dir=args.save_dir)
