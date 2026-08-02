import os
import argparse
import numpy as np
import torch

from data.dataset_loader import get_dataloaders
from models.lightweight_reid import LightweightReIDNet
from utils.metrics import compute_distance_matrix, eval_market1501

def evaluate_subset(model, data_dir, dataset_type, device, batch_size=16):
    dataset, _, query_loader, gallery_loader = get_dataloaders(
        data_dir=data_dir, dataset_type=dataset_type, batch_size=batch_size
    )
    
    q_feats, q_pids, q_cams = [], [], []
    g_feats, g_pids, g_cams = [], [], []

    model.eval()
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

    return {
        'dataset_type': dataset_type,
        'num_query': len(q_pids),
        'num_gallery': len(g_pids),
        'rank1': round(rank1, 2),
        'rank5': round(rank5, 2),
        'rank10': round(rank10, 2),
        'mAP': round(mAP_pct, 2)
    }

def main(data_dir='./Data_set', model_path='./checkpoints/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=> Evaluating Model on Device: {device}")

    num_classes = 751  # default dummy class count for evaluation feature extraction
    model = LightweightReIDNet(num_classes=num_classes, feat_dim=512)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"=> Successfully loaded model checkpoint: '{model_path}'")
    else:
        print(f"=> Warning: Model checkpoint '{model_path}' not found! Evaluating initialized model weights.")

    model.to(device)

    subsets = ['both_small', 'with_bag', 'without_bag', 'both_large']
    results = []

    print("\n==============================================================")
    print("      LIGHTWEIGHT PERSON RE-IDENTIFICATION BENCHMARK RESULTS   ")
    print("==============================================================")
    print(f"{'Subset':<15} | {'Query Imgs':<10} | {'Gallery Imgs':<12} | {'Rank-1 (%)':<10} | {'Rank-5 (%)':<10} | {'mAP (%)':<10}")
    print("-" * 80)

    for sub in subsets:
        try:
            res = evaluate_subset(model, data_dir, sub, device)
            results.append(res)
            print(f"{res['dataset_type']:<15} | {res['num_query']:<10} | {res['num_gallery']:<12} | {res['rank1']:<10.2f} | {res['rank5']:<10.2f} | {res['mAP']:<10.2f}")
        except Exception as e:
            print(f"{sub:<15} | Error: {e}")

    print("==============================================================\n")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Person Re-ID Model Performance")
    parser.add_argument('--data_dir', type=str, default='./Data_set')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth')

    args = parser.parse_args()
    main(data_dir=args.data_dir, model_path=args.model_path)
