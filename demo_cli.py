import os
import glob
import time
import argparse
from utils.reid_engine import ReIDEngine
from data.dataset_loader import PersonReIDDataset

def run_demo(query_img_path, dataset_type='both_small', top_k=5, model_path='./checkpoints/best_model.pth'):
    print("==========================================================================")
    print("   LIGHTWEIGHT DCR-ReID (DEEP COMPONENT RECONSTRUCTION) INFERENCE ENGINE   ")
    print("   Base Paper: DCR-ReID for Cloth-Changing & Accessory Person Re-ID       ")
    print("   Course: 23CSE373 - Computer Vision                                    ")
    print("==========================================================================")

    data_dir = './Data_set'
    dataset = PersonReIDDataset(data_dir=data_dir, dataset_type=dataset_type)
    gallery_samples = dataset.gallery_samples

    print(f"=> Query Image  : {query_img_path}")
    print(f"=> Gallery Size : {len(gallery_samples)} images ({dataset_type})")
    print(f"=> Searching Top-{top_k} Matches using CRD & DAD Disentanglement...\n")

    engine = ReIDEngine(model_path=model_path if os.path.exists(model_path) else None)

    t0 = time.time()
    matches, _, _ = engine.rank_query(query_img_path, gallery_samples[:250], top_k=top_k)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"{'Rank':<6} | {'Similarity':<12} | {'Distance':<10} | {'Person ID':<10} | {'Camera ID':<10} | {'Gallery Image Path'}")
    print("-" * 90)

    for m in matches:
        print(f"#{m['rank']:<5} | {m['similarity']:<11.2f}% | {m['distance']:<10.4f} | {m['pid']:<10} | {m['camid']:<10} | {m['img_path']}")

    print("--------------------------------------------------------------------------")
    print(f"=> Inference Completed in {elapsed_ms:.1f} ms ({len(gallery_samples[:250])} gallery comparisons)")
    print(f"=> Model Backbone: Lightweight DCR-ReID (CRD Disentangled + DAD Assembled BNNeck)")
    print("==========================================================================\n")

def find_default_query(data_dir='./Data_set', dataset_type='both_small'):
    candidates = [
        os.path.join(data_dir, dataset_type, 'query'),
        os.path.join(data_dir, 'with_bag', 'query'),
        os.path.join(data_dir, 'without_bag', 'query'),
        os.path.join(data_dir, 'both_small', 'query')
    ]
    for c in candidates:
        if os.path.exists(c):
            imgs = glob.glob(os.path.join(c, '*.jpg'))
            if imgs:
                return imgs[0]
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLI Person Re-ID Query Search (Lightweight DCR-ReID)")
    parser.add_argument('--query', type=str, default=None, help='Path to query image')
    parser.add_argument('--dataset_type', type=str, default='both_small')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth')

    args = parser.parse_args()

    query_path = args.query
    if not query_path or not os.path.exists(query_path):
        query_path = find_default_query('./Data_set', args.dataset_type)
        if query_path:
            print(f"=> Auto-selected available query image: {query_path}")
        else:
            print(f"Error: No query image found for dataset {args.dataset_type}.")
            exit(1)

    run_demo(query_path, dataset_type=args.dataset_type, top_k=args.top_k, model_path=args.model_path)
