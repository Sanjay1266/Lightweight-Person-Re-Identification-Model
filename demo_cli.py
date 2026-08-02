import os
import argparse
from utils.reid_engine import ReIDEngine
from data.dataset_loader import PersonReIDDataset

def run_demo(query_img_path, dataset_type='both_small', top_k=5):
    print("==============================================================")
    print("   LIGHTWEIGHT PERSON RE-IDENTIFICATION CLI DEMO ENGINE       ")
    print("==============================================================")

    data_dir = './Data_set'
    dataset = PersonReIDDataset(data_dir=data_dir, dataset_type=dataset_type)
    gallery_samples = dataset.gallery_samples

    print(f"=> Query Image  : {query_img_path}")
    print(f"=> Gallery Size : {len(gallery_samples)} images ({dataset_type})")
    print(f"=> Searching Top-{top_k} Matches...\n")

    engine = ReIDEngine(model_path='./checkpoints/best_model.pth')
    matches, _, _ = engine.rank_query(query_img_path, gallery_samples[:200], top_k=top_k)

    print(f"{'Rank':<6} | {'Similarity':<12} | {'Person ID':<10} | {'Camera ID':<10} | {'Gallery Image Path'}")
    print("-" * 80)

    for m in matches:
        print(f"#{m['rank']:<5} | {m['similarity']:<11}% | {m['pid']:<10} | {m['camid']+1:<10} | {m['img_path']}")

    print("==============================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLI Person Re-ID Query Search")
    parser.add_argument('--query', type=str, default='./Data_set/both_small/query/0083_c1_f10.jpg')
    parser.add_argument('--dataset_type', type=str, default='both_small')
    parser.add_argument('--top_k', type=int, default=5)

    args = parser.parse_args()
    if os.path.exists(args.query):
        run_demo(args.query, dataset_type=args.dataset_type, top_k=args.top_k)
    else:
        print(f"Error: Query image '{args.query}' not found.")
