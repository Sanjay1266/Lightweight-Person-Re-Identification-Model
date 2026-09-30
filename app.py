"""
Web Application Backend for Lightweight DCR-ReID
Base Paper: "DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification" (IEEE TCSVT 2023)
Course: 23CSE373 - Computer Vision | Amrita School of Engineering
"""

import os
import glob
import random
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch

from utils.reid_engine import ReIDEngine
from utils.visualization import generate_attention_heatmap, plot_tsne_embeddings
from evaluate import PAPER_TARGET_BENCHMARKS

app = Flask(__name__, static_folder='static', template_folder='templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data_set')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoints', 'best_model.pth')
OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Re-ID Inference Engine with verified checkpoint
engine = ReIDEngine(model_path=CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None)

def get_gallery_samples(dataset_type='both_small', limit=200):
    gallery_dir = os.path.join(DATA_DIR, dataset_type, 'bounding_box_test')
    if not os.path.exists(gallery_dir):
        return []
    img_paths = glob.glob(os.path.join(gallery_dir, '*.jpg'))
    samples = []
    import re
    pattern = re.compile(r'([-\d]+)_c(\d+)')
    for path in img_paths[:limit]:
        fname = os.path.basename(path)
        res = pattern.search(fname)
        if res:
            pid, camid = map(int, res.groups())
            if pid != -1:
                samples.append((path, pid, camid))
    return samples


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sample_queries', methods=['GET'])
def get_sample_queries():
    dataset_type = request.args.get('dataset_type', 'both_small')
    query_dir = os.path.join(DATA_DIR, dataset_type, 'query')
    if not os.path.exists(query_dir):
        return jsonify({'error': 'Dataset subset query directory not found'}), 404

    img_paths = glob.glob(os.path.join(query_dir, '*.jpg'))
    sampled = random.sample(img_paths, min(24, len(img_paths)))
    
    samples_info = []
    for path in sampled:
        fname = os.path.basename(path)
        samples_info.append({
            'filename': fname,
            'rel_path': f"{dataset_type}/query/{fname}"
        })
    return jsonify({'samples': samples_info})


@app.route('/dataset_image/<path:filename>')
def serve_dataset_image(filename):
    return send_from_directory(DATA_DIR, filename)


@app.route('/api/reid_query', methods=['POST'])
def handle_reid_query():
    top_k = int(request.form.get('top_k', 10))
    dataset_type = request.form.get('dataset_type', 'both_small')
    
    query_img_path = None
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        temp_path = os.path.join(OUTPUT_DIR, f"temp_query_{random.randint(1000, 9999)}.jpg")
        file.save(temp_path)
        query_img_path = temp_path
    elif 'sample_rel_path' in request.form:
        rel_path = request.form['sample_rel_path']
        query_img_path = os.path.join(DATA_DIR, rel_path)

    if not query_img_path or not os.path.exists(query_img_path):
        return jsonify({'error': 'No valid query image provided'}), 400

    gallery = get_gallery_samples(dataset_type=dataset_type, limit=250)
    if not gallery:
        return jsonify({'error': f'No gallery images available for {dataset_type}'}), 404

    top_matches, query_pil, query_tensor = engine.rank_query(query_img_path, gallery, top_k=top_k)

    # Format paths for Web UI
    for match in top_matches:
        abs_p = match['img_path']
        rel_p = os.path.relpath(abs_p, DATA_DIR).replace('\\', '/')
        match['web_url'] = f"/dataset_image/{rel_p}"

    # Generate 4-Panel DCR-ReID Component Disentanglement Visualization
    heatmap_filename = f"heatmap_{random.randint(10000, 99999)}.png"
    heatmap_save_path = os.path.join(OUTPUT_DIR, heatmap_filename)
    generate_attention_heatmap(engine.model, query_tensor, query_pil, save_path=heatmap_save_path)

    # Format Query Web URL
    if query_img_path.startswith(DATA_DIR):
        rel_query = os.path.relpath(query_img_path, DATA_DIR).replace('\\', '/')
        query_web_url = f"/dataset_image/{rel_query}"
    else:
        rel_query = os.path.relpath(query_img_path, BASE_DIR).replace('\\', '/')
        query_web_url = f"/static/{os.path.relpath(query_img_path, os.path.join(BASE_DIR, 'static')).replace('\\', '/')}"

    return jsonify({
        'query_url': query_web_url,
        'heatmap_url': f"/static/outputs/{heatmap_filename}",
        'matches': top_matches,
        'dataset_type': dataset_type
    })


@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    return jsonify({
        'title': 'Lightweight Person Re-Identification Model (DCR-ReID)',
        'course': '23CSE373 - Computer Vision',
        'base_paper': 'DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification (IEEE TCSVT 2023)',
        'team': [
            {'name': 'G N Bhuvaneshwaran', 'roll': 'CB.SC.U4CSE24218'},
            {'name': 'Sanjay MS', 'roll': 'CB.SC.U4CSE24248'},
            {'name': 'Sanjay S', 'roll': 'CB.SC.U4CSE24249'}
        ],
        'branches': {
            'pi_branch': 'Person Identification Branch: 3.2M Backbone, BNNeck & Hard Triplet Ranking with clothes-invariant inference',
            'cr_branch': 'Component Reconstruction Branch: 3-way Channel Decomposition (P⁺, P⁻, Pᵗ) with 4-block ψ decoders',
            'ci_branch': 'Clothes Identification Branch: DAD Assembled Disentanglement (G_i = F⁺ ⊕ F_ai⁻ ⊕ Fᵗ) & Adversarial Loss'
        },
        'specs': {
            'backbone_params': '3.24 M',
            'model_size': '12.9 MB',
            'inference_latency': '18.4 ms',
            'throughput': '54.3 FPS',
            'resnet50_comparison': {
                'resnet50_params': '25.6 M',
                'param_reduction': '87.3%',
                'speedup': '3.8x faster'
            }
        },
        'convergence_status': {
            'status': 'In Progress / Iterative Multi-Stage Training',
            'current_checkpoint_rank1': '13.13% (both_small) / 56.00% (both_large)',
            'target_benchmark_rank1': '88.52% (both_small)',
            'target_benchmark_map': '82.40% (both_small)'
        }
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    # Reports both empirical checkpoint performance and target paper convergence benchmarks
    metrics = [
        {
            'dataset_type': 'both_small',
            'condition': 'Combined Benchmark',
            'num_query': 475,
            'num_gallery': 2146,
            'current_rank1': 13.13,
            'current_rank5': 24.95,
            'current_map': 6.98,
            'target_rank1': 88.52,
            'target_rank5': 96.24,
            'target_map': 82.40,
            'convergence_pct': 14.8
        },
        {
            'dataset_type': 'with_bag',
            'condition': 'Person with Bag / Accessory',
            'num_query': 322,
            'num_gallery': 1131,
            'current_rank1': 8.50,
            'current_rank5': 18.20,
            'current_map': 9.22,
            'target_rank1': 85.10,
            'target_rank5': 94.81,
            'target_map': 79.12,
            'convergence_pct': 10.0
        },
        {
            'dataset_type': 'without_bag',
            'condition': 'Person without Bag',
            'num_query': 179,
            'num_gallery': 986,
            'current_rank1': 44.44,
            'current_rank5': 63.89,
            'current_map': 30.79,
            'target_rank1': 91.24,
            'target_rank5': 97.60,
            'target_map': 85.74,
            'convergence_pct': 48.7
        },
        {
            'dataset_type': 'both_large',
            'condition': 'Full Surveillance Benchmark',
            'num_query': 1265,
            'num_gallery': 10048,
            'current_rank1': 56.00,
            'current_rank5': 92.00,
            'current_map': 44.90,
            'target_rank1': 87.80,
            'target_rank5': 95.92,
            'target_map': 81.65,
            'convergence_pct': 63.8
        }
    ]
    return jsonify({'metrics': metrics})


if __name__ == '__main__':
    print("=> Starting Lightweight DCR-ReID Web Dashboard on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
