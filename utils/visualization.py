"""
Visualization Utilities for Lightweight DCR-ReID
Base Paper: "DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification" (IEEE TCSVT 2023)

Generates:
1. 4-Panel Component Disentanglement Visualizer:
   - Original Query
   - Clothes-Irrelevant Body Geometry (Y_i^+)
   - Clothes-Relevant Apparel / Bag Region (Y_i^-)
   - Human Contour Boundary Map (Y_i^t)
2. DAD Assembled Identity Feature Activation Overlay
3. t-SNE Embedding Scatter Plot
"""

import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

def generate_attention_heatmap(model, img_tensor, original_pil_img, save_path=None):
    """
    Generates 4-panel visual component maps from the DCR-ReID model:
    1. Input Query Image
    2. Body Shape & Facial Geometry (Clothes-Irrelevant Y_i^+)
    3. Apparel & Bag Pattern (Clothes-Relevant Y_i^-)
    4. DAD Assembled Feature Overlay (amplified identity landmarks)
    """
    model.eval()
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        if hasattr(model, 'extract_disentangled_maps'):
            maps = model.extract_disentangled_maps(img_tensor)
            recon_irrel = maps['recon_irrel'][0, 0].cpu().numpy()
            recon_rel = maps['recon_rel'][0, 0].cpu().numpy()
            recon_contour = maps['recon_contour'][0, 0].cpu().numpy()
            assembled = maps['assembled_feat'][0]
            activation_map = assembled
        else:
            stem_out = model.stem(img_tensor)
            s1 = model.stage1(stem_out)
            s2 = model.stage2(s1)
            s3 = model.stage3(s2)
            base_feat = model.stage4(s3)
            activation_map = base_feat[0]
            recon_irrel = torch.mean(base_feat[0], dim=0).cpu().numpy()
            recon_rel = torch.mean(base_feat[0], dim=0).cpu().numpy()
            recon_contour = torch.mean(base_feat[0], dim=0).cpu().numpy()

    # Average activation across channels for DAD overlay
    heatmap = torch.mean(activation_map, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    w, h = original_pil_img.size

    # Resize component maps to image resolution
    def resize_map(m):
        m = np.maximum(m, 0)
        if np.max(m) > 0:
            m = m / np.max(m)
        img_m = Image.fromarray(np.uint8(255 * m)).resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(img_m) / 255.0

    irrel_np = resize_map(recon_irrel)
    rel_np = resize_map(recon_rel)
    contour_np = resize_map(recon_contour)
    heatmap_np = resize_map(heatmap)

    # Overlay
    img_np = np.asarray(original_pil_img).astype(np.float32) / 255.0
    cmap = plt.get_cmap('magma')
    colored_heatmap = cmap(heatmap_np)[:, :, :3]
    overlay = 0.55 * img_np + 0.45 * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), facecolor='#0f172a')
    
    titles = [
        "1. Input Query",
        "2. Body Shape & Face (Y⁺)",
        "3. Clothing & Bag (Y⁻)",
        "4. DAD Assembled Feature"
    ]

    axes[0].imshow(original_pil_img)
    axes[1].imshow(irrel_np, cmap='Blues')
    axes[2].imshow(rel_np, cmap='Reds')
    axes[3].imshow(overlay)

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=10, fontweight='bold', color='#f8fafc', pad=8)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        return save_path
    else:
        plt.close()
        return overlay


def plot_tsne_embeddings(embeddings, labels, bag_flags, save_path):
    """
    Plots t-SNE 2D scatter plot of person identity embeddings categorized by identity and bag condition.
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(embeddings) - 1)))
    coords_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    unique_pids = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pids)))

    for idx, pid in enumerate(unique_pids):
        mask_with_bag = (labels == pid) & (bag_flags == 1)
        mask_without_bag = (labels == pid) & (bag_flags == 0)

        if np.any(mask_with_bag):
            ax.scatter(coords_2d[mask_with_bag, 0], coords_2d[mask_with_bag, 1],
                       c=[colors[idx]], marker='o', label=f'PID {pid} (Bag)', alpha=0.85, s=60, edgecolors='#f8fafc', linewidths=0.5)

        if np.any(mask_without_bag):
            ax.scatter(coords_2d[mask_without_bag, 0], coords_2d[mask_without_bag, 1],
                       c=[colors[idx]], marker='^', label=f'PID {pid} (No Bag)', alpha=0.85, s=60, edgecolors='#38bdf8', linewidths=0.5)

    ax.set_title("t-SNE Embedding Manifold (DCR-ReID Clothes Disentanglement)", color='#f8fafc', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#334155')

    handles, legend_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(legend_labels[:10], handles[:10]))
    legend = ax.legend(by_label.values(), by_label.keys(), loc='upper right', facecolor='#0f172a', edgecolor='#334155')
    for text in legend.get_texts():
        text.set_color('#f8fafc')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return save_path
