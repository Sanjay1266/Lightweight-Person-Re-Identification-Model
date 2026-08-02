import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

def generate_attention_heatmap(model, img_tensor, original_pil_img, save_path=None):
    """
    Generates visual attention map (Grad-CAM style feature activation map) from the trained model's final conv layer.
    """
    model.eval()
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        stem_out = model.stem(img_tensor)
        l1 = model.layer1(stem_out)
        l2 = model.layer2(l1)
        l3 = model.layer3(l2)
        feat_map = model.layer4(l3)

    # Average activation across channels
    heatmap = torch.mean(feat_map[0], dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    # Resize heatmap to match image size
    w, h = original_pil_img.size
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.Resampling.BILINEAR)
    heatmap_np = np.asarray(heatmap_img) / 255.0

    # Overlay heatmap on top of original image
    img_np = np.asarray(original_pil_img).astype(np.float32) / 255.0
    cmap = plt.get_cmap('jet')
    colored_heatmap = cmap(heatmap_np)[:, :, :3]

    overlay = 0.55 * img_np + 0.45 * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(original_pil_img)
    axes[0].set_title("Input Query Image")
    axes[0].axis('off')

    axes[1].imshow(heatmap_np, cmap='jet')
    axes[1].set_title("Attention Activation")
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title("CBAM Feature Focus Overlay")
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
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
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    unique_pids = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pids)))

    for idx, pid in enumerate(unique_pids):
        mask_with_bag = (labels == pid) & (bag_flags == 1)
        mask_without_bag = (labels == pid) & (bag_flags == 0)

        if np.any(mask_with_bag):
            plt.scatter(coords_2d[mask_with_bag, 0], coords_2d[mask_with_bag, 1],
                        c=[colors[idx]], marker='o', label=f'PID {pid} (Bag)', alpha=0.8, s=50)

        if np.any(mask_without_bag):
            plt.scatter(coords_2d[mask_without_bag, 0], coords_2d[mask_without_bag, 1],
                        c=[colors[idx]], marker='^', label=f'PID {pid} (No Bag)', alpha=0.8, s=50)

    plt.title("t-SNE Projection of Lightweight Person Re-ID Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
