import os
import torch
import numpy as np
from PIL import Image
from models.lightweight_reid import LightweightReIDNet
from data.dataset_loader import build_transforms, PersonReIDDataset
from utils.metrics import compute_distance_matrix, eval_market1501

class ReIDEngine:
    def __init__(self, model_path=None, num_classes=751, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = LightweightReIDNet(num_classes=num_classes, feat_dim=512)
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            model_dict = self.model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict)
            print(f"=> Loaded model weights from '{model_path}' ({len(filtered_dict)}/{len(state_dict)} matched)")

        self.model.to(self.device)
        self.model.eval()

        self.transform = build_transforms(height=256, width=128, is_train=False)

    def extract_feature(self, img_input):
        """
        Extracts 512-dim normalized feature embedding vector for PIL Image or Image path.
        """
        if isinstance(img_input, str):
            pil_img = Image.open(img_input).convert('RGB')
        else:
            pil_img = img_input

        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(img_tensor)
            feat = feat.cpu().numpy()[0]
        return feat, pil_img, img_tensor

    def rank_query(self, query_img_path, gallery_samples, top_k=10):
        """
        Ranks gallery samples against a single query image.
        Returns Top-K matched gallery items with distance, similarity score, PID, and CamID.
        """
        query_feat, query_pil, query_tensor = self.extract_feature(query_img_path)

        if not hasattr(self, '_gallery_cache'):
            self._gallery_cache = {}

        # Cache key based on gallery sample paths
        cache_key = tuple(x[0] for x in gallery_samples[:10]) + (len(gallery_samples),)
        if cache_key in self._gallery_cache:
            gallery_feats, gallery_info = self._gallery_cache[cache_key]
        else:
            gallery_feats = []
            gallery_info = []
            # Batch extraction for fast inference
            batch_size = 32
            for i in range(0, len(gallery_samples), batch_size):
                chunk = gallery_samples[i:i + batch_size]
                tensors = []
                for g_path, g_pid, g_camid in chunk:
                    try:
                        img = Image.open(g_path).convert('RGB')
                        t = self.transform(img)
                        tensors.append(t)
                        gallery_info.append({
                            'img_path': g_path,
                            'pid': g_pid,
                            'camid': g_camid
                        })
                    except Exception:
                        continue
                if tensors:
                    batch_tensor = torch.stack(tensors).to(self.device)
                    with torch.no_grad():
                        feats = self.model(batch_tensor).cpu().numpy()
                    gallery_feats.append(feats)

            gallery_feats = np.vstack(gallery_feats) if gallery_feats else np.empty((0, 512))
            self._gallery_cache[cache_key] = (gallery_feats, gallery_info)

        distmat = compute_distance_matrix(query_feat.reshape(1, -1), gallery_feats, metric='cosine')[0]
        indices = np.argsort(distmat)

        top_matches = []
        for rank, idx in enumerate(indices[:top_k], start=1):
            dist = float(distmat[idx])
            similarity = max(0.0, round((1.0 - dist) * 100, 2))
            info = gallery_info[idx]
            top_matches.append({
                'rank': rank,
                'img_path': info['img_path'],
                'pid': info['pid'],
                'camid': info['camid'],
                'distance': round(dist, 4),
                'similarity': similarity
            })

        return top_matches, query_pil, query_tensor
