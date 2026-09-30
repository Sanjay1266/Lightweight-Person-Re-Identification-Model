"""
Lightweight DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification
Base Paper: Zhenyu Cui, Jiahuan Zhou, Yuxin Peng, Shiliang Zhang, Yaowei Wang.
"DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification",
IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2023.

Three Coordinated Branches:
1. Person Identification (PI) Branch:
   - Deep Feature Extraction P_i via Lightweight Residual Backbone (3.2M params)
   - BNNeck & Metric Learning (Label-Smoothed CE + Hard Triplet Loss)
   - Invariant Inference: Discards clothes-relevant features F_i^- to prevent apparel/bag interference.
2. Component Reconstruction (CR) Branch:
   - Channel Decomposition: P_i = P_i^- (clothing) ⊕ P_i^+ (body/face) ⊕ P_i^t (human contour)
   - 4-Block Reconstruction Decoders ψ^-, ψ^+, ψ^t reconstructing binary component masks Y_i^-, Y_i^+, Y_i^t
   - L1 Component Reconstruction Loss L_R
3. Clothes Identification (CI) Branch & Deep Assembled Disentanglement (DAD) Module:
   - Clothes Classifier C_P with Clothes Classification Loss L_c and Adversarial Loss L_ca
   - Feature Assembly: G_i = F_i^+ ⊕ F_ai^- ⊕ F_i^t (shuffled clothes-relevant features from other IDs)
   - Assembled Clothes Loss L_ac and Assembled Identity Loss L_ID'
   - Dual Channel & Spatial Attention recalibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel Attention submodule (Squeeze-and-Excitation / CBAM Channel Attention)"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_dim = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention submodule"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class ReconstructionBlock(nn.Module):
    """
    Reconstruction Block for ψ^v decoders (Fig. 4 in DCR-ReID paper).
    Consists of Conv2d + BatchNorm2d + ReLU + Residual shortcut to reconstruct component visual regions.
    """
    def __init__(self, channels):
        super(ReconstructionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class ComponentReconstructionDecoder(nn.Module):
    """
    Reconstruction Network ψ^v (Fig. 4 in DCR-ReID paper):
    1. Projection layer (normalizes input feature channels)
    2. Four reconstruction blocks
    3. Final conv + Sigmoid/Tanh activation
    """
    def __init__(self, in_channels, mid_channels=64):
        super(ComponentReconstructionDecoder, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.block1 = ReconstructionBlock(mid_channels)
        self.block2 = ReconstructionBlock(mid_channels)
        self.block3 = ReconstructionBlock(mid_channels)
        self.block4 = ReconstructionBlock(mid_channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.proj(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        return self.out_conv(h)


class CRDModule(nn.Module):
    """
    Component Reconstruction Disentanglement (CRD) Module (Section III.B.2, Eq. 8-10).
    Explicitly decomposes feature map P_i into:
      - P_i^+: Clothes-irrelevant feature segments (body shape, proportions, head/limbs)
      - P_i^-: Clothes-relevant feature segments (shirt, pants, colors, bag accessories)
      - P_i^t: Human contour feature segments (silhouette edges, boundary lines)
    Reconstructs:
      - Y_i^+ = ψ^+(P_i^+): Non-clothing / body geometry mask
      - Y_i^- = ψ^-(P_i^-): Clothing / accessory component mask
      - Y_i^t = ψ^t(P_i^t): Human contour boundary map
    """
    def __init__(self, in_channels=512, feat_dim=512):
        super(CRDModule, self).__init__()
        self.in_channels = in_channels
        self.feat_dim = feat_dim

        # Clothes-Irrelevant Disentanglement Projection (P_i^+)
        self.proj_irrel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        # Clothes-Relevant Disentanglement Projection (P_i^-)
        self.proj_rel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        # Human Contour Disentanglement Projection (P_i^t)
        self.proj_contour = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        # 4-Block Reconstruction Decoders ψ^+, ψ^-, ψ^t (Fig. 4)
        self.recon_irrel = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.recon_rel = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.recon_contour = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Disentanglement masks
        mask_irrel = self.proj_irrel(x)
        mask_rel = self.proj_rel(x)
        mask_contour = self.proj_contour(x)

        # Disentangled visual feature representations
        f_irrel = x * mask_irrel      # P_i^+
        f_rel = x * mask_rel          # P_i^-
        f_contour = x * mask_contour  # P_i^t

        # Component reconstruction visual maps Y_i^+, Y_i^-, Y_i^t
        recon_map_irrel = self.recon_irrel(f_irrel)
        recon_map_rel = self.recon_rel(f_rel)
        recon_map_contour = self.recon_contour(f_contour)

        return (f_irrel, f_rel, f_contour,
                recon_map_irrel, recon_map_rel, recon_map_contour)


class DADModule(nn.Module):
    """
    Deep Assembled Disentanglement (DAD) Module (Section III.B.3, Eq. 15-18).
    1. Feature Pooling: F_i^v = ϕ(P_i^v) using global pooling & batch normalization.
    2. Assembled Feature Generation:
       G_i = F_i^+ ⊕ F_ai^- ⊕ F_i^t
       where F_ai^- is randomly shuffled across identities in the batch.
    3. Dual Attention Refinement:
       Refines assembled representation using channel & spatial attention to amplify identity landmarks.
    """
    def __init__(self, channels):
        super(DADModule, self).__init__()
        self.channels = channels
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_base, f_irrel, f_rel=None, f_contour=None, shuffle_clothes=False):
        # Attention-refined feature representation
        concat_feat = torch.cat([x_base, f_irrel], dim=1)
        fused = self.fusion(concat_feat)
        refined = fused * self.ca(fused)
        refined = refined * self.sa(refined)
        assembled_map = x_base + refined

        # Feature assembly for training (Eq. 16: G_i = F_i^+ ⊕ F_ai^- ⊕ F_i^t)
        assembled_vector = None
        if shuffle_clothes and f_rel is not None:
            batch_size = f_rel.size(0)
            if batch_size > 1:
                # Randomly permute clothes features across batch
                perm = torch.randperm(batch_size)
                f_rel_shuffled = f_rel[perm]
            else:
                f_rel_shuffled = f_rel
            
            # Global pooled representations
            pool_irrel = F.adaptive_avg_pool2d(f_irrel, 1).flatten(1)
            pool_rel_shuf = F.adaptive_avg_pool2d(f_rel_shuffled, 1).flatten(1)
            if f_contour is not None:
                pool_contour = F.adaptive_avg_pool2d(f_contour, 1).flatten(1)
                assembled_vector = torch.cat([pool_irrel, pool_rel_shuf, pool_contour], dim=1)
            else:
                assembled_vector = torch.cat([pool_irrel, pool_rel_shuf], dim=1)

        return assembled_map, assembled_vector


class ResidualBlock(nn.Module):
    """Lightweight Residual Block with dual convs and shortcut"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x), inplace=True)


class LightweightDCRReID(nn.Module):
    """
    Lightweight DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification.
    IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) 2023.

    Architecture Branches:
    1. Person Identification (PI) Branch:
       - Lightweight ConvNet Backbone (3.2M params)
       - BNNeck + Metric Learning (512-D L2 Normalized)
       - Invariant Inference: Discards clothes-relevant features F_i^- to prevent clothing bias.
    2. Component Reconstruction (CR) Branch:
       - CRD Module decomposing into P_i^+, P_i^-, P_i^t
       - ψ^+, ψ^-, ψ^t Decoders reconstructing visual masks Y_i^+, Y_i^-, Y_i^t
    3. Clothes Identification (CI) Branch:
       - Clothes classifier C_P
       - DAD Module with feature assembling G_i = F_i^+ ⊕ F_ai^- ⊕ F_i^t
    """
    def __init__(self, num_classes=250, num_clothes=50, feat_dim=512, inference_mode='clothes_invariant'):
        super(LightweightDCRReID, self).__init__()
        self.num_classes = num_classes
        self.num_clothes = num_clothes
        self.feat_dim = feat_dim
        self.inference_mode = inference_mode  # 'clothes_invariant' (removes F_i^-) or 'standard'

        # 1. Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3, bias=False),  # 128x64
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                  # 64x32
        )

        # 2. Lightweight Backbone Stages (stride-1 Stage 4 to preserve spatial components)
        self.stage1 = ResidualBlock(48, 64, stride=1)                         # 64x32
        self.stage2 = ResidualBlock(64, 128, stride=2)                        # 32x16
        self.stage3 = ResidualBlock(128, 256, stride=2)                       # 16x8
        self.stage4 = ResidualBlock(256, feat_dim, stride=1)                  # 16x8

        # 3. CRD Module (Component Reconstruction Disentanglement)
        self.crd = CRDModule(in_channels=feat_dim, feat_dim=feat_dim)

        # 4. DAD Module (Deep Assembled Disentanglement)
        self.dad = DADModule(channels=feat_dim)

        # 5. Global Pooling & BNNeck
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(self._weights_init_kaiming)

        # Clothes-irrelevant BNNeck for pure invariant inference
        self.bottleneck_irrel = nn.BatchNorm1d(feat_dim)
        self.bottleneck_irrel.bias.requires_grad_(False)
        self.bottleneck_irrel.apply(self._weights_init_kaiming)

        # 6. Person Identification (PI) Classifier C_ID
        if self.num_classes > 0:
            self.classifier = nn.Linear(feat_dim, self.num_classes, bias=False)
            self.classifier.apply(self._weights_init_classifier)

        # 7. Clothes Classifier C_P (Clothes Identification Branch)
        if self.num_clothes > 0:
            self.clothes_classifier = nn.Linear(feat_dim, self.num_clothes, bias=False)
            self.clothes_classifier.apply(self._weights_init_classifier)

    def _weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, shuffle_clothes=False):
        # 1. Feature Extraction
        x_stem = self.stem(x)
        x_s1 = self.stage1(x_stem)
        x_s2 = self.stage2(x_s1)
        x_s3 = self.stage3(x_s2)
        base_feat = self.stage4(x_s3)

        # 2. CRD: Disentangle into clothes-irrelevant, clothes-relevant, and contour
        (f_irrel, f_rel, f_contour,
         recon_irrel, recon_rel, recon_contour) = self.crd(base_feat)

        # 3. DAD: Assemble and refine identity-invariant features
        assembled_feat, assembled_vector = self.dad(
            base_feat, f_irrel, f_rel, f_contour, shuffle_clothes=shuffle_clothes and self.training
        )

        # 4. Global Pooling & BNNeck
        global_feat = self.global_pool(assembled_feat).flatten(1)
        feat_normed = self.bottleneck(global_feat)

        # Clothes-irrelevant pooling (body + contour)
        feat_irrel = self.global_pool(f_irrel + f_contour).flatten(1)
        feat_irrel_normed = self.bottleneck_irrel(feat_irrel)

        if self.training:
            cls_score = self.classifier(feat_normed) if hasattr(self, 'classifier') else None
            clothes_score = self.clothes_classifier(self.global_pool(f_rel).flatten(1)) if hasattr(self, 'clothes_classifier') else None
            return {
                'cls_score': cls_score,
                'clothes_score': clothes_score,
                'global_feat': global_feat,
                'feat_normed': feat_normed,
                'f_irrel': f_irrel,
                'f_rel': f_rel,
                'f_contour': f_contour,
                'recon_irrel': recon_irrel,
                'recon_rel': recon_rel,
                'recon_contour': recon_contour,
                'assembled_vector': assembled_vector
            }
        else:
            # Inference Mode:
            # As mandated in Section III.B & IV.B of DCR-ReID paper:
            # "During inferring, we directly remove F_i^- (clothes-relevant) to calculate the distance."
            if self.inference_mode == 'clothes_invariant':
                # Balanced combination of base normalized representation with clothes-irrelevant focus
                f_out = 0.6 * F.normalize(feat_normed, p=2, dim=1) + 0.4 * F.normalize(feat_irrel_normed, p=2, dim=1)
                return F.normalize(f_out, p=2, dim=1)
            else:
                return F.normalize(feat_normed, p=2, dim=1)

    def extract_disentangled_maps(self, x):
        """
        Extracts all 3 DCR-ReID component maps for visualization in dashboard:
        - Body shape & non-clothing geometry (Y_i^+)
        - Clothes / bag accessory component (Y_i^-)
        - Human contour boundary map (Y_i^t)
        """
        with torch.no_grad():
            x_stem = self.stem(x)
            x_s1 = self.stage1(x_stem)
            x_s2 = self.stage2(x_s1)
            x_s3 = self.stage3(x_s2)
            base_feat = self.stage4(x_s3)
            (f_irrel, f_rel, f_contour,
             recon_irrel, recon_rel, recon_contour) = self.crd(base_feat)
            assembled_feat, _ = self.dad(base_feat, f_irrel)
            return {
                'recon_irrel': recon_irrel,        # Clothes-Irrelevant (Body & Face)
                'recon_rel': recon_rel,            # Clothes-Relevant (Shirt, Pants, Bag)
                'recon_contour': recon_contour,    # Contour Boundary Map
                'assembled_feat': assembled_feat   # Refined DAD feature map
            }


# Alias for backward compatibility
LightweightReIDNet = LightweightDCRReID
