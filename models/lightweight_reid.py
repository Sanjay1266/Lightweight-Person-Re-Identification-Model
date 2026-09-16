import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation / Channel Attention submodule"""
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


class CRDModule(nn.Module):
    """
    Component Reconstruction Disentanglement (CRD) Module.
    Based on DCR-ReID (Deep Component Reconstruction for Cloth-Changing Person Re-Identification).
    Disentangles feature representation into:
      1) Clothes-Irrelevant features (f_irrel): Body shape, contour, facial structure, height ratio
      2) Clothes-Relevant features (f_rel): Shirt, pants, color, texture, bag accessories
    """
    def __init__(self, in_channels, feat_dim=512):
        super(CRDModule, self).__init__()
        self.in_channels = in_channels
        self.feat_dim = feat_dim
        
        # Clothes-Irrelevant Disentanglement Projection (Contour & Body Shape)
        self.proj_irrel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        # Clothes-Relevant Disentanglement Projection (Clothing / Accessory / Bag patterns)
        self.proj_rel = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        # Component Reconstruction Decoder heads (reconstructs binary body contour vs clothing regions)
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

    def forward(self, x):
        # Disentanglement masks
        mask_irrel = self.proj_irrel(x)
        mask_rel = self.proj_rel(x)

        f_irrel = x * mask_irrel
        f_rel = x * mask_rel

        # Component reconstruction maps (Body Contour vs Clothing/Accessory region)
        recon_map_irrel = self.recon_irrel(f_irrel)
        recon_map_rel = self.recon_rel(f_rel)

        return f_irrel, f_rel, recon_map_irrel, recon_map_rel


class DADModule(nn.Module):
    """
    Deep Assembled Disentanglement (DAD) Module.
    Dynamically assembles and recalibrates the identity representations.
    Suppresses transient clothes/bag interference and amplifies identity-invariant cues.
    """
    def __init__(self, channels):
        super(DADModule, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_base, f_irrel):
        # Combine base feature with disentangled clothes-irrelevant feature
        concat_feat = torch.cat([x_base, f_irrel], dim=1)
        fused = self.fusion(concat_feat)
        # Apply dual attention to focus on salient identity landmarks
        refined = fused * self.ca(fused)
        refined = refined * self.sa(refined)
        return x_base + refined


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
    Lightweight DCR-ReID: Deep Component Reconstruction for Cloth-Changing & Accessory Person Re-Identification.
    
    Architecture (Workflow from presentation 23CSE373 - Computer Vision):
    1. Input Images (256x128)
    2. Image Preprocessing & Augmentations
    3. Lightweight Feature Extraction Backbone (~3.2M params)
    4. CRD Module (Component Reconstruction Disentanglement)
    5. DAD Module (Deep Assembled Disentanglement)
    6. Feature Matching & BNNeck (512-D L2 Normalized)
    7. Performance Evaluation (Rank-1, Rank-5, Rank-10, mAP)
    """
    def __init__(self, num_classes=751, feat_dim=512):
        super(LightweightDCRReID, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # 1. Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3, bias=False),  # 128x64
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                  # 64x32
        )

        # 2. Lightweight Backbone Stages
        self.stage1 = ResidualBlock(48, 64, stride=1)                         # 64x32
        self.stage2 = ResidualBlock(64, 128, stride=2)                        # 32x16
        self.stage3 = ResidualBlock(128, 256, stride=2)                       # 16x8
        self.stage4 = ResidualBlock(256, feat_dim, stride=1)                  # 16x8 (stride 1 preserves spatial resolution)

        # 3. CRD Module (Component Reconstruction Disentanglement)
        self.crd = CRDModule(in_channels=feat_dim, feat_dim=feat_dim)

        # 4. DAD Module (Deep Assembled Disentanglement)
        self.dad = DADModule(channels=feat_dim)

        # 5. Global Pooling & BNNeck
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(self._weights_init_kaiming)

        # 6. Person Identification (PI) Classifier
        if self.num_classes > 0:
            self.classifier = nn.Linear(feat_dim, self.num_classes, bias=False)
            self.classifier.apply(self._weights_init_classifier)

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

    def forward(self, x):
        # Feature Extraction
        x_stem = self.stem(x)
        x_s1 = self.stage1(x_stem)
        x_s2 = self.stage2(x_s1)
        x_s3 = self.stage3(x_s2)
        base_feat = self.stage4(x_s3)

        # CRD: Disentangle into clothes-irrelevant vs clothes-relevant
        f_irrel, f_rel, recon_irrel, recon_rel = self.crd(base_feat)

        # DAD: Assemble and refine identity-invariant features
        assembled_feat = self.dad(base_feat, f_irrel)

        # Global Pooling
        global_feat = self.global_pool(assembled_feat).flatten(1)

        # BNNeck
        feat_normed = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat_normed) if self.num_classes > 0 else None
            return cls_score, global_feat, assembled_feat
        else:
            # Normalized features for fast cosine / Euclidean retrieval
            return F.normalize(feat_normed, p=2, dim=1)

    def extract_disentangled_maps(self, x):
        """
        Extracts component attention maps for visualization in dashboard
        """
        with torch.no_grad():
            x_stem = self.stem(x)
            x_s1 = self.stage1(x_stem)
            x_s2 = self.stage2(x_s1)
            x_s3 = self.stage3(x_s2)
            base_feat = self.stage4(x_s3)
            f_irrel, f_rel, recon_irrel, recon_rel = self.crd(base_feat)
            return {
                'recon_irrel': recon_irrel,  # Body shape & contour (clothes-irrelevant)
                'recon_rel': recon_rel       # Clothing / bag region (clothes-relevant)
            }


# Alias for backward compatibility
LightweightReIDNet = LightweightDCRReID
