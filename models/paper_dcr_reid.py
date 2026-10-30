"""
Paper-Faithful Implementation of DCR-ReID (Deep Component Reconstruction Re-ID)
Reference: Zhenyu Cui, Jiahuan Zhou, Yuxin Peng, Shiliang Zhang, Yaowei Wang,
"DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification",
IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol. 33, No. 8, August 2023.

Three Coordinated Branches:
1. Person Identification (PI) Branch:
   - ResNet-50 backbone (ImageNet-pretrained, layer4 stride-1)
   - Stacked max+avg pooling + BatchNorm1d
   - Identity Classifier C_ID with Label-Smoothed CE Loss L_ID
2. Component Reconstruction Disentanglement (CRD) Branch:
   - 1x1 Conv projecting 2048 -> 2560 channels
   - Channel decomposition: P_i = P_i^+ (1024) ⊕ P_i^- (512) ⊕ P_i^t (1024)
   - Three 4-block residual decoders (ψ^+, ψ^-, ψ^t) with Tanh activation
   - L1 component reconstruction loss L_R against binary target masks (Eq. 10)
3. Clothes Identification (CI) Branch & Deep Assembled Disentanglement (DAD) Module:
   - Clothes Classifier C_P with clothes classification loss L_c (Eq. 11-12)
   - Clothes Adversarial Loss L_ca over intra-identity clothes categories (Eq. 13-14)
   - DAD feature assembly: G_i = F_i^+ ⊕ F_ai^- ⊕ F_i^t (Eq. 16)
   - Assembled clothes loss L_ac (Eq. 17-18) and assembled identity loss L'_ID
   - Two-Stage Optimization:
     * Stage 1: L_c + L_ID + L_R
     * Stage 2: L_ID + L_c + L_ca + α * L_ac + γ * L'_ID + L_R
       (PRCC: α=0.05, γ=1.0; LTCC: α=0.05, γ=0.0)
4. Inference:
   - Discards clothes-relevant features F_i^-
   - Returns L2-normalized F_i^+ ⊕ F_i^t (4096 dimensions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ReconstructionBlock(nn.Module):
    """
    Reconstruction Block for ψ^v decoders (Fig. 4 in DCR-ReID paper).
    Consists of Conv2d + BatchNorm2d + ReLU + Conv2d + BatchNorm2d + Residual shortcut + ReLU.
    """
    def __init__(self, channels):
        super(ReconstructionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class ComponentReconstructionDecoder(nn.Module):
    """
    Component Reconstruction Network ψ^v (Fig. 4 in DCR-ReID paper):
    1. Projection layer (Conv2d 3x3 + BatchNorm2d + ReLU)
    2. Four residual reconstruction blocks
    3. Final 1x1 Conv + Tanh activation
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
            nn.Tanh()
        )

    def forward(self, x):
        h = self.proj(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        return self.out_conv(h)


class StackedGlobalPooling(nn.Module):
    """
    Stacked Max-Pooling and Average-Pooling φ(P_i^v) as specified in Section IV.B:
    "Global pooling is the stack of max-pooling and average-pooling."
    Followed by BatchNorm1d.
    """
    def __init__(self, in_channels):
        super(StackedGlobalPooling, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.bn = nn.BatchNorm1d(in_channels * 2)
        self.bn.bias.requires_grad_(False)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x):
        avg_out = self.avg_pool(x).flatten(1)
        max_out = self.max_pool(x).flatten(1)
        pooled = torch.cat([avg_out, max_out], dim=1)
        return self.bn(pooled)


class PaperDCRReID(nn.Module):
    """
    Paper-Faithful DCR-ReID Architecture (IEEE TCSVT 2023).
    Backbone: ImageNet-pretrained ResNet-50 with layer4 stride=1.
    Input Size: 384 x 192 (or 256 x 128 adaptable).
    Feature Map Channels: P_i^+ (1024), P_i^- (512), P_i^t (1024) -> Total 2560.
    Inference Feature: F_i^+ ⊕ F_i^t (4096 dimensions, L2 normalized).
    """
    def __init__(self, num_classes=150, num_clothes=300, pretrained=True, num_identities=None):
        super(PaperDCRReID, self).__init__()
        if num_identities is not None:
            num_classes = num_identities
        self.num_classes = num_classes
        self.num_clothes = num_clothes

        # 1. ResNet-50 Backbone
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
        else:
            resnet = models.resnet50(weights=None)

        # Modify layer4 to have stride-1 (standard person Re-ID practice and DCR-ReID specification)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 2. CRD Channel Projection: 2048 -> 2560 channels
        # (Page 10: "We use a convolutional layer with a kernel size of 1x1 to achieve the transformation")
        self.crd_proj = nn.Sequential(
            nn.Conv2d(2048, 2560, kernel_size=1, bias=False),
            nn.BatchNorm2d(2560),
            nn.ReLU(inplace=True)
        )

        # 3. Component Reconstruction Decoders ψ^+, ψ^-, ψ^t (Fig. 4)
        self.psi_plus = ComponentReconstructionDecoder(in_channels=1024, mid_channels=64)    # Body / Non-clothes
        self.psi_minus = ComponentReconstructionDecoder(in_channels=512, mid_channels=64)    # Clothes
        self.psi_contour = ComponentReconstructionDecoder(in_channels=1024, mid_channels=64) # Contour

        # 4. Stacked Global Pooling φ(P_i^v) (Section IV.B)
        self.pool_plus = StackedGlobalPooling(1024)    # 2048-D
        self.pool_minus = StackedGlobalPooling(512)     # 1024-D
        self.pool_contour = StackedGlobalPooling(1024)  # 2048-D

        # 5. Classifiers
        # Identity Classifier C_ID on clothes-irrelevant features F_i^+ (or F_i^+ ⊕ F_i^t)
        self.classifier_id = nn.Linear(2048, num_classes, bias=False)
        nn.init.normal_(self.classifier_id.weight, std=0.001)

        # Clothes Classifier C_P on clothes-relevant features F_i^- (1024-D)
        self.classifier_clothes = nn.Linear(1024, num_clothes, bias=False)
        nn.init.normal_(self.classifier_clothes.weight, std=0.001)

        # DAD Assembled Classifiers on G_i (2048 + 1024 + 2048 = 5120-D)
        self.classifier_assembled_id = nn.Linear(5120, num_classes, bias=False)
        nn.init.normal_(self.classifier_assembled_id.weight, std=0.001)

        self.classifier_assembled_clothes = nn.Linear(5120, num_clothes, bias=False)
        nn.init.normal_(self.classifier_assembled_clothes.weight, std=0.001)

    def extract_backbone_features(self, x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return h

    def forward(self, x, pids=None, shuffle_clothes=False):
        # 1. Feature Extraction through ResNet-50
        feat_2048 = self.extract_backbone_features(x)

        # 2. CRD Channel Projection & Splitting (Eq. 7-8)
        # P_i ∈ R^{2560 x H x W}
        p_i = self.crd_proj(feat_2048)
        p_plus = p_i[:, 0:1024, :, :]       # P_i^+ (Clothes-irrelevant body shape & anatomy, 1024-D)
        p_minus = p_i[:, 1024:1536, :, :]   # P_i^- (Clothes-relevant apparel features, 512-D)
        p_contour = p_i[:, 1536:2560, :, :] # P_i^t (Human contour boundary features, 1024-D)

        # 3. Component Reconstruction Maps Y_i^v = ψ^v(P_i^v) (Eq. 9, Fig. 4)
        y_plus = self.psi_plus(p_plus)
        y_minus = self.psi_minus(p_minus)
        y_contour = self.psi_contour(p_contour)

        # 4. Stacked Pooling F_i^v = φ(P_i^v) (Eq. 15)
        f_plus = self.pool_plus(p_plus)        # 2048-D
        f_minus = self.pool_minus(p_minus)     # 1024-D
        f_contour = self.pool_contour(p_contour) # 2048-D

        if not self.training:
            # Inference Mode:
            # Section III.B & IV.B: "During inferring, we directly remove F_i^- in PI branch
            # to calculate the distance between the query image and the images in the gallery."
            f_infer = torch.cat([f_plus, f_contour], dim=1)  # 4096-D
            return F.normalize(f_infer, p=2, dim=1)

        # Training Mode:
        # Classifiers
        id_logits = self.classifier_id(f_plus)
        clothes_logits = self.classifier_clothes(f_minus)

        # DAD Feature Assembly: G_i = F_i^+ ⊕ F_ai^- ⊕ F_i^t (Eq. 16)
        if shuffle_clothes and f_minus.size(0) > 1:
            bs = f_minus.size(0)
            if pids is not None:
                # Sample donor instance with different identity (id(ai) != id(i))
                perm = torch.empty(bs, dtype=torch.long, device=x.device)
                for i in range(bs):
                    diff_mask = (pids != pids[i]).nonzero(as_tuple=True)[0]
                    if len(diff_mask) > 0:
                        rand_idx = torch.randint(len(diff_mask), (1,), device=x.device).item()
                        perm[i] = diff_mask[rand_idx]
                    else:
                        perm[i] = (i + 1) % bs
            else:
                # Cyclic roll ensures perm[i] != i
                perm = torch.roll(torch.arange(bs, device=x.device), shifts=1)

            # Detach donor clothes features to prevent cross-identity gradient leakage into donor branch
            f_minus_shuffled = f_minus[perm].detach()
        else:
            f_minus_shuffled = f_minus

        g_i = torch.cat([f_plus, f_minus_shuffled, f_contour], dim=1) # 5120-D
        assembled_id_logits = self.classifier_assembled_id(g_i)
        assembled_clothes_logits = self.classifier_assembled_clothes(g_i)

        return {
            'id_logits': id_logits,
            'clothes_logits': clothes_logits,
            'assembled_id_logits': assembled_id_logits,
            'assembled_clothes_logits': assembled_clothes_logits,
            'p_plus': p_plus,
            'p_minus': p_minus,
            'p_contour': p_contour,
            'y_plus': y_plus,
            'y_minus': y_minus,
            'y_contour': y_contour,
            'f_plus': f_plus,
            'f_minus': f_minus,
            'f_contour': f_contour,
            'g_i': g_i
        }
