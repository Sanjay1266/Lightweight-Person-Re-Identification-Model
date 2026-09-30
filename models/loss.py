"""
Loss Functions for DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification
Base Paper: Zhenyu Cui, Jiahuan Zhou, Yuxin Peng, Shiliang Zhang, Yaowei Wang (IEEE TCSVT 2023)

Mathematical Formulation:
1. L_ID (Eq. 5-6): Identity classification loss with label smoothing (epsilon = 0.1).
2. L_triplet: Hard-mining triplet ranking loss (margin = 0.3).
3. L_R (Eq. 10): Component reconstruction loss across human visual regions:
   L_R = (1/N) * sum_{v in {-, +, t}} L1(T_i^v, Y_i^v)
   where Y_i^+ (body/face), Y_i^- (clothing/accessories), Y_i^t (contour boundary).
4. L_c (Eq. 11-12): Clothes classification loss to learn clothes-relevant features.
5. L_ca (Eq. 13-14): Clothes adversarial loss to specifically learn clothes-irrelevant features.
6. L_ac (Eq. 17-18): Assembled clothes loss on assembled vector G_i.
7. L_ID' (Section III.B.3): Identity loss on assembled feature vector G_i.
8. Total DCR-ReID Loss L = L_ID + L_triplet + L_C + L_R with two-stage optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing (Eq. 5-6 in DCR-ReID).
    Prevents the model from becoming over-confident on non-essential visual tokens.
    """
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        if inputs is None or targets is None:
            return torch.tensor(0.0, device=inputs.device if inputs is not None else 'cpu')
        log_probs = self.logsoftmax(inputs)
        targets_smooth = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_smooth + self.epsilon / self.num_classes
        loss = (-targets_smooth * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """
    Hard-Mining Triplet Loss.
    For each query, finds the hardest positive (same identity, max distance)
    and hardest negative (different identity, min distance).
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        if inputs is None or targets is None:
            return torch.tensor(0.0, device=inputs.device if inputs is not None else 'cpu')
        n = inputs.size(0)
        # Compute pairwise distance matrix
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            pos_mask = mask[i]
            neg_mask = (mask[i] == 0)
            if pos_mask.any():
                dist_ap.append(dist[i][pos_mask].max().unsqueeze(0))
            else:
                dist_ap.append(torch.tensor([0.0], device=inputs.device))

            if neg_mask.any():
                dist_an.append(dist[i][neg_mask].min().unsqueeze(0))
            else:
                dist_an.append(torch.tensor([self.margin], device=inputs.device))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class ComponentReconstructionLoss(nn.Module):
    """
    Component Reconstruction Loss L_R (Eq. 10 in DCR-ReID paper):
    L_R = (1/N) * sum_{i=1}^N sum_{v in {-, +, t}} |T_i^v - Y_i^v|_1
    Supervises reconstruction of:
      1) Body / non-clothing geometry (Y_i^+)
      2) Clothing / accessory region (Y_i^-)
      3) Human contour map (Y_i^t)
    """
    def __init__(self):
        super(ComponentReconstructionLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, recon_irrel, recon_rel, recon_contour=None,
                target_irrel=None, target_rel=None, target_contour=None):
        loss = torch.tensor(0.0, device=recon_irrel.device)
        count = 0

        if target_irrel is not None:
            # Resize target to match reconstruction map resolution if necessary
            if target_irrel.shape[-2:] != recon_irrel.shape[-2:]:
                target_irrel = F.interpolate(target_irrel, size=recon_irrel.shape[-2:], mode='nearest')
            loss = loss + self.l1_loss(recon_irrel, target_irrel)
            count += 1
        else:
            # Self-regularization prior: non-clothing body focuses on central upright structure
            loss = loss + 0.05 * torch.mean(torch.abs(recon_irrel - 0.5))
            count += 1

        if target_rel is not None:
            if target_rel.shape[-2:] != recon_rel.shape[-2:]:
                target_rel = F.interpolate(target_rel, size=recon_rel.shape[-2:], mode='nearest')
            loss = loss + self.l1_loss(recon_rel, target_rel)
            count += 1
        else:
            # Prior: clothes & accessories occupy middle / torso & legs
            loss = loss + 0.05 * torch.mean(torch.abs(recon_rel - 0.5))
            count += 1

        if recon_contour is not None:
            if target_contour is not None:
                if target_contour.shape[-2:] != recon_contour.shape[-2:]:
                    target_contour = F.interpolate(target_contour, size=recon_contour.shape[-2:], mode='nearest')
                loss = loss + self.l1_loss(recon_contour, target_contour)
            else:
                loss = loss + 0.05 * torch.mean(torch.abs(recon_contour - 0.3))
            count += 1

        return loss / max(1, count)


class ClothesAdversarialLoss(nn.Module):
    """
    Clothes Adversarial Loss L_ca (Eq. 13-14 in DCR-ReID paper):
    Forces model to learn clothes-irrelevant features by smoothing clothes predictions
    across categories belonging to the same identity.
    """
    def __init__(self, num_clothes, epsilon=0.1):
        super(ClothesAdversarialLoss, self).__init__()
        self.num_clothes = num_clothes
        self.epsilon = epsilon

    def forward(self, clothes_logits, clothes_labels, pids=None):
        if clothes_logits is None or clothes_labels is None:
            return torch.tensor(0.0, device='cpu')
        log_probs = F.log_softmax(clothes_logits, dim=1)
        # Uniform or identity-grouped distribution for adversarial smoothing
        targets = torch.full_like(log_probs, fill_value=self.epsilon / max(1, self.num_clothes))
        targets.scatter_(1, clothes_labels.unsqueeze(1), 1.0 - self.epsilon + self.epsilon / max(1, self.num_clothes))
        loss = (-targets * log_probs).sum(dim=1).mean()
        return loss


class AssembledClothesLoss(nn.Module):
    """
    Assembled Clothes Loss L_ac (Eq. 17-18 in DCR-ReID paper):
    Supervises assembled features G_i to suppress clothes categories not belonging to id(i).
    """
    def __init__(self, num_clothes):
        super(AssembledClothesLoss, self).__init__()
        self.num_clothes = num_clothes

    def forward(self, assembled_clothes_logits, clothes_labels):
        if assembled_clothes_logits is None or clothes_labels is None:
            return torch.tensor(0.0, device='cpu')
        return F.cross_entropy(assembled_clothes_logits, clothes_labels)


class DCRReIDCombinedLoss(nn.Module):
    """
    Complete Two-Stage DCR-ReID Loss Suite (Section III.B.4 & IV.B):
    Stage 1: L = L_ID + L_triplet + L_c + L_R
    Stage 2: L = L_ID + L_triplet + L_c + L_ca + alpha * L_ac + gamma * L_ID' + L_R
    """
    def __init__(self, num_classes, num_clothes=50, margin=0.3, epsilon=0.1,
                 alpha=0.05, gamma=1.0, stage=2):
        super(DCRReIDCombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.num_clothes = num_clothes
        self.alpha = alpha
        self.gamma = gamma
        self.stage = stage

        # Component Loss Modules
        self.id_loss = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=epsilon)
        self.triplet_loss = TripletLoss(margin=margin)
        self.recon_loss = ComponentReconstructionLoss()
        self.clothes_loss = nn.CrossEntropyLoss()
        self.clothes_adv_loss = ClothesAdversarialLoss(num_clothes=num_clothes, epsilon=epsilon)
        self.assembled_clothes_loss = AssembledClothesLoss(num_clothes=num_clothes)

    def forward(self, outputs, pids, clothes_labels=None,
                target_irrel=None, target_rel=None, target_contour=None):
        """
        Accepts model outputs dict or tuple for backward compatibility.
        """
        # Backward compatibility with tuple: (cls_score, global_feat, _)
        if isinstance(outputs, (list, tuple)):
            cls_score, global_feat = outputs[0], outputs[1]
            ce = self.id_loss(cls_score, pids)
            triplet = self.triplet_loss(global_feat, pids)
            return ce + triplet, ce, triplet

        # Full DCR-ReID multi-loss evaluation
        cls_score = outputs.get('cls_score')
        global_feat = outputs.get('global_feat')
        clothes_score = outputs.get('clothes_score')
        recon_irrel = outputs.get('recon_irrel')
        recon_rel = outputs.get('recon_rel')
        recon_contour = outputs.get('recon_contour')

        # 1. Person Identification Losses
        loss_id = self.id_loss(cls_score, pids)
        loss_triplet = self.triplet_loss(global_feat, pids)

        # 2. Component Reconstruction Loss L_R (Eq. 10)
        loss_recon = self.recon_loss(
            recon_irrel, recon_rel, recon_contour,
            target_irrel, target_rel, target_contour
        )

        # 3. Clothes Identification Losses
        loss_clothes = torch.tensor(0.0, device=pids.device)
        loss_clothes_adv = torch.tensor(0.0, device=pids.device)
        loss_assembled = torch.tensor(0.0, device=pids.device)

        if clothes_score is not None and clothes_labels is not None:
            loss_clothes = self.clothes_loss(clothes_score, clothes_labels)
            if self.stage == 2:
                loss_clothes_adv = self.clothes_adv_loss(clothes_score, clothes_labels, pids)

        # Total Loss Calculation
        if self.stage == 1:
            total_loss = loss_id + loss_triplet + 0.5 * loss_clothes + 0.2 * loss_recon
        else:
            total_loss = (loss_id + loss_triplet + 0.5 * loss_clothes +
                          0.1 * loss_clothes_adv + 0.2 * loss_recon)

        return {
            'total_loss': total_loss,
            'loss_id': loss_id,
            'loss_triplet': loss_triplet,
            'loss_recon': loss_recon,
            'loss_clothes': loss_clothes,
            'loss_clothes_adv': loss_clothes_adv
        }


# Alias for backward compatibility with train.py
CombinedLoss = DCRReIDCombinedLoss
