"""
Paper-Faithful DCR-ReID Loss Suite
Reference: Zhenyu Cui et al., "DCR-ReID: Deep Component Reconstruction for Cloth-Changing Person Re-Identification",
IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2023.

Two-Stage Optimization Strategy:
Stage 1 (Eq. 11, 5, 10):
  L_stage1 = L_c + L_ID + L_R
Stage 2 (Eq. 19-20):
  L = L_ID + L_C + L_R
  where L_C = L_c + L_ca + α * L_ac + γ * L'_ID
  PRCC & CCVID: α = 0.05, γ = 1.0
  LTCC: α = 0.05, γ = 0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing (Eq. 5-6 in DCR-ReID).
    """
    def __init__(self, num_classes, epsilon=0.1):
        super(LabelSmoothCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        if inputs is None or targets is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')
        log_probs = self.logsoftmax(inputs)
        targets_smooth = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
        targets_smooth = (1.0 - self.epsilon) * targets_smooth + self.epsilon / self.num_classes
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss


class ComponentReconstructionLoss(nn.Module):
    """
    L1 Component Reconstruction Loss L_R (Eq. 10 in DCR-ReID paper):
    L_R = (1/N) * sum_{i=1}^N sum_{v in {-, +, t}} |T_i^v - Y_i^v|_1
    """
    def __init__(self):
        super(ComponentReconstructionLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, y_plus, y_minus, y_contour, t_plus, t_minus, t_contour):
        loss = torch.tensor(0.0, device=y_plus.device)
        count = 0

        for y, t in [(y_plus, t_plus), (y_minus, t_minus), (y_contour, t_contour)]:
            if y is not None and t is not None:
                if t.shape[-2:] != y.shape[-2:]:
                    t = F.interpolate(t, size=y.shape[-2:], mode='nearest')
                # If target is [0, 1] and prediction from Tanh is [-1, 1], map target to [-1, 1]
                if t.min() >= 0.0:
                    t = 2.0 * t - 1.0
                loss = loss + self.l1(y, t)
                count += 1

        return loss / max(1, count)


class ClothesAdversarialLoss(nn.Module):
    """
    Clothes Adversarial Loss L_ca (Eq. 13-14 in DCR-ReID paper):
    q(c) = 1 - ε + ε/K  for c = c_i
    q(c) = ε/K          for c != c_i and id(c) == id(i)
    q(c) = 0            for id(c) != id(i)
    where K is the number of clothes categories under id(i), ε = 0.1.
    Denominator in Eq. 13 normalizes over u(x_i, c) + sum_{id(j) != id(i)} u(x_i, j).
    """
    def __init__(self, num_clothes, epsilon=0.1):
        super(ClothesAdversarialLoss, self).__init__()
        self.num_clothes = num_clothes
        self.epsilon = epsilon

    def forward(self, clothes_logits, clothes_labels, pids, pid_to_clothes_map=None):
        if clothes_logits is None or clothes_labels is None:
            return torch.tensor(0.0, device=clothes_logits.device if clothes_logits is not None else 'cpu')

        batch_size = clothes_logits.size(0)
        probs = F.softmax(clothes_logits, dim=1)  # u(x_i, c)

        # Build q(c) target distributions per sample
        q_targets = torch.zeros_like(probs)
        losses = []

        for i in range(batch_size):
            ci = clothes_labels[i].item()
            pid_i = pids[i].item()

            if pid_to_clothes_map is not None and pid_i in pid_to_clothes_map:
                intra_clothes = list(pid_to_clothes_map[pid_i])
            else:
                # Default PRCC condition: exactly 2 clothing classes per identity
                # (c_same = pid*2, c_diff = pid*2 + 1)
                c_base = pid_i * 2
                intra_clothes = [c_base, c_base + 1]

            K = max(1, len(intra_clothes))

            # Identify valid intra clothes indices within num_clothes
            valid_intra = [c for c in intra_clothes if c < self.num_clothes]
            intra_mask = torch.zeros(self.num_clothes, dtype=torch.bool, device=clothes_logits.device)
            for c in valid_intra:
                intra_mask[c] = True

            # Total probability mass on intra-identity clothes
            sum_intra = probs[i][intra_mask].sum()

            # Paper Eq. 13 & 14:
            # L_ca = - sum_{c=1}^{N_c} q(c) * log( u(x_i, c) / (u(x_i, c) + sum_{id(j) != id(i)} u(x_i, j)) )
            # Denominator: u(x_i, c) + (1.0 - sum_intra)
            sample_loss = torch.tensor(0.0, device=clothes_logits.device)
            for c in valid_intra:
                if c == ci:
                    qc = 1.0 - self.epsilon + (self.epsilon / K)
                else:
                    qc = self.epsilon / K

                u_c = probs[i, c]
                denom = torch.clamp(u_c + (1.0 - sum_intra), min=1e-8)
                term = torch.log(torch.clamp(u_c / denom, min=1e-8, max=1.0))
                sample_loss = sample_loss - (qc * term)

            losses.append(sample_loss)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=clothes_logits.device)


class AssembledClothesLoss(nn.Module):
    """
    Assembled Clothes Loss L_ac (Eq. 17-18 in DCR-ReID paper):
    Supervises assembled feature G_i to predict fine-grained clothes belonging to id(i)
    and suppresses clothes categories belonging to other identities:
    h(c) = 1/K for id(c) == id(i), 0 otherwise.
    Denominator: C_P(G_i | c) + sum_{id(j) != id(i)} C_P(G_i | j)
    """
    def __init__(self, num_clothes):
        super(AssembledClothesLoss, self).__init__()
        self.num_clothes = num_clothes

    def forward(self, assembled_clothes_logits, pids, pid_to_clothes_map=None):
        if assembled_clothes_logits is None or pids is None:
            return torch.tensor(0.0, device=assembled_clothes_logits.device if assembled_clothes_logits is not None else 'cpu')

        batch_size = assembled_clothes_logits.size(0)
        probs = F.softmax(assembled_clothes_logits, dim=1)
        losses = []

        for i in range(batch_size):
            pid_i = pids[i].item()
            if pid_to_clothes_map is not None and pid_i in pid_to_clothes_map:
                intra_clothes = list(pid_to_clothes_map[pid_i])
            else:
                c_base = pid_i * 2
                intra_clothes = [c_base, c_base + 1]

            K = max(1, len(intra_clothes))
            valid_intra = [c for c in intra_clothes if c < self.num_clothes]
            intra_mask = torch.zeros(self.num_clothes, dtype=torch.bool, device=assembled_clothes_logits.device)
            for c in valid_intra:
                intra_mask[c] = True

            sum_intra = probs[i][intra_mask].sum()
            sample_loss = torch.tensor(0.0, device=assembled_clothes_logits.device)

            for c in valid_intra:
                hc = 1.0 / K
                u_c = probs[i, c]
                denom = torch.clamp(u_c + (1.0 - sum_intra), min=1e-8)
                term = torch.log(torch.clamp(u_c / denom, min=1e-8, max=1.0))
                sample_loss = sample_loss - (hc * term)

            losses.append(sample_loss)

        return torch.stack(losses).mean()


class PaperDCRReIDCombinedLoss(nn.Module):
    """
    Exact Two-Stage DCR-ReID Loss Suite (Section III.B.4 & IV.B):
    Stage 1:
      L_stage1 = L_c + L_ID + L_R
    Stage 2:
      L = L_ID + L_c + L_ca + α * L_ac + γ * L'_ID + L_R
    """
    def __init__(self, num_classes, num_clothes, alpha=0.05, gamma=1.0, epsilon=0.1, stage=2):
        super(PaperDCRReIDCombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.num_clothes = num_clothes
        self.alpha = alpha
        self.gamma = gamma
        self.stage = stage

        self.loss_id = LabelSmoothCrossEntropy(num_classes=num_classes, epsilon=epsilon)
        self.loss_id_prime = LabelSmoothCrossEntropy(num_classes=num_classes, epsilon=epsilon)
        self.loss_clothes = nn.CrossEntropyLoss()
        self.loss_ca = ClothesAdversarialLoss(num_clothes=num_clothes, epsilon=epsilon)
        self.loss_ac = AssembledClothesLoss(num_clothes=num_clothes)
        self.loss_recon = ComponentReconstructionLoss()

    def forward(self, outputs, pids, clothes_labels,
                target_plus=None, target_minus=None, target_contour=None,
                pid_to_clothes_map=None, stage=None):
        cur_stage = stage if stage is not None else self.stage
        id_logits = outputs['id_logits']
        clothes_logits = outputs['clothes_logits']
        assembled_id_logits = outputs.get('assembled_id_logits')
        assembled_clothes_logits = outputs.get('assembled_clothes_logits')
        y_plus = outputs['y_plus']
        y_minus = outputs['y_minus']
        y_contour = outputs['y_contour']

        # 1. Identity loss L_ID
        l_id = self.loss_id(id_logits, pids)

        # 2. Clothes classification loss L_c
        l_c = self.loss_clothes(clothes_logits, clothes_labels)

        # 3. Component reconstruction loss L_R
        l_r = self.loss_recon(y_plus, y_minus, y_contour, target_plus, target_minus, target_contour)

        if cur_stage == 1:
            total_loss = l_c + l_id + l_r
            return {
                'total_loss': total_loss,
                'l_id': l_id,
                'l_c': l_c,
                'l_r': l_r,
                'l_ca': torch.tensor(0.0, device=pids.device),
                'l_ac': torch.tensor(0.0, device=pids.device),
                'l_id_prime': torch.tensor(0.0, device=pids.device)
            }
        else:
            # Stage 2: L = L_ID + L_c + L_ca + α * L_ac + γ * L'_ID + L_R
            l_ca = self.loss_ca(clothes_logits, clothes_labels, pids, pid_to_clothes_map)
            l_ac = self.loss_ac(assembled_clothes_logits, pids, pid_to_clothes_map) if assembled_clothes_logits is not None else torch.tensor(0.0, device=pids.device)
            l_id_prime = self.loss_id_prime(assembled_id_logits, pids) if assembled_id_logits is not None else torch.tensor(0.0, device=pids.device)

            total_loss = l_id + l_c + l_ca + (self.alpha * l_ac) + (self.gamma * l_id_prime) + l_r
            return {
                'total_loss': total_loss,
                'l_id': l_id,
                'l_c': l_c,
                'l_ca': l_ca,
                'l_ac': l_ac,
                'l_id_prime': l_id_prime,
                'l_r': l_r
            }
