import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    Prevents the model from becoming over-confident on non-essential visual tokens (like bags).
    """
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """
    Hard-Mining Triplet Loss.
    For each query, finds the hardest positive (same identity, max distance) and hardest negative (different identity, min distance).
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance matrix
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, margin=0.3, epsilon=0.1):
        super(CombinedLoss, self).__init__()
        self.ce_loss = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=epsilon)
        self.triplet_loss = TripletLoss(margin=margin)

    def forward(self, cls_score, global_feat, targets):
        ce = self.ce_loss(cls_score, targets)
        triplet = self.triplet_loss(global_feat, targets)
        return ce + triplet, ce, triplet
