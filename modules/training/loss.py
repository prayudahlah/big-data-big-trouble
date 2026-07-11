import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ClassBalancedLoss(nn.Module):
    def __init__(self, beta=0.999, gamma=2.0, num_classes=3, reduction="mean"):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        self.register_buffer("effective_num", torch.zeros(num_classes))
        self.register_buffer("weights", torch.ones(num_classes))

    def update(self, labels):
        counts = torch.bincount(labels, minlength=self.num_classes).float()
        self.effective_num = 1.0 - self.beta ** counts
        self.weights = (1.0 - self.beta) / (self.effective_num + 1e-8)
        self.weights = self.weights / self.weights.sum() * self.num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        cb_loss = ((1 - pt) ** self.gamma * ce_loss)
        weights = self.weights.to(inputs.device).gather(0, targets)
        cb_loss = cb_loss * weights
        if self.reduction == "mean":
            return cb_loss.mean()
        elif self.reduction == "sum":
            return cb_loss.sum()
        return cb_loss
