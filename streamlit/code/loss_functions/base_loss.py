import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CustomBCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, predictions, targets):
        return self.loss(predictions, targets)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_exp = torch.exp(-bce)
        loss = self.alpha * (1 - bce_exp) ** self.gamma * bce
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., **kwargs):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions).contiguous()
        targets = targets.contiguous()
        intersection = (predictions * targets).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) /
                     (predictions.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + self.smooth)))
        return loss.mean()

class IOULoss(nn.Module):
    def __init__(self, smooth=1., **kwargs):
        super(IOULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1., **kwargs):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce_with_logits = nn.BCEWithLogitsLoss(**kwargs)
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, predictions, targets):
        bce = self.bce_with_logits(predictions, targets)
        predictions = torch.sigmoid(predictions)
        dice = self.dice_loss(predictions, targets)
        return bce * self.bce_weight + dice * (1 - self.bce_weight)
