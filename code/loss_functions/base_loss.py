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
        # binary_cross_entropy_with_logits는 sigmoid를 포함하므로 raw logits을 바로 사용
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
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
    

class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, alpha=0.25, gamma=2, smooth=1.0, **kwargs):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, **kwargs)
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, predictions, targets):
        # Focal Loss 계산
        focal = self.focal_loss(predictions, targets)
        # Dice Loss 계산
        dice = self.dice_loss(predictions, targets)
        # 결합된 손실 반환
        return self.focal_weight * focal + (1 - self.focal_weight) * dice

class FocalIOULoss(nn.Module):
    def __init__(self, focal_alpha=0.25, focal_gamma=2, iou_smooth=1., focal_weight=0.5, iou_weight=0.5):
        """
        Args:
            focal_alpha (float): Focal Loss의 alpha 값.
            focal_gamma (float): Focal Loss의 gamma 값.
            iou_smooth (float): IOU Loss의 smooth 값.
            focal_weight (float): Focal Loss의 가중치.
            iou_weight (float): IOU Loss의 가중치.
        """
        super(FocalIOULoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.iou_loss = IOULoss(smooth=iou_smooth)
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): 모델의 예측값 (logits).
            targets (torch.Tensor): 타겟값 (binary masks).
        """
        focal = self.focal_loss(predictions, targets)
        iou = self.iou_loss(predictions, targets)
        return focal * self.focal_weight + iou * self.iou_weight
