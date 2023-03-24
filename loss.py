# -*- coding: utf-8 -*-
# @Time    : 27/12/2022 4:33 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
from util.dice_score import dice_loss
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
import torch


def lw_loss(pred, gt):
    pred = pred.squeeze()
    gt = gt.squeeze()
    gt = torch.tensor(gt, dtype=torch.int64)

    gt = one_hot(gt, 4)
    # print(gt)
    loss = 0
    criterion = nn.CrossEntropyLoss()
    # for d in range(len(gt)):
    #     # print("nonzero:", torch.nonzero(gt))
    #     w = abs(d - torch.nonzero(gt))
    #     # print("w", w)
    #     print("pred:", pred)
    #     print("gt:", gt)
    #     loss += w * (pred[d] - gt[d])
    # # loss += dice_loss(pred.unsqueeze(dim=0), gt.unsqueeze(dim=0), multiclass=False)
    # epsilon = 1e-6
    # set_inner = 2 * (pred * gt).sum()
    # set_sum = pred.sum() + gt.sum()
    # set_sum = torch.where(set_sum == 0, set_inner, set_sum)
    # dice = (set_inner + epsilon) / (set_sum + epsilon)
    # loss += (1 - dice)

    loss += criterion(pred.float(), gt.float())
    return loss


# Define the Focal Loss function
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        logits = logits.float()  # Convert to torch.float32
        labels = labels.long()  # Convert to torch.int64
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss



def unet_loss(model, masks_pred, true_masks):
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    if model.n_classes == 1:
        loss = criterion(masks_pred.squeeze(1), true_masks.float())
        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
    else:
        loss = criterion(masks_pred, true_masks)
        loss += dice_loss(
            F.softmax(masks_pred, dim=1).float(),
            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
            multiclass=True
        )
    return loss


def unetpp_loss(model, masks_pred, true_masks):
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    loss = 0
    for i in masks_pred:
        if model.n_classes == 1:
            loss += criterion(i.squeeze(1), true_masks.float())
            loss += dice_loss(F.sigmoid(i.squeeze(1)), true_masks.float(), multiclass=False)
        else:
            loss += criterion(i, true_masks)
            loss += dice_loss(
                F.softmax(i, dim=1).float(),
                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
    return loss/len(masks_pred)


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        bn = inputs.shape[0]

        # flatten label and prediction tensors
        inputs = inputs.view(bn, -1)
        targets = targets.view(bn, -1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky
