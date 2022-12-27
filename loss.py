# -*- coding: utf-8 -*-
# @Time    : 27/12/2022 4:33 PM
# @Author  : Breeze
# @Email   : breezewrf@gmail.com
from utils.dice_score import dice_loss
from torch import nn
import torch.nn.functional as F


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
