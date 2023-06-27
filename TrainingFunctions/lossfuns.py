import numpy as np
import torch as torch
import torch.nn as nn


def lnarr(arr):
    h, w = arr.shape
    out = np.zeros((h, w))
    for k in range(h):
        for i in range(w):
            out[k, i] = np.log(arr[k, i])
    return out


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth
        )

        return 1 - dice


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets):
        return nn.functional.binary_cross_entropy(inputs, targets)


def diceloss(inputs, targets, smooth=1):
    intersection = (inputs * targets).sum()
    dice = 1 - (2.0 * intersection + smooth) / (
        inputs.sum() + targets.sum() + smooth
    )
    return dice


def bceloss(inputs, targets):
    inputs = inputs.flatten()
    targets = targets.flatten(dtype=int)
    n = len(inputs)
    s = 0
    for k in range(n):
        s = (
            s
            + targets[k] * np.log(inputs[k])
            + (1 - targets[k]) * np.log(1 - inputs[k])
        )
    return (-1 / n) * s


def dicebcelossnp(inputs, targets, smooth=1):
    return diceloss(inputs, targets, smooth) + bceloss(inputs, targets)


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        dice = diceloss(inputs, targets, smooth)
        bce = nn.functional.binary_cross_entropy(inputs, targets)
        return bce + dice
