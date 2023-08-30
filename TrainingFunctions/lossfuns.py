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


def accuracy(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    return (inputs == targets).sum() / targets.size(0)


def diceloss(inputs: torch.Tensor, targets: torch.Tensor, smooth=1):
    intersection = (inputs * targets).sum()
    dice = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return dice


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        return diceloss(inputs, targets, smooth)


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets):
        return nn.functional.binary_cross_entropy(inputs, targets)


def dicebcelossnp(inputs: torch.Tensor, targets: torch.Tensor, smooth=1):
    return diceloss(inputs, targets, smooth) + nn.functional.binary_cross_entropy(
        inputs, targets
    )


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        return dicebcelossnp(inputs, targets, smooth)


def tverskyloss(
    inputs: torch.Tensor, targets: torch.Tensor, smooth=1, alpha=0.5, beta=0.5
):
    tp = (inputs * targets).sum()
    fp = (inputs * (1 - targets)).sum()
    fn = ((1 - inputs) * targets).sum()
    return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


class TverskyLoss(nn.Module):
    def __init__(self):
        super(TverskyLoss, self).__init()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        return tverskyloss(inputs, targets, smooth, alpha, beta)
