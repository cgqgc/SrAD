from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the Sorensen-Dice coefficient:

    dice = 2 * TP / (2 * TP + FP + FN)

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Check if predictions and targets are binary
    if not np.all(np.logical_or(preds == 0, preds == 1)):
        raise ValueError("Predictions must be binary")
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError("Targets must be binary")

    # Compute Dice
    dice = 2 * np.sum(preds[targets == 1]) / (np.sum(preds) + np.sum(targets))

    return dice


def compute_best_dice(
    preds: np.ndarray,
    targets: np.ndarray,
    # n_thresh: float = 100,
    n_thresh: float = 200,
    num_processes: int = 8,
):
    """
    Compute the best dice score for n_thresh thresholds.

    :param predictions: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param n_thresh: Number of thresholds to check.
    """
    preds, targets = np.array(preds), np.array(targets)

    # thresholds = np.linspace(preds.max(), preds.min(), n_thresh)
    num = preds.size
    step = num // n_thresh
    indices = np.arange(0, num, step)
    thresholds = np.sort(preds.reshape(-1))[indices]

    with Pool(num_processes) as pool:
        fn = partial(_dice_multiprocessing, preds, targets)
        scores = pool.map(fn, thresholds)

    scores = np.stack(scores, 0)
    max_dice = scores.max()
    max_thresh = thresholds[scores.argmax()]
    return max_dice, max_thresh


def _dice_multiprocessing(
    preds: np.ndarray, targets: np.ndarray, threshold: float
) -> float:
    return compute_dice(np.where(preds > threshold, 1, 0), targets)



import warnings
from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r"""Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(
                out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C
            )
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


class AELoss(nn.Module):
    def __init__(self, grad_score=False):
        super(AELoss, self).__init__()
        self.grad_score = grad_score

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']
        # loss_ = net_out['loss_']
        loss = (net_in - x_hat) ** 2
        # if loss_ is not None and not anomaly_score:
        #     loss += 0.001 * loss_

        if anomaly_score:
            if self.grad_score:
                grad = torch.abs(torch.autograd.grad(loss.mean(), net_in)[0])
                return torch.mean(grad, dim=[1], keepdim=True) if keepdim else torch.mean(grad, dim=[1, 2, 3])
            else:
                return torch.mean(loss, dim=[1], keepdim=True) if keepdim else torch.mean(loss, dim=[1, 2, 3])
        else:
            return loss.mean()