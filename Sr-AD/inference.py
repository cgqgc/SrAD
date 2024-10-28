import os
from functools import partial
from multiprocessing import Pool

import cv2
import deepinv as dinv
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn import metrics
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataload import BraTSAD, PseADSeg
from networks.unet import UNet
from utils.util import *


def reverse_normalization(img):
    img = img.clamp(-1, 1) * 0.5 + 0.5
    img = img * 255
    img = img.detach().cpu().numpy()
    img = img.astype(np.uint8)
    img = img.squeeze()
    img = Image.fromarray(img)
    return img


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

    tn, fp, fn, tp = metrics.confusion_matrix(
        targets.reshape(-1), preds.reshape(-1)
    ).ravel()

    recall = tp / (tp + fn)
    spec = tn / (tn + fp)

    pix_ap = metrics.average_precision_score(targets.reshape(-1), preds.reshape(-1))

    return dice, recall, spec, pix_ap


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
        results = pool.map(fn, thresholds)

    dices, recalls, specs, pix_aps = zip(*results)
    dices = np.stack(dices, 0)
    recalls = np.stack(recalls, 0)
    specs = np.stack(specs, 0)
    pix_aps = np.stack(pix_aps, 0)
    max_dice = dices.max()
    recall = recalls[dices.argmax()]
    spec = specs[dices.argmax()]
    pix_ap = pix_aps[dices.argmax()]

    max_thresh = thresholds[dices.argmax()]

    return max_dice, max_thresh, recall, spec, pix_ap


def _dice_multiprocessing(
    preds: np.ndarray, targets: np.ndarray, threshold: float
) -> float:

    return compute_dice(np.where(preds > threshold, 1, 0), targets)


folds = [0, 1, 2, 3, 4]
datasets = ["BraTS2021"]

in_planes = 1
net = UNet(in_channels=in_planes, n_classes=in_planes).cuda()
criterion = AELoss()

device = torch.device("cuda")
for dataset in datasets:
    for fold in folds:
        print(f"-----------------------------fold {fold}-----------------------------")

        if dataset == "BraTS2021":
            ds = "brats"
        else:
            ds = "pseseg"
        net.load_state_dict(
            torch.load(
                f"/data/qh_20T_share_file/qgc/AnomalyDetection/Sr-AD/Experiment/{ds}/fold_{fold}/checkpoints/bmodel_dice.pt",
                weights_only=True,
                map_location=device,
            )
        )
        net = net.eval()

        normalize = transforms.Normalize((0.5,), (0.5,))
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        data_path = os.path.join("/data/qh_20T_share_file/qgc/dataset/MedAD", dataset)
        if dataset == "BraTS2021":
            data_path = os.path.join("/data/qh_20T_share_file/qgc/dataset/MedAD", dataset)
            test_set = BraTSAD(
                main_path=data_path, img_size=64, transform=test_transform, mode="test"
            )
        else:
            test_set = PseADSeg(
                main_path=data_path, img_size=64, transform=test_transform, mode="test"
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1, shuffle=False, num_workers=0
        )

        refine_anomaly_maps = []
        abnormal_anomaly_maps = []
        abnormal_masks = []

        old_dices = []
        new_dices = []
        old_recalls = []
        new_recalls = []
        old_specs = []
        new_specs = []
        old_pixaps = []
        new_pixaps = []
        names = []

        i = 0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                img = batch["img"]
                mask = batch["mask"]
                name = batch["name"]
                if np.sum(mask.numpy()) == 0:
                    continue
                names.append(name[0])
                img = img.cuda()
                net_out = net(img)
                x_hat = net_out["x_hat"]

                loss = (img - x_hat) ** 2

                anomaly_score_map = (
                    torch.mean(loss, dim=[1], keepdim=True).detach().cpu()
                )

                best_dice, best_thresh, recall, spec, pix_ap = compute_best_dice(
                    anomaly_score_map.squeeze().squeeze().numpy(),
                    mask.detach().squeeze().cpu().numpy(),
                )

                old_dices.append(best_dice)
                old_recalls.append(recall)
                old_specs.append(spec)
                old_pixaps.append(pix_ap)

                a = np.where(anomaly_score_map > best_thresh, 1, 0)

                abnormal_anomaly_maps.append(anomaly_score_map)

                img1 = img.detach().cpu()

                normal_tissue_map_ = np.where(a, -1, img1)
                normal_tissue_map = torch.from_numpy(normal_tissue_map_).cuda().float()

                lesion_map_ = np.where(a, img1, -1)
                lesion_map = torch.from_numpy(lesion_map_).cuda().float()

                lesion_hat = net(lesion_map)["x_hat"]

                lesion_hat = torch.nn.functional.tanh(lesion_hat)

                new_img = lesion_hat + normal_tissue_map
                new_img = torch.nn.functional.tanh(new_img)

                loss_ = (img - new_img) ** 2

                residual_ = torch.mean(loss_, dim=[1], keepdim=True).detach().cpu()

                dice, thresh, new_recall, new_spec, new_pix_ap = compute_best_dice(
                    residual_.squeeze().squeeze().numpy(),
                    mask.squeeze().detach().cpu().numpy(),
                )

                b = np.where(residual_ > thresh, 1, 0)

                abnormal_masks.append(mask)
                if best_dice > dice:
                    refine_anomaly_maps.append(anomaly_score_map)
                    new_dices.append(best_dice)
                    new_recalls.append(recall)
                    new_specs.append(spec)
                    new_pixaps.append(pix_ap)

                else:
                    refine_anomaly_maps.append(residual_)
                    new_dices.append(dice)
                    new_recalls.append(new_recall)
                    new_specs.append(new_spec)
                    new_pixaps.append(new_pix_ap)

        average_old_dice = sum(old_dices) / len(abnormal_masks)
        average_new_dice = sum(new_dices) / len(abnormal_masks)
        average_old_recall = sum(old_recalls) / len(abnormal_masks)
        average_new_recall = sum(new_recalls) / len(abnormal_masks)
        average_old_spec = sum(old_specs) / len(abnormal_masks)
        average_new_spec = sum(new_specs) / len(abnormal_masks)
        average_old_pixap = sum(old_pixaps) / len(abnormal_masks)
        average_new_pixap = sum(new_pixaps) / len(abnormal_masks)

        olds = {
            "names": names,
            "dices": old_dices,
            "recalls": old_recalls,
            "specs": old_specs,
            "pixaps": old_pixaps,
        }
        news = {
            "names": names,
            "dices": new_dices,
            "recalls": new_recalls,
            "specs": new_specs,
            "pixaps": new_pixaps,
        }

        print(f"old_dice: {average_old_dice}")
        print(f"new_dice: {average_new_dice}")
        print(f"old_recall: {average_old_recall}")
        print(f"new_recall: {average_new_recall}")
        print(f"old_spec: {average_old_spec}")
        print(f"new_spec: {average_new_spec}")
        print(f"old_pixap: {average_old_pixap}")
        print(f"new_pixap: {average_new_pixap}")
