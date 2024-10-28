import os
import time

import pandas as pd
from PIL import Image
from torch.utils import data
import json
from joblib import Parallel, delayed
import numpy as np
from torchvision import transforms
import torch
import glob
import SimpleITK as sitk
from scipy import ndimage
import nibabel as nib
import cv2
from copy import deepcopy
from PIL import Image, ImageOps, ImageFilter
import random

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def add_stransforms(x):
    '''
        Apply randomly strong transforms to img x.
    '''
    x_s = deepcopy(x)

    if random.random() < 0.8:
        x_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(x_s)
    x_s = transforms.RandomGrayscale(p=0.2)(x_s)
    x_s = blur(x_s, p=0.5)

    return x_s



def parallel_load(img_dir, img_list, img_size, n_channel=1, resample="bilinear", verbose=0):
    # mode = "L" if n_channel == 1 else "RGB"
    mode = "L"
    if resample == "bilinear":
        resample = Image.BILINEAR
    elif resample == "nearest":
        resample = Image.NEAREST
    else:
        raise Exception
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert(mode).resize(
            (img_size, img_size), resample=resample))(file) for file in img_list)

def nii_parallel_load(img_dir, img_list, img_size,resample="bilinear", verbose=0):
    if resample == "bilinear":
        resample = cv2.INTER_LINEAR
    elif resample == "nearest":
        resample = cv2.INTER_NEAREST
    else:
        raise Exception
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: cv2.resize(nib.load(os.path.join(img_dir,file)).get_fdata(),
            (img_size,img_size),interpolation=resample))(file) for file in img_list)



class BraTSAD(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train"):
        super(BraTSAD, self).__init__()
        assert mode in ["train", "test"]

        self.mode = mode
        self.root = main_path
        self.res = img_size
        self.labels = []
        self.masks = []
        self.img_ids = []
        self.slices = []
        self.transform = transform

        self.random_mask = None

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal_dir = os.path.join(self.root, "test", "normal")
            test_abnormal_dir = os.path.join(self.root, "test", "tumor")
            test_mask_dir = os.path.join(self.root, "test", "annotation")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)
            test_masks = [e.replace("flair", "seg") for e in test_abnormal]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(test_normal_dir, test_normal, img_size)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size)

            self.masks += len(test_normal) * [np.zeros((img_size, img_size))]
            self.masks += parallel_load(test_mask_dir, test_masks, img_size, resample="nearest")  # 0/255

            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.mode == "train":
            if self.random_mask is not None:
                img_masked = self.random_mask(img)
                return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
            else:
                img_s = add_stransforms(img)
                img = self.transform(img)
                img_s = self.transform(img_s)
                return {'img': img, 'img_s': img_s, 'label': label, 'name': img_id}
        else:
            mask = np.array(self.masks[index])
            mask = (mask > 0).astype(np.uint8)
            img = self.transform(img)
            return {'img': img, 'label': label, 'name': img_id, 'mask': mask}

    def __len__(self):
        return len(self.slices)

class BUSI_AD(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train"):
        super(BUSI_AD, self).__init__()
        assert mode in ["train", "test"]

        self.mode = mode
        self.root = main_path
        self.res = img_size
        self.labels = []
        self.masks = []
        self.img_ids = []
        self.slices = []
        self.transform = transform

        self.random_mask = None

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal_dir = os.path.join(self.root, "test", "normal")
            test_abnormal_dir = os.path.join(self.root, "test", "tumor")
            test_mask_dir = os.path.join(self.root, "test", "annotation")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)
            test_masks = [e for e in test_abnormal]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(test_normal_dir, test_normal, img_size)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size)

            self.masks += len(test_normal) * [np.zeros((img_size, img_size))]
            self.masks += parallel_load(test_mask_dir, test_masks, img_size, resample="nearest")  # 0/255

            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.mode == "train":
            if self.random_mask is not None:
                img_masked = self.random_mask(img)
                return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
            else:
                return {'img': img, 'label': label, 'name': img_id}
        else:
            mask = np.array(self.masks[index])
            mask = (mask > 0).astype(np.uint8)
            return {'img': img, 'label': label, 'name': img_id, 'mask': mask}

    def __len__(self):
        return len(self.slices)

class PseADSeg(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None, mode="train"):
        super(PseADSeg, self).__init__()
        assert mode in ["train", "test"]

        self.mode = mode
        self.root = main_path
        self.res = img_size
        self.labels = []
        self.masks = []
        self.img_ids = []
        self.slices = []
        self.transform = transform
        self.random_mask = None

        print("Loading images")
        if mode == "train":
            data_dir = os.path.join(self.root, "train")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Loaded {} normal images, {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # test
            test_normal_dir = os.path.join(self.root, "test", "normal")
            test_abnormal_dir = os.path.join(self.root, "test", "abnormal")
            test_mask_dir = os.path.join(self.root, "test", "annotation")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)
            test_masks = [e.replace("__", "_Merge_") for e in test_abnormal]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            self.slices += parallel_load(test_normal_dir, test_normal, img_size)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size)

            self.masks += len(test_normal) * [np.zeros((img_size, img_size))]
            self.masks += parallel_load(test_mask_dir, test_masks, img_size, resample="nearest")  # 0/255

            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images. {:.3f}s".format(len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.mode == "train":
            if self.random_mask is not None:
                img_masked = self.random_mask(img)
                return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
            else:
                return {'img': img, 'label': label, 'name': img_id}
        else:
            mask = np.array(self.masks[index])
            mask = (mask > 0).astype(np.uint8)
            return {'img': img, 'label': label, 'name': img_id, 'mask': mask}

    def __len__(self):
        return len(self.slices)
    
