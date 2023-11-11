import numpy as np
import torch
import random
from torch import nn as nn
import albumentations as A
import cv2
import math


def to_tensor(sample):
    ct, mask = sample
    ct_t = torch.from_numpy(ct).to(torch.float32)
    mask_t = torch.from_numpy(mask).to(torch.long)
    return ct_t, mask_t


class TransformationTrain:
    def __init__(self):
        self.transform = A.Compose([
            A.Affine(translate_percent=0.03, p=0.5, mode=cv2.BORDER_REPLICATE),
            A.Rotate(limit=90, p=0.5, crop_border=True),
            A.RandomCrop(width=64, height=64),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(blur_limit=3, p=0.5)
        ])

    def __call__(self, ct, mask):
        transformed = self.transform(image=ct.T, mask=mask.T.astype(int))
        ct_t, mask_t = transformed['image'].T, transformed['mask'].T
        ct_t, mask_t = to_tensor((ct_t, mask_t))
        return ct_t, mask_t


class TransformationValidation:

    def __call__(self, ct, mask):
        ct_t, mask_t = to_tensor((ct, mask))
        return ct_t, mask_t
