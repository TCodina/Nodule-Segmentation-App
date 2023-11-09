import numpy as np
import torch
import random
from torch import nn as nn
from torchvision import transforms
import math


def to_tensor(sample):
    ct, mask = sample
    ct_t = torch.from_numpy(ct).to(torch.float32)
    mask_t = torch.from_numpy(mask).to(torch.long)
    return ct_t, mask_t


def random_crop(sample, crop_size=64):
    ct, mask = sample
    row_offset = random.randrange(0, crop_size // 2)
    col_offset = random.randrange(0, crop_size // 2)
    ct_cropped = ct[:, row_offset:row_offset + crop_size, col_offset:col_offset + crop_size]
    mask_cropped = mask[:, row_offset:row_offset + crop_size, col_offset:col_offset + crop_size]
    return ct_cropped, mask_cropped


class TransformationTrain:
    def __init__(self, crop_size=64, flip=True, offset=0.03, scale=0.2, rotate=True, noise=25.0):
        self.crop_size = crop_size
        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def __call__(self, ct, mask):
        # if ct.ndim == 3:  # if batch index not present, add it
        #     ct = np.expand_dims(ct, 0)
        #     mask = np.expand_dims(mask, 0)
        ct_t, mask_t = to_tensor(random_crop((ct, mask), crop_size=self.crop_size))
        return ct_t, mask_t


class TransformationValidation:

    def __call__(self, ct, mask):
        # if ct.ndim == 3:  # if batch index not present, add it
        #     ct = np.expand_dims(ct, 0)
        #     mask = np.expand_dims(mask, 0)
        # to tensor
        ct_t, mask_t = to_tensor((ct, mask))

        return ct_t, mask_t


class Augmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self.build_2d_transformation_matrix()  # transformation matrix which flip, shift, scale and rotate
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)  # dim: [input_g.shape[0], 3, 3]
        transform_t = transform_t.to(input_g.device, torch.float32)  # send transformation to GPU!
        # create affine transformation from 2d transformation
        affine_t = F.affine_grid(transform_t[:, :2], input_g.size(), align_corners=False)
        # apply affine transformation to input and label
        augmented_input_g = F.grid_sample(input_g, affine_t, padding_mode='border', align_corners=False)
        augmented_label_g = F.grid_sample(label_g.to(torch.float32), affine_t, padding_mode='border', align_corners=False)

        # add noise to already transformed input
        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5  # convert mask back to boolean

    def build_2d_transformation_matrix(self):
        """
        build transformation matrix which will be later applied over original samples to produce augmented data.
        Depending on argument values, this transformation will flip, shift, scale and/or rotate 2D data.

        return (torch.Tensor):
            2D transformation matrix
        """

        transform_t = torch.eye(3)  # initialize 3D identity transformation

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[2, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            transform_t @= rotation_t

        return transform_t
