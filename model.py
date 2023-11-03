import math
import random

import torch
from torch import nn as nn
import torch.nn.functional as F

from util.unet import UNet  # UNet from https://github.com/jvanvugt/pytorch-unet


# wrapper around the ready-to-use UNet to add batchnorm at beginning and a Sigmoid function at the end.
# On top of that initialize weights in a convenient way.
class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights (and biases) for all UNet submodules (Conv, Linear, etc.) in an efficient way
        """
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu', a=0)
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output


# Take input (tensor) and label (mask) and apply transformations to them to get augmented data.
# TODO: why is this inside this file and as a subclass of Module?
class SegmentationAugmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build2dTransformMatrix()  # transformation matrix which flip, shift, scale and rotate
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

    def _build2dTransformMatrix(self):
        """
        build transformation matrix which will be later applied over original samples to produce augmented data.
        Depending on argument values, this transformation will flip, shift, scale and/or rotate 2D data.

        return (torch.Tensor):
            2D transformation matrix
        """

        transform_t = torch.eye(
            3)  # initialize 3D identity transformation (TODO: 1 extra dimension for the batch number?)

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
