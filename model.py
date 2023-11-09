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
