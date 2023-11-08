import numpy as np
import torch
import random


class ToTensor(object):
    """Convert arrays to Tensors."""
    def __call__(self, ct, mask):
        return torch.from_numpy(ct).to(torch.float32), torch.from_numpy(mask).to(torch.long)


class RandomCrop(object):
    """picks random 64x64 crop inside original 96x96"""
    def __call__(self, ct_cub, mask):
        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_cropped = ct_cub[:, row_offset:row_offset + 64, col_offset:col_offset + 64]
        mask_cropped = mask[:, row_offset:row_offset + 64, col_offset:col_offset + 64]
        return ct_cropped, mask_cropped


