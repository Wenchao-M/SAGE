# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
from torch import nn
from einops import rearrange


# Kernel size .
# Padding to keep the same dimensions.
# Strides, max pooling, average pooling?

class Masked_conv(nn.Module):
    def __init__(self, in_chan, out_chan, pool_size=2, pool_type='max'):
        super().__init__()
        assert pool_type in ['max', 'avg']
        self.conv = nn.Conv1d(in_chan, out_chan,
                              kernel_size=3, stride=1, padding=(1,))
        if pool_type == 'max':
            self.pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x):
        x = x.permute((0, 2, 1))
        x = self.conv(x)
        x = self.pool(x)
        return x.permute((0, 2, 1))


class Masked_up_conv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_chan, out_chan, 3, 2, 0)

    def forward(self, x):
        x = x.permute((0, 2, 1))
        y = self.conv(x)[..., 1:]
        y = y.permute((0, 2, 1))
        return y
