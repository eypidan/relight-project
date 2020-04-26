import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class Constant(nn.Module):

    def __init__(self):
        super().__init__()
        self.constant = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        out = self.constant(x)
        return out


