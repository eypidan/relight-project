import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.dconv_down1 = double_conv(3, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)
        self.dconv_down5 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        #self.upsample = nn.Upsample(size=,scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv_up4 = double_conv(512, 256)

        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv_up3 = double_conv(256, 128)

        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv_up2 = double_conv(128, 64)

        self.upsample1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dconv_up1 = double_conv(64, 32)

        self.conv_last = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)

        x = self.upsample4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)

        x = self.upsample3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out


