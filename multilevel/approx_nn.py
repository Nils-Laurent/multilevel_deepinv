import deepinv.models.dncnn as dncnn
import torch.nn as nn
import torch
import math
from deepinv.models.unet import BFBatchNorm2d
from torch.nn import Conv2d


class ApproxNN(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            depth=5,
            bias=True,
            nf=64,
            pretrained="download",
            device="cpu",
    ):
        super().__init__()
        self.depth = depth

        in_channels = in_channels + 1  # because of sigma

        self.body = nn.Sequential(
            *[ConvNextBlock(in_channels, in_channels) for n in range(depth)]
        )

        self.m_tail = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=3)

    @staticmethod
    def format_param(x, param):
        if isinstance(param, torch.Tensor):
            if len(param.size()) > 0:
                if x.get_device() > -1:
                    param = param[
                        int(x.get_device() * x.shape[0]) : int(
                            (x.get_device() + 1) * x.shape[0]
                        )
                            ]
                    param_map = param.to(x.device).view(x.size(0), 1, 1, 1)
                else:
                    param_map = param.view(x.size(0), 1, 1, 1).to(x.device)
                param_map = param_map.expand(-1, 1, x.size(2), x.size(3))
            else:
                param = param.item()
                param_map = (
                    torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3))
                    .fill_(param)
                    .to(x.device)
                )
        else:
            param_map = (
                torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3))
                .fill_(param)
                .to(x.device)
            )

        return param_map

    def forward(self, x, sigma):
        x_sigma = self.format_param(x, sigma)  # sigma is constant on a new channel
        out = self.body(x_sigma)
        out = self.m_tail(out)
        return out + x


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode='affine', bias=False, ksize=7,
                 padding_mode='circular', batch_norm=False):
        super().__init__()

        rgb_s = 4  # rgb + sigma

        self.conv1 = Conv2d(in_channels, in_channels, kernel_size=ksize, groups=in_channels,
                               stride=1, padding=ksize // 2, bias=bias, padding_mode=padding_mode)
        if batch_norm:
            self.BatchNorm = BFBatchNorm2d(in_channels, use_bias=bias) if bias else nn.BatchNorm2d(in_channels)
        else:
            self.BatchNorm = nn.Identity()

        self.conv2 = Conv2d(in_channels, rgb_s*in_channels, kernel_size=1, stride=1, padding=0, bias=bias, padding_mode=padding_mode)

        self.nonlin = nn.GELU()
        self.conv3 = Conv2d(rgb_s*in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.BatchNorm(out)
        out = self.nonlin(out)
        out = self.conv3(out)
        return out + x

