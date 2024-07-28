import torch.nn as nn
import torch
from torch.nn import Conv2d

from deepinv.models.drunet import conv as drunet_conv


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, bias=False, ksize=7,
                 padding_mode='circular', batch_norm=False, ic=2):
        # ic int : internal size coefficient
        super().__init__()

        self.conv1 = Conv2d(in_channels, in_channels, kernel_size=ksize, groups=in_channels,
                               stride=1, padding=ksize // 2, bias=bias, padding_mode=padding_mode)
        if batch_norm:
            self.BatchNorm = nn.BatchNorm2d(in_channels)
        else:
            self.BatchNorm = nn.Identity()

        self.conv2 = Conv2d(in_channels, ic*in_channels, kernel_size=1, stride=1, padding=0, bias=bias, padding_mode=padding_mode)

        self.nonlin = nn.GELU()
        self.conv3 = Conv2d(ic*in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.BatchNorm(out)
        out = self.nonlin(out)
        out = self.conv3(out)
        return out + x


class Student(torch.nn.Module):
    def __init__(
            self,
            layers=10,
            nc=32,
            cnext_ic=2,
            pretrained=None,
    ):
        super(Student, self).__init__()
        self.nc_drunet = 64
        self.nc = nc

        self.convin = Conv2d(in_channels=4, out_channels=nc, kernel_size=1)
        if nc != self.nc_drunet:
            self.convout0 = Conv2d(in_channels=nc, out_channels=self.nc_drunet, kernel_size=1)
        self.convout = drunet_conv(in_channels=self.nc_drunet, out_channels=3, bias=False, mode="C")

        self.net = torch.nn.Sequential()
        self.internal_out = 0

        for i in range(layers):
            self.net.add_module(f'block_{i}', ConvNextBlock(in_channels=nc, ic=cnext_ic))

        if pretrained is not None:
            ckpt_drunet = torch.load(
                pretrained, map_location=lambda storage, loc: storage
            )
            self.load_state_dict(ckpt_drunet['state_dict'], strict=True)
            self.eval()

    def internal_forward(self, x, sigma):
        if isinstance(sigma, torch.Tensor):
            if sigma.ndim > 0:
                noise_level_map = sigma.view(x.size(0), 1, 1, 1)
                noise_level_map = noise_level_map.expand(-1, 1, x.size(2), x.size(3))
            else:
                noise_level_map = torch.ones(
                    (x.size(0), 1, x.size(2), x.size(3)), device=x.device
                ) * sigma[None, None, None, None].to(x.device)
        else:
            noise_level_map = (
                torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
                * sigma
            )
        x = torch.cat((x, noise_level_map), 1)
        out = self.convin(x)
        out = self.net(out)
        if self.nc_drunet != self.nc:
            out = self.convout0(out)
        return out

    def forward(self, x, sigma, update_parameters=False):
        out = self.internal_forward(x, sigma)
        if self.training and update_parameters:
            self.internal_out = out
        return self.convout(out) #+ x


class Student0(torch.nn.Module):
    def __init__(
            self,
            layers=10,
            nc=32,
            cnext_ic=2,
            pretrained=None,
    ):
        super(Student0, self).__init__()
        self.nc_drunet = 64
        self.nc = nc

        self.convin = Conv2d(in_channels=4, out_channels=nc, kernel_size=1)
        if nc != self.nc_drunet:
            self.convout0 = Conv2d(in_channels=nc, out_channels=self.nc_drunet, kernel_size=1)
        self.convout = Conv2d(in_channels=self.nc_drunet, out_channels=3, kernel_size=1)

        self.net = torch.nn.Sequential()
        self.internal_out = 0

        for i in range(layers):
            self.net.add_module(f'block_{i}', ConvNextBlock(in_channels=nc, ic=cnext_ic))

        if pretrained is not None:
            ckpt_drunet = torch.load(
                pretrained, map_location=lambda storage, loc: storage
            )
            self.load_state_dict(ckpt_drunet['state_dict'], strict=True)
            self.eval()

    def internal_forward(self, x, sigma):
        if isinstance(sigma, torch.Tensor):
            if sigma.ndim > 0:
                noise_level_map = sigma.view(x.size(0), 1, 1, 1)
                noise_level_map = noise_level_map.expand(-1, 1, x.size(2), x.size(3))
            else:
                noise_level_map = torch.ones(
                    (x.size(0), 1, x.size(2), x.size(3)), device=x.device
                ) * sigma[None, None, None, None].to(x.device)
        else:
            noise_level_map = (
                torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
                * sigma
            )
        x = torch.cat((x, noise_level_map), 1)
        out = self.convin(x)
        out = self.net(out)
        if self.nc_drunet != self.nc:
            out = self.convout0(out)
        return out

    def forward(self, x, sigma, update_parameters=False):
        out = self.internal_forward(x, sigma)
        if self.training and update_parameters:
            self.internal_out = out
        return self.convout(out) #+ x