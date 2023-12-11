import math
import torch
from timm.models.layers import trunc_normal_
from torch import nn
import torch.nn.functional as F

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class DSMEncoder(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, p=0.2, factor=4):
        super().__init__()
        hidden = inp // factor
        self.block = Residual(nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size, groups=inp, padding=1), #"same"
        ))
        self.norm = ChanLayerNorm(inp)
        self.mlp1 = nn.Conv2d(inp, hidden, 1)
        self.relu = nn.GELU()
        self.drop = nn.Dropout(p)
        self.mlp2 = nn.Conv2d(hidden, oup, 1)

    def forward(self, x, B=None, H=None, W=None):
        B, w ,h ,c = x.shape
        x = self.block(x)
        x = self.mlp1(x)
        # x = spatial_shift(x)
        # x1 = spatial_shift(x[:, :c//3, :, :])
        # x2 = spatial_shift_2(x[:, c//3:c//3 * 2, :, :])
        # x = torch.cat((x1, x2, x[:, c//3 * 2:, :, :]), dim=1)
        x = self.relu(x)
        x = self.mlp2(x)
        x = self.drop(x)
        return x

class DSMEncoderMult(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, p=0.2, factor=4):
        super().__init__()
        hidden = inp * factor
        self.block = Residual(nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size, groups=inp,  padding=1), #"same"
        ))
        self.norm = ChanLayerNorm(inp)
        self.mlp1 = nn.Conv2d(inp, hidden, 1)
        self.relu = nn.GELU()
        self.drop = nn.Dropout(p)
        self.mlp2 = nn.Conv2d(hidden, oup, 1)

    def forward(self, x, B=None, H=None, W=None):
        # 原来的dw使用
        B, w ,h ,c = x.shape
        x = self.block(x)
        x = self.mlp1(x)
        x1 = spatial_shift(x[:, :c//3, :, :])
        x2 = spatial_shift_2(x[:, c//3:c//3 * 2, :, :])
        x = torch.cat((x1, x2, x[:, c//3 * 2:, :, :]), dim=1)
        x = self.relu(x)
        x = self.mlp2(x)
        x = self.drop(x)
        return x

def spatial_shift(x):
    B, N, H, W = x.shape
    pad = 2 // 2
    shift_size = 2
    xn = F.pad(x, (pad, pad, pad, pad) , "constant", 0)
    xs = torch.chunk(xn, shift_size, 1)
    x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-pad, pad+1))]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, pad, H)
    x_s = torch.narrow(x_cat, 3, pad, W)
    return x_s
    # B, w ,h ,c = x.shape
    # x_1 = torch.roll(x[:, :c//4, :, :], shifts=(2), dims=(2))
    # x_2 = torch.roll(x[:, c//4:c//2, :, :], shifts=(-2), dims=(2))
    # x_3 = torch.roll(x[:, c//2:c*3//4, :, :], shifts=(2), dims=(3))
    # x_4 = torch.roll(x[:, 3*c//4:, :, :], shifts=(-2), dims=(3))
    # return torch.cat((x_1, x_2, x_3, x_4), dim=1)

def spatial_shift_2(x):
    B, N, H, W = x.shape
    pad = 2 // 2
    shift_size = 2
    xn = F.pad(x, (pad, pad, pad, pad) , "constant", 0)
    xs = torch.chunk(xn, shift_size, 1)
    x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-pad, pad+1))]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, pad, H)
    x_s = torch.narrow(x_cat, 3, pad, W)
    return x_s

class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """


    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn",  nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) #+ x

class PatchModule(nn.Module):
    def __init__(self, inp, oup, patch_size):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Conv2d(inp, oup, patch_size, patch_size),
            # ChanLayerNorm(oup),
            # nn.GELU(),
            # nn.BatchNorm2d(oup),
            ChanLayerNorm(oup),
            nn.GELU(),
        )

    def forward(self, x):
        return self.fn(x)
    
def bchw_to_bhwc(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, height, width, channels]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, height, width, channels]
    """
    return input.permute(0, 2, 3, 1)


def bhwc_to_bchw(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, channels, height, width]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, channels, height, width]
    """
    return input.permute(0, 3, 1, 2)