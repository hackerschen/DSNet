import torch
import torch.nn.functional as F
from torch import nn

from .dwmlp import ChanLayerNorm, PatchModule, DSMEncoder, DSMEncoderMult

__all__ = ['UNext_impr_m']

from timm.models.layers import trunc_normal_
import math

class DSNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, expandOut=8, embed_dims=[16, 24, 40, 80, 112],
                 encoder1_kernel=3, encoder2_kernel=3, encoder3_kernel=3, factor1=4, factor2=4, factor3=4):
        super().__init__()
        self.patch1 = nn.Sequential(
            PatchModule(input_channels, embed_dims[0] // 2, 2),
            nn.Conv2d(embed_dims[0] // 2, embed_dims[0], 2, 2),
            ChanLayerNorm(embed_dims[0]),
        )

        self.patch2 = PatchModule(embed_dims[0], embed_dims[1], 2)
        self.patch3 = PatchModule(embed_dims[1], embed_dims[2], 2)
        self.patch4 = PatchModule(embed_dims[2], embed_dims[3], 2)
        self.patch5 = PatchModule(embed_dims[3], embed_dims[4], 2)

        self.encoder3 = DSMEncoder(embed_dims[2], embed_dims[2], kernel_size=encoder3_kernel, factor=factor3, p=0.)

        self.encoder1 = DSMEncoderMult(embed_dims[0], embed_dims[0], kernel_size=encoder1_kernel, factor=factor1, p=0.)
        self.encoder2 = DSMEncoderMult(embed_dims[1], embed_dims[1], kernel_size=encoder2_kernel, factor=factor2, p=0.)

        factor4 = 4
        self.block1 = DSMEncoder(embed_dims[3], embed_dims[3], kernel_size=3, factor=factor4, p=0.)

        self.block2Before = nn.Conv2d(embed_dims[4], embed_dims[4], kernel_size=3, padding=1, groups=embed_dims[4])
        self.block2 = GlobalSparseAttn(embed_dims[4], attn_drop=0., proj_drop=0.)

        decoder_kernel = 3
        decoder_padding = 1

        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], decoder_kernel, stride=1, padding=decoder_padding)
        self.decoder2 = nn.Conv2d(embed_dims[3], embed_dims[2], decoder_kernel, stride=1, padding=decoder_padding)
        self.decoder3 = nn.Conv2d(embed_dims[2], embed_dims[1], decoder_kernel, stride=1, padding=decoder_padding)
        self.decoder4 = nn.Conv2d(embed_dims[1], embed_dims[0], decoder_kernel, stride=1, padding=decoder_padding)
        self.decoder5 = nn.Conv2d(embed_dims[0], embed_dims[0]//2, decoder_kernel, stride=1, padding=decoder_padding)
        self.decoder8 = nn.Conv2d(embed_dims[0] // 2, 8, decoder_kernel, stride=1, padding=decoder_padding)

        self.dbn1 = ChanLayerNorm(embed_dims[3])
        self.dbn2 = ChanLayerNorm(embed_dims[2])
        self.dbn3 = ChanLayerNorm(embed_dims[1])
        self.dbn4 = ChanLayerNorm(embed_dims[0])
        self.dbn5 = ChanLayerNorm(embed_dims[0]//2)

        self.impr1 = DSA(outW=expandOut, outH=expandOut, p=0.)
        self.impr2 = DSA(outW=expandOut, outH=expandOut, p=0.)

        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        self.LocalProp = nn.ConvTranspose2d(embed_dims[0], 8, 4, stride=4)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = self.patch1(x)
        out = self.encoder1(out)
        t1 = out

        ### Stage 2
        out = self.patch2(out)
        out = self.encoder2(out)
        t2 = out
        ### Stage 3
        out = self.patch3(out)
        out = self.encoder3(out)
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out = self.patch4(out)
        out = self.block1(out)
        out = self.impr1(out)
        t4 = out

        ### Bottleneck

        out = self.patch5(out)
        out = self.block2Before(out) + out
        B, C, H, W = out.shape
        out = out.flatten(2).transpose(1, 2).contiguous()
        out = self.block2(out, H, W)

        out = out.transpose(1, 2).reshape(B, C, H, W).contiguous()

        ### Stage 4
        out = F.gelu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=2, mode='bilinear'))
        out = self.impr2(out)

        ### Stage 3
        out = torch.add(out, t4)
        out = F.gelu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t3)
        out = F.gelu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t2)
        out = F.gelu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=2, mode='bilinear'))
        out = torch.add(out, t1)
        
        out = self.LocalProp(out)

        out = self.final(out)
        return out


class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.2, proj_drop=0.2,  sr_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        sr_ratio = torch.tensor(sr_ratio).item()
        self.sr= sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            self.LocalProp = nn.UpsamplingBilinear2d(scale_factor=(sr_ratio, sr_ratio))
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H:int, W:int):
        B, N, C = x.shape
        if self.sr > 1.:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H/self.sr), int(W/self.sr))
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DSA(nn.Module):
    def __init__(self, r=None, outH=10, outW=10, p=0.2):
        super(DSA, self).__init__()
        N = outW * outH
        r = int(N//2) if r is None else r
        self.outH = outH
        self.outW = outW

        self.adaptive = nn.AdaptiveAvgPool2d((outH, outW))
        self.proj1 = nn.Linear(N, r)
        self.relu = nn.GELU()
        self.norm = nn.LayerNorm(r)
        self.proj2 = nn.Linear(r, N)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        B, C, H, W = x.shape
        input = self.adaptive(x)
        input = input.flatten(2).contiguous()
        input = input
        input = self.proj1(input)
        input = self.norm(input)
        output = self.proj2(input)
        output = self.relu(output)
        output = self.drop(output)

        output = output.reshape(B, C, self.outH, self.outW)
        ration_h, ration_w = H/self.outH, W/self.outW
        output = F.interpolate(output, scale_factor=(ration_h, ration_w), mode='bilinear')
        output = torch.sigmoid(output)
        output = x * output + x
        return output
