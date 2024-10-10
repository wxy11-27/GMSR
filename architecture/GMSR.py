from thop import profile
import torch

import torch.nn as nn
from typing import Optional, Callable
from einops import rearrange, repeat
import math
import numpy as np
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
import torch.fft as fft
import torch.nn.functional as F
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError


        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
################################################

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input):
        input = input.permute(0, 2, 3, 1)
        B,H,W,C = input.shape
        x = self.ln_1(input)
        x = self.drop_path(self.self_attention(x))
        x = x.view(B, H, W, C).contiguous().permute(0, 3, 1, 2)
        return x
class SpectralAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels,1, bias=False),
            nn.Sigmoid()
        )
    def compute_spectral_gradient(self, spectral_image):
        # spectral_gradient = torch.cat([torch.gradient(spectral_image[:, i], dim=1, edge_order=2).unsqueeze(1) for i in range(spectral_image.shape[1])], dim=1)
        # spectral_gradient_norm = torch.norm(spectral_gradient, p=2, dim=1, keepdim=True)
        spectral_gradients = spectral_image[:, 1:, :, :] - spectral_image[:, :-1, :, :]
        normalized_gradient = (spectral_gradients - spectral_gradients.min()) / (
                    spectral_gradients.max() - spectral_gradients.min() + 1e-8)
        return normalized_gradient

    def forward(self, spectral_image):
        spectrum_gradient_map = self.compute_spectral_gradient(spectral_image)
        last_channel = spectrum_gradient_map[:, -1:, :, :]
        gradient_expanded = torch.cat((spectrum_gradient_map, last_channel), dim=1)
        # 将梯度值归一化到0到1之间
        avg_out = self.fc(self.avg_pool(gradient_expanded))
        max_out = self.fc(self.max_pool(gradient_expanded))
        out = avg_out + max_out
        gradient_normalized = torch.sigmoid(out)
        enhanced_hyperspectral = spectral_image * gradient_normalized
        return enhanced_hyperspectral
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7 ):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算空间维度上的梯度
        # 水平方向上的梯度
        gradient_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        gradient_x = F.pad(gradient_x, (0, 1, 0, 0), "reflect")
        # 垂直方向上的梯度
        gradient_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        gradient_y = F.pad(gradient_y, (0, 0, 0, 1), "reflect")
        # 拼接两个方向上的梯度
        spatial_gradients = torch.cat((gradient_x, gradient_y), dim=1)

        min_val = torch.min(spatial_gradients)
        max_val = torch.max(spatial_gradients)
        # 归一化
        normalized_gradients = (spatial_gradients - min_val) / (max_val - min_val)

        avg_out = torch.mean(normalized_gradients, dim=1, keepdim=True)
        max_out, _ = torch.max(normalized_gradients, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv1(attention)
        return x*self.sigmoid(attention)


class GMSR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=31,
                 #dim=48,
                 dim=31,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 decoder=False,
                 ):
        super(GMSR, self).__init__()
        self.output1 = nn.Conv2d(3, 10, kernel_size=1, stride=1, padding=0, bias=bias)
        self.output2 = nn.Conv2d(10, 20, kernel_size=1, stride=1, padding=0, bias=bias)
        self.output3 = nn.Conv2d(20, 31, kernel_size=1, stride=1, padding=0, bias=bias)
        self.VSS1 = VSSBlock(
            hidden_dim=10,
            drop_path=0.1,  # drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=nn.LayerNorm,
            mlp_ratio=2.,  # mlp_ratio,
            d_state=16,
            input_resolution=16,
        )
        self.VSS2 = VSSBlock(
            hidden_dim=20,
            drop_path=0.1,  # drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=nn.LayerNorm,
            mlp_ratio=2.,  # mlp_ratio,
            d_state=16,
            input_resolution=32,
        )
        self.VSS3 = VSSBlock(
            hidden_dim=31,
            drop_path=0.1,  # drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=nn.LayerNorm,
            mlp_ratio=2.,  # mlp_ratio,
            d_state=16,
            input_resolution=64,
        )
        self.spectral_attention1 = SpectralAttention(10)
        self.spectral_attention2 = SpectralAttention(20)
        self.spectral_attention3 = SpectralAttention(31)
        self.spatial_attention = SpatialAttention()
        self.conv1 = nn.Sequential(
            nn.Conv2d(30, 10, 1, 1, 0),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(60, 20, 1, 1, 0),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(93, 31, 1, 1, 0),
            nn.GELU()
        )
    def forward(self,rgb):#(16,16,512)
        a0 = self.output1(rgb)
        a1 = self.VSS1(a0)
        a2 = self.spatial_attention(a0)
        a3 = self.spectral_attention1(a0)
        a = a0+self.conv1(torch.cat([a1, a2, a3], 1))

        b0 = self.output2(a)
        b1 = self.VSS2(b0)
        b2 = self.spatial_attention(b0)
        b3 = self.spectral_attention2(b0)
        b = b0 + self.conv2(torch.cat([b1, b2, b3], 1))

        c0 = self.output3(b)
        c1 = self.VSS3(c0)
        c2 = self.spatial_attention(c0)
        c3 = self.spectral_attention3(c0)
        c = c0 + self.conv3(torch.cat([c1, c2, c3], 1))

        return c


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # input_tensor = torch.rand(6, 3, 256, 256)
    input_tensor = torch.rand(1, 3, 512, 482).cuda()

    model = GMSR().cuda()
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # print(output_tensor.shape)
    macs, params = profile(model, inputs=(input_tensor,))
    print('Parameters number is {}; Flops: {}'.format(params, macs))
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    print(torch.__version__)