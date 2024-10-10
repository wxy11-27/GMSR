import torch
import torch.nn as nn
import math

class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        # if self.bn is not None:
        #     x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

##########################################################################################
class ChannelAttention(nn.Module):
    def __init__(self, Channel_nums):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma = 2
        self.b = 1
        self.k = self.get_kernel_num(Channel_nums)
        self.conv1d = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k // 2)
        self.sigmoid = nn.Sigmoid()

    def get_kernel_num(self, C):  # odd|t|最近奇数
        t = math.log2(C) / self.gamma + self.b / self.gamma
        floor = math.floor(t)
        k = floor + (1 - floor % 2)
        return k

    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max
        F_add_ = F_add.squeeze(-1).permute(0, 2, 1)
        F_add_ = self.conv1d(F_add_).permute(0, 2, 1).unsqueeze(-1)
        out = self.sigmoid(F_add_)
        return out

##########################################################################################
class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.ChannelAttention = ChannelAttention(gate_channels)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        x_out31 = self.ChannelAttention(x)*x
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 4) * (x_out + x_out11 + x_out21 + x_out31)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out