import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from thop import profile

def conv3_3(in_channels, out_channels, pad_mode='zeros', bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                     padding_mode=pad_mode, bias=bias)


def conv1_1(in_channels, out_channels, bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)


class AttentionCorECAModule(nn.Module):
    def __init__(self, k_size=5):
        super(AttentionCorECAModule, self).__init__()
        # self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))  # computing macs for errors
        # self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.gate_h = nn.Conv2d(1, 1, kernel_size=(k_size, 1), padding=(k_size//2, 0), bias=False)
        self.gate_w = nn.Conv2d(1, 1, kernel_size=(1, k_size), padding=(0, k_size//2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_pool_h = torch.mean(x, dim=2, keepdim=True).permute(0, 2, 1, 3)  # b,c,1,w => b,1,c,w
        y_pool_w = torch.mean(x, dim=3, keepdim=True).permute(0, 3, 2, 1)  # b,c,h,1 => b,1,h,c
        y_h = self.gate_h(y_pool_h).permute(0, 2, 1, 3)  # b,1,c,w => b,c,1,w
        y_w = self.gate_w(y_pool_w).permute(0, 3, 2, 1)  # b,1,h,c => b,c,h,1
        y_h = self.sigmoid(y_h)
        y_w = self.sigmoid(y_w)
        out = x * y_h.expand_as(x) * y_w.expand_as(x)
        return out


class PydSpaSpeConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, regroup=False):
        super(PydSpaSpeConv, self).__init__()
        self.bias = bias
        self.regroup = regroup
        if not self.regroup:
            self.spe_conv = conv1_1(in_channels=in_channels, out_channels=out_channels, bias=bias)
            self.spa_conv = conv3_3(in_channels=in_channels, out_channels=out_channels, bias=bias)
            self.spa_spe_conv = nn.Sequential(OrderedDict([
                ('spa', conv3_3(in_channels=in_channels, out_channels=out_channels, bias=bias)),
                ('spe', conv1_1(in_channels=in_channels, out_channels=out_channels, bias=bias))
            ]))
        else:
            self.combine_conv = conv3_3(in_channels=in_channels, out_channels=out_channels, bias=bias)

    def forward(self, x):
        if not self.regroup:
            out1 = self.spe_conv(x)
            out2 = self.spa_conv(x)
            out3 = self.spa_spe_conv(x)
            out = out1 + out2 + out3
        else:
            out = self.combine_conv(x)
        return out

    def perform_regroup(self, ):
        self.regroup = True
        self.combine_conv = conv3_3(in_channels=self.spa_conv.in_channels, out_channels=self.spa_conv.out_channels, bias=self.bias)
        if self.bias:
            regroup_k, regroup_b = self.get_regroup_kernel_bias()
            self.combine_conv.weight.data, self.combine_conv.bias.data = regroup_k, regroup_b
        else:
            regroup_k = self.get_regroup_kernel_bias()
            self.combine_conv.weight.data = regroup_k
        self.__delattr__('spa_conv')
        self.__delattr__('spe_conv')
        self.__delattr__('spa_spe_conv')

    def get_regroup_kernel_bias(self, ):
        spa_k = self.spa_conv.weight.data
        spe_k = self.spe_conv.weight.data
        # spa_spe_conv
        spa3_3_k = self.spa_spe_conv.spa.weight.data
        c1, c0, _, _ = spa3_3_k.size()
        spa3_3_k = spa3_3_k.view(c1, -1).permute(1, 0).contiguous()
        spe1_1_k = self.spa_spe_conv.spe.weight.data
        c2, c1, _, _ = spe1_1_k.size()
        spe1_1_k = spe1_1_k.view(c2, -1).permute(1, 0).contiguous()
        spa_spe_k = torch.matmul(spa3_3_k, spe1_1_k)
        spa_spe_k = spa_spe_k.permute(1, 0).contiguous().view(c2, c0, 3, 3)

        if self.bias:
            spa_b = self.spa_conv.bias.data
            spe_b = self.spe_conv.bias.data
            # spa_spe_conv
            spa3_3_b = self.spa_spe_conv.spa.bias.data.unsqueeze(1)
            spe1_1_k = self.spa_spe_conv.spe.weight.data
            spe1_1_b = self.spa_spe_conv.spe.bias.data
            c2, _, _, _ = spe1_1_k.size()
            spe1_1_k = spe1_1_k.view(c2, -1)
            spa_spe_b = torch.matmul(spe1_1_k, spa3_3_b).squeeze(1) + spe1_1_b

            return spa_k + torch.nn.functional.pad(spe_k, [1, 1, 1, 1]) + spa_spe_k, spa_b+spe_b+spa_spe_b

        return spa_k+torch.nn.functional.pad(spe_k, [1, 1, 1, 1])+spa_spe_k


class SpaSpeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, k_size=5):
        super(SpaSpeBlock, self).__init__()
        self.stage1_conv = PydSpaSpeConv(in_channels=in_channels, out_channels=out_channels, bias=bias)
        self.relu = nn.PReLU()
        self.stage2_conv = PydSpaSpeConv(in_channels=out_channels, out_channels=out_channels, bias=bias)
        self.att_spe = AttentionCorECAModule(k_size=k_size)

    def forward(self, x):
        residual = x
        out = self.stage1_conv(x)
        out = self.relu(out)
        out = self.stage2_conv(out)
        out = self.att_spe(out)
        out += residual
        return out


class RepCPSI(nn.Module):
    def __init__(self, in_channels=3, inter_channels=80, out_channels=31, block2DNum=8, bias=True, k_size=7):
        super(RepCPSI, self).__init__()
        self.in_conv2D = conv3_3(in_channels=in_channels, out_channels=inter_channels, bias=bias)
        self.seq_spaspe = nn.ModuleList(
            [SpaSpeBlock(inter_channels, inter_channels, bias, k_size) for _ in range(block2DNum)])
        self.out_conv2D = conv3_3(in_channels=inter_channels, out_channels=out_channels, bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
        #         m.weight.data.normal_(0, sqrt(2./n))  # the devide  2./n  carefully

    def forward(self, x):
        out = self.in_conv2D(x)
        residual = out
        for _, block in enumerate(self.seq_spaspe):
            out = block(out)
        out += residual
        out = self.out_conv2D(out)
        return out


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # input_tensor = torch.rand(6, 3, 256, 256)
    input_tensor = torch.rand(1, 3, 512, 482).cuda()

    model = RepCPSI().cuda()
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # print(output_tensor.shape)
    macs, params = profile(model, inputs=(input_tensor,))
    print('Parameters number is {}; Flops: {}'.format(params, macs))
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    print(torch.__version__)