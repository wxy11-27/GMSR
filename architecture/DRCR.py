import torch
from torch import nn
from torch.nn import functional as F
from thop import profile

class AWCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AWCA, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h*w).unsqueeze(1)

        mask = self.conv(x).view(b, 1, h*w)
        mask = self.softmax(mask).unsqueeze(-1)
        y = torch.matmul(input_x, mask).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction = 8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayer_2(nn.Module):
    def __init__(self, channel, reduction = 8):
        super(SELayer_2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y1).view(b, c, 1, 1)
        y2 = self.max_pool(x).view(b, c)
        y2 = self.fc2(y2).view(b, c, 1, 1)
        y = y1 + y2
        return x * y.expand_as(x)


class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, reduction=8, dimension=2, sub_sample=False, bn_layer=False):
        super(NONLocalBlock2D, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // reduction

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0, bias=False)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0, bias=False),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.constant_(self.W.weight, 0)
            # nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=False)
        # self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                    kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        # phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # f = torch.matmul(theta_x, phi_x)
        f = self.count_cov_second(theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def count_cov_second(self, input):
        x = input
        batchSize, dim, M = x.data.shape
        x_mean_band = x.mean(2).view(batchSize, dim, 1).expand(batchSize, dim, M)
        y = (x - x_mean_band).bmm(x.transpose(1, 2)) / M
        return y


class PSNL(nn.Module):
    def __init__(self, channels):
        super(PSNL, self).__init__()
        # nonlocal module
        self.non_local = NONLocalBlock2D(channels)

    def forward(self,x):
        # divide feature map into 4 part
        batch_size, C, H, W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]

        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return nonlocal_feat


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            pass
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



class DRAB(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim, k1_size=3, k2_size=1, dilation=1):
        super(DRAB, self).__init__()
        self.conv1 = Conv3x3(in_dim, in_dim, 3, 1)
        self.relu1 = nn.PReLU()
        self.conv2 = Conv3x3(in_dim, in_dim, 3, 1)
        self.relu2 = nn.PReLU()
        # T^{l}_{1}: (conv.)
        self.up_conv = Conv3x3(in_dim, res_dim, kernel_size=k1_size, stride=1, dilation=dilation)
        self.up_relu = nn.PReLU()
        self.se = AWCA(res_dim)
        # T^{l}_{2}: (conv.)
        self.down_conv = Conv3x3(res_dim, out_dim, kernel_size=k2_size, stride=1)
        self.down_relu = nn.PReLU()

    def forward(self, x, res):
        x_r = x
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        x += x_r
        x = self.relu2(x)
        # T^{l}_{1}
        x = self.up_conv(x)
        x += res
        x = self.up_relu(x)
        res = x
        x = self.se(x)
        # T^{l}_{2}
        x = self.down_conv(x)
        x += x_r
        x = self.down_relu(x)
        return x, res



class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, in_channels, latent_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(ResidualDenseBlock_5C, self).__init__()
        # dense convolutions
        self.conv1 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv2 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv3 = Conv2dLayer(in_channels, latent_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv4 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv5 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        self.conv6 = Conv2dLayer(in_channels * 2, in_channels, kernel_size, stride, padding, dilation, pad_type,
                                 activation, norm, sn)
        # self.cspn2_guide = GMLayer(in_channels)
        # self.cspn2 = Affinity_Propagate_Channel()
        self.se1 = SELayer(in_channels)
        self.se2 = SELayer(in_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # guidance2 = self.cspn2_guide(x3)
        # x3_2 = self.cspn2(guidance2, x3)
        x3_2 = self.se1(x)
        x4 = self.conv4(torch.cat((x3, x3_2), 1))
        x5 = self.conv5(torch.cat((x2, x4), 1))
        x6 = self.conv6(torch.cat((x1, x5), 1))+self.se2(x3_2)
        return x6


class AWAN(nn.Module):
    def __init__(self, inplanes=3, planes=31, channels=200, n_DRBs=8):
        super(AWAN, self).__init__()
        # 2D Nets
        self.input_conv2D = Conv3x3(inplanes, channels, 3, 1)
        self.input_prelu2D = nn.PReLU()
        self.head_conv2D = Conv3x3(channels, channels, 3, 1)
        self.denosing = Denoising(channels)

        # self.cspn_guide = Conv2D(channels, 8)
        # self.cspn = Affinity_Propagate_Spatial()

        # self.backbone = nn.ModuleList(
        #     [DRAB(in_dim=channels, out_dim=channels, res_dim=channels, k1_size=5, k2_size=3, dilation=1) for _ in
        #      range(n_DRBs)])
        self.backbone = nn.ModuleList(
            [ResidualDenseBlock_5C(channels, channels) for _ in range(n_DRBs)])
        self.tail_conv2D = Conv3x3(channels, channels, 3, 1)
        self.output_prelu2D = nn.PReLU()
        self.output_conv2D = Conv3x3(channels, planes, 3, 1)
        # self.tail_nonlocal = PSNL(planes)

    def forward(self, x):
        out = self.DRN2D(x)
        return out

    def DRN2D(self, x):
        out = self.input_prelu2D(self.input_conv2D(x))
        out = self.head_conv2D(out)
        # residual = out
        # guidance = self.cspn_guide(out)
        # out = self.cspn(guidance, out)
        out = self.denosing(out)

        for i, block in enumerate(self.backbone):
            out = block(out)

        out = self.tail_conv2D(out)
        # out = torch.add(out, residual)
        out = self.output_conv2D(self.output_prelu2D(out))
        # out = self.tail_nonlocal(out)
        return out


class Conv2D(nn.Module):
    def __init__(self, in_channel=256, out_channel=8):
        super(Conv2D, self).__init__()
        self.guide_conv2D = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

    def forward(self, x):
        spatial_guidance = self.guide_conv2D(x)
        return spatial_guidance


class Affinity_Propagate_Spatial(nn.Module):

    def __init__(self, prop_time=1, prop_kernel=3, norm_type='8sum'):
        super(Affinity_Propagate_Spatial, self).__init__()
        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        assert prop_kernel == 3, 'this version only support 8 (3x3 - 1) neighborhood'

        self.norm_type = norm_type
        assert norm_type in ['8sum', '8sum_abs']

        self.in_feature = 1
        self.out_feature = 1


    def forward(self, guidance, blur_depth):
        gate_wb, gate_sum = self.affinity_normalization(guidance)   # [1,8,1,64,64], [1,1,64,64]

        # pad input and convert to 8 channel 3D features
        raw_depth_input = blur_depth    # [1, 256, 64, 64]
        result_depth = blur_depth   # [1, 256, 64, 64]

        for i in range(self.prop_time):
            # one propagation
            # spn_kernel = self.prop_kernel
            result_depth = self.pad_blur_depth(result_depth)    # [1, 8, 256, 64, 64]
            neigbor_weighted_sum = torch.sum(gate_wb * result_depth, 1, keepdim=True)    # [1, 1, 256, 64, 64]
            neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)  # [1, 256, 64, 64]
            result_depth = neigbor_weighted_sum

            if '8sum' in self.norm_type:
                result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth    # [1, 256, 64, 64]
            else:
                raise ValueError('unknown norm %s' % self.norm_type)

        return result_depth


    def affinity_normalization(self, guidance):

        # normalize features
        if 'abs' in self.norm_type:
            guidance = torch.abs(guidance)

        gate1_wb_cmb = guidance.narrow(1, 0                   , self.out_feature)
        gate2_wb_cmb = guidance.narrow(1, 1 * self.out_feature, self.out_feature)
        gate3_wb_cmb = guidance.narrow(1, 2 * self.out_feature, self.out_feature)
        gate4_wb_cmb = guidance.narrow(1, 3 * self.out_feature, self.out_feature)
        gate5_wb_cmb = guidance.narrow(1, 4 * self.out_feature, self.out_feature)
        gate6_wb_cmb = guidance.narrow(1, 5 * self.out_feature, self.out_feature)
        gate7_wb_cmb = guidance.narrow(1, 6 * self.out_feature, self.out_feature)
        gate8_wb_cmb = guidance.narrow(1, 7 * self.out_feature, self.out_feature)

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # top pad
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))
        gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)

        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)

        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)

        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)

        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)

        right_bottm_pad = nn.ZeroPad2d((2, 0, 2, 0))
        gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)

        gate_wb = torch.cat((gate1_wb_cmb, gate2_wb_cmb, gate3_wb_cmb, gate4_wb_cmb,
                             gate5_wb_cmb, gate6_wb_cmb, gate7_wb_cmb, gate8_wb_cmb), 1)

        # normalize affinity using their abs sum
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = torch.sum(gate_wb_abs, 1, keepdim=True)

        gate_wb = torch.div(gate_wb, abs_weight)
        gate_sum = torch.sum(gate_wb, 1, keepdim=True)

        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]
        gate_wb = gate_wb[:, :, :, 1:-1, 1:-1]
        return gate_wb, gate_sum


    def pad_blur_depth(self, blur_depth):
        # top pad
        left_top_pad = nn.ZeroPad2d((0, 2, 0, 2))   # 左右上下
        blur_depth_1 = left_top_pad(blur_depth).unsqueeze(1)
        center_top_pad = nn.ZeroPad2d((1, 1, 0, 2))
        blur_depth_2 = center_top_pad(blur_depth).unsqueeze(1)
        right_top_pad = nn.ZeroPad2d((2, 0, 0, 2))
        blur_depth_3 = right_top_pad(blur_depth).unsqueeze(1)
        # center pad
        left_center_pad = nn.ZeroPad2d((0, 2, 1, 1))
        blur_depth_4 = left_center_pad(blur_depth).unsqueeze(1)
        right_center_pad = nn.ZeroPad2d((2, 0, 1, 1))
        blur_depth_5 = right_center_pad(blur_depth).unsqueeze(1)
        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0, 2, 2, 0))
        blur_depth_6 = left_bottom_pad(blur_depth).unsqueeze(1)
        center_bottom_pad = nn.ZeroPad2d((1, 1, 2, 0))
        blur_depth_7 = center_bottom_pad(blur_depth).unsqueeze(1)
        right_bottm_pad = nn.ZeroPad2d((2, 0, 2, 0))
        blur_depth_8 = right_bottm_pad(blur_depth).unsqueeze(1)
        result_depth = torch.cat((blur_depth_1, blur_depth_2, blur_depth_3, blur_depth_4,
                                  blur_depth_5, blur_depth_6, blur_depth_7, blur_depth_8), 1)
        result_depth = result_depth[:, :, :, 1:-1, 1:-1]
        return result_depth

class Denoising(nn.Module):
    def __init__(self, in_channel):
        super(Denoising, self).__init__()
        self.in_channel = in_channel
        self.activation = nn.LeakyReLU(0.2, inplace = True)
        self.conv0_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv0_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_0_cat = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)
        self.conv2_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv2_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_2_cat = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)
        self.conv4_33 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv4_11 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.conv_4_cat = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)

        self.conv_cat = nn.Conv2d(in_channel*3, in_channel, 3, 1, 1)
    
    def forward(self, x):

        x_0 = x
        x_2 = F.avg_pool2d(x, 2, 2)
        x_4 = F.avg_pool2d(x_2, 2, 2)
        # x_8 = F.avg_pool2d(x_4, 2, 2)

        x_0 = torch.cat([self.conv0_33(x_0), self.conv0_11(x_0)], 1)
        x_0 = self.activation(self.conv_0_cat(x_0))

        x_2 = torch.cat([self.conv2_33(x_2), self.conv2_11(x_2)], 1)
        x_2 = F.interpolate(self.activation(self.conv_2_cat(x_2)), scale_factor=2, mode='bilinear')

        x_4 = torch.cat([self.conv2_33(x_4), self.conv2_11(x_4)], 1)
        x_4 = F.interpolate(self.activation(self.conv_4_cat(x_4)), scale_factor=4, mode='bilinear')

        # x_8 = torch.cat([self.conv2_33(x_8), self.conv2_11(x_8)], 1)
        # x_8 = F.interpolate(self.activation(self.conv_2_cat(x_8)), scale_factor=8, mode='bilinear')

        x = x + self.activation(self.conv_cat(torch.cat([x_0, x_2, x_4], 1)))
        return x


if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # input_tensor = torch.rand(4, 3, 256, 256)
    input_tensor = torch.rand(1, 3, 512, 482)
    
    model = AWAN(3, 31, 100, 10)
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # print(output_tensor.shape)
    macs, params = profile(model, inputs=(input_tensor, ))
    print('Parameters number is {}; Flops: {}'.format(params, macs))
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    
    print(torch.__version__)




