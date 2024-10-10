import torch
import torch.nn as nn
from thop import profile
# UNet with PixelShuffle & MaxPool

class double_conv(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_features, output_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout2d(p=p_drop),
            nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class down_block(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(down_block, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=p_drop),
            double_conv(input_features, output_features, negative_slope, p_drop)
        )
    def forward(self, x):
        x = self.down(x)
        return x

class up_block(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(up_block, self).__init__()
        self.up = nn.PixelShuffle(upscale_factor=2)
        self.conv = nn.Sequential(
            nn.Dropout2d(p=p_drop),
            double_conv(int(input_features/4+output_features), output_features, negative_slope, p_drop)
        )
    def forward(self, x, x_pre):
        x = self.up(x)
        x = torch.cat((x, x_pre), 1)
        x = self.conv(x)
        return x

class Model(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(Model, self).__init__()
        self.in_conv = double_conv(input_features, 64, negative_slope, p_drop)
        self.down1 = down_block(64, 128, negative_slope, p_drop) # W/2
        self.down2 = down_block(128, 256, negative_slope, p_drop) # W/4
        self.down3 = down_block(256, 512, negative_slope, p_drop) # W/8
        self.bottleneck = down_block(512, 1024, negative_slope, p_drop) # W/16
        self.up1 = up_block(1024, 512, negative_slope, p_drop) # W/8
        self.up2 = up_block(512, 256, negative_slope, p_drop) # W/4
        self.up3 = up_block(256, 128, negative_slope, p_drop) # W/2
        self.out_conv = up_block(128, 64, negative_slope, p_drop) # W
        self.out = nn.Conv2d(64, output_features, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        conv = self.in_conv(x) # W 64
        down1 = self.down1(conv) # W/2 128
        down2 = self.down2(down1) # W/4 256
        down3 = self.down3(down2) # W/8 512
        bottleneck = self.bottleneck(down3) # W/16 1024
        up = self.up1(bottleneck, down3) # W/8 512
        up = self.up2(up, down2) # W/4 256
        up = self.up3(up, down1) # W/2 128
        up = self.out_conv(up, conv) # W 64
        return self.out(up)
if __name__ == "__main__":
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # input_tensor = torch.rand(6, 3, 256, 256)
    input_tensor = torch.rand(1, 3, 512, 480).cuda()

    model = Model(3,31).cuda()
    # model = nn.DataParallel(model).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    # print(output_tensor.shape)
    macs, params = profile(model, inputs=(input_tensor,))
    print('Parameters number is {}; Flops: {}'.format(params, macs))
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    print(torch.__version__)