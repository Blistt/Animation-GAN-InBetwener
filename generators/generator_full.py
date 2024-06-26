import torch
from torch import nn
from utils.utils import crop

class ContractingBlock(nn.Module):
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels * 2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, i1, i2=None):
        if i2 is not None:
          x = self.conv(torch.cat((i1, i2), dim=1))
        else:
          x = self.conv(i1)
        return x


class UNetFull(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=32, use_dropout=False, use_bn=False):
        super(UNetFull, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_dropout=use_dropout, use_bn=use_bn)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=use_dropout, use_bn=use_bn)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=use_dropout, use_bn=use_bn)
        self.contract4 = ContractingBlock(hidden_channels * 8, use_bn=use_bn)
        self.contract5 = ContractingBlock(hidden_channels * 16, use_bn=use_bn)
        self.contract6 = ContractingBlock(hidden_channels * 32, use_bn=use_bn)
        self.expand0 = ExpandingBlock(hidden_channels * 64, use_bn=use_bn)
        self.expand1 = ExpandingBlock(hidden_channels * 32, use_bn=use_bn)
        self.expand2 = ExpandingBlock(hidden_channels * 16, use_bn=use_bn)
        self.expand3 = ExpandingBlock(hidden_channels * 8, use_bn=use_bn)
        self.expand4 = ExpandingBlock(hidden_channels * 4, use_bn=use_bn)
        self.expand5 = ExpandingBlock(hidden_channels * 2, use_bn=use_bn)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.activation = torch.nn.Tanh()

    def forward(self, i1, i2):
        x0 = self.upfeature(i1, i2)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)
        x7 = self.expand0(x6, x5)
        x8 = self.expand1(x7, x4)
        x9 = self.expand2(x8, x3)
        x10 = self.expand3(x9, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.downfeature(x12)
        return self.activation(xn)