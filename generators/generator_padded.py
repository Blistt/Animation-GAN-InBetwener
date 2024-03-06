import torch
from torch import nn
from utils.utils import crop

class ContractingBlock(nn.Module):
    def __init__(self, input_channels):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3, padding=1, bias=False)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x


class ExpandingBlock(nn.Module):
    def __init__(self, input_channels):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(input_channels, input_channels//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(input_channels//2, input_channels//2, kernel_size=3, stride=1,padding=1, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x, skip_con_x):
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
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


"""
Whole Model
"""
class UNetPadded(nn.Module):
    def __init__(self, input_channels, output_channels=1, hidden_channels=64):
        super(UNetPadded, self).__init__()
        # "Every step in the expanding path consists of an upsampling of the feature map"
        print('Using UNetPadded')
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.expand1 = ExpandingBlock(hidden_channels * 16)
        self.expand2 = ExpandingBlock(hidden_channels * 8)
        self.expand3 = ExpandingBlock(hidden_channels * 4)
        self.expand4 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.activation = nn.Sigmoid()

    def forward(self, i1, i2):
        x0 = self.upfeature(i1, i2)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.expand1(x4, x3)
        x6 = self.expand2(x5, x2)
        x7 = self.expand3(x6, x1)
        x8 = self.expand4(x7, x0)
        xn = self.downfeature(x8)
        return self.activation(xn)