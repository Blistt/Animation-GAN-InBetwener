import torch
import torch.nn as nn
import os
import sys
from torch.nn import functional as F
import torch.nn.utils as nn_utils



class ContractingBlock(nn.Module):
    def __init__(self, input_channels, kernel_size=4, stride=2, spectral=True):
        super(ContractingBlock, self).__init__()
        if spectral:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(input_channels, input_channels*2, kernel_size=kernel_size,
                                                           stride=stride, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(input_channels, input_channels*2, kernel_size=kernel_size,
                                    stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(input_channels * 2)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, kernel_size=4, stride=2, spectral=True):
        super(ExpandingBlock, self).__init__()
        if spectral:
            self.upconv = nn.utils.spectral_norm(nn.ConvTranspose2d(input_channels, input_channels//2, kernel_size=kernel_size, 
                                                                    stride=stride, padding=1, bias=False))
        else:
            self.upconv = nn.ConvTranspose2d(input_channels, input_channels//2, kernel_size=kernel_size, stride=stride, 
                                             padding=1, bias=False)
        self.bn = nn.BatchNorm2d(input_channels // 2)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels, spectral=True):
        super(FeatureMapBlock, self).__init__()
        if spectral:
            self.conv = nn.utils.spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size=1))
        else:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, i1, i2=None):
        if i2 is not None:
          x = self.conv(torch.cat((i1, i2), dim=1))
        else:
          x = self.conv(i1)
        return x


class GeneratorLight(nn.Module):
    def __init__(self, input_channels, hidden_channels, filter_size=4, stride_size=2, padd=False, spetral=True):
        super(GeneratorLight, self).__init__()
        self.padd = padd
        filter_size = 4
        stride_size = 2
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, kernel_size=3, stride=1)
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=filter_size, stride=stride_size)
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=filter_size, stride=stride_size)
        self.contract4 = ContractingBlock(hidden_channels * 8, kernel_size=filter_size, stride=stride_size)
        self.expand1 = ExpandingBlock(hidden_channels *16, kernel_size=filter_size, stride=stride_size)
        self.expand2 = ExpandingBlock(hidden_channels * 8, kernel_size=filter_size, stride=stride_size)
        self.expand3 = ExpandingBlock(hidden_channels * 4, kernel_size=filter_size, stride=stride_size)
        self.expand4 = ExpandingBlock(hidden_channels * 2, kernel_size=3, stride=1)
        self.downfeature = FeatureMapBlock(hidden_channels, input_channels//2)
        self.activation = nn.Sigmoid()
               
    def padding(self, i0, i2):
        '''
        Padding on input if the input frames are not in square size or size is not multiples of 32
        :param i0:
        :param i2:
        '''
        h0 = int(list(i0.size())[2])
        w0 = int(list(i0.size())[3])
        h2 = int(list(i2.size())[2])
        w2 = int(list(i2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if (h0 % 32 != 0 or (h0 - w0) < 0):
            pad_h = 32 - (h0 % 32) if (h0 - w0) >= 0 else 32 - (h0 % 32) + (w0 - h0)
            i0 = F.pad(i0, (0, 0, 0, pad_h))
            i2 = F.pad(i2, (0, 0, 0, pad_h))
            h_padded = True

        if (w0 % 32 != 0 or (h0 - w0) > 0):
            pad_w = 32 - (w0 % 32) if (h0 - w0) <= 0 else 32 - (h0 % 32) + (h0 - w0)
            i0 = F.pad(i0, (0, pad_w, 0, 0))
            i2 = F.pad(i2, (0, pad_w, 0, 0))
            w_padded = True
        
        return i0, i2, h_padded, w_padded, h0, w0
        
    def forward(self, x0, x2):

        # # Padding on input if the input frames are not in square size or size
        # is not multiples of 32
        i0, i2, h_padded, w_padded, h0, w0 = self.padding(x0, x2)
        x1 = self.upfeature(i0, i2)
        x1 = self.contract1(x1)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.expand1(x4)
        x6 = self.expand2(x5)
        x7 = self.expand3(x6)
        x8 = self.expand4(x7)
        x9 = self.downfeature(x8)
            
        if h_padded:
            x9 = x9[:, :, 0:h0, :]
        if w_padded:
            x9 = x9[:, :, :, 0:w0]
         
        return self.activation(x9)