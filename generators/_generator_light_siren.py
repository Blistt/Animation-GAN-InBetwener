import torch
import torch.nn as nn
import os
import sys
from torch.nn import functional as F
from siren_pytorch import Sine



class GeneratorLight(nn.Module):
    
    def __init__(self, input_channels, hidden_channels, scaling=1.):
        super(GeneratorLight, self).__init__()
        filter_size = 4
        stride_size = 2
                
        self.down_sample_blocks = nn.Sequential(
            nn.Conv2d(input_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.BatchNorm2d(hidden_channels * 2),
            Sine(scaling),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            nn.BatchNorm2d(hidden_channels * 2),
            Sine(scaling),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            nn.BatchNorm2d(hidden_channels * 4),
            Sine(scaling),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size/2
            nn.BatchNorm2d(hidden_channels * 8),
            Sine(scaling)
            )
        
        self.up_sample_block = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            nn.BatchNorm2d(hidden_channels * 4),
            Sine(scaling),
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            nn.BatchNorm2d(hidden_channels * 2),
            Sine(scaling),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=filter_size, stride=stride_size, padding=1, bias=False),  # size*2
            nn.BatchNorm2d(hidden_channels),
            Sine(scaling),
            nn.ConvTranspose2d(hidden_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False),  # size
            nn.Sigmoid()
            )
    
    def forward(self, tensor0, tensor2):
        
        out = torch.cat((tensor0, tensor2), 1)  # @UndefinedVariable
        
        out_down = self.down_sample_blocks(out)
        out_up = self.up_sample_block(out_down)
        
        return out_up