import torch
import torch.nn as nn
import os


class DiscriminatorCropDouble(nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(DiscriminatorCropDouble, self).__init__()
                
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),         
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),          
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),                 
            nn.BatchNorm2d(hidden_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_channels * 8, hidden_channels * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(hidden_channels * 8, hidden_channels * 16, kernel_size=4, stride=2, padding=1, bias=False),       
            nn.BatchNorm2d(hidden_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_channels * 16, hidden_channels * 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(hidden_channels * 16, hidden_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),   
            nn.BatchNorm2d(hidden_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_channels * 8, hidden_channels * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(hidden_channels * 8, 1, kernel_size=4, stride=2, padding=0, bias=False)
            )
    
    def forward(self, data):
        return self.conv_blocks(data)
    
