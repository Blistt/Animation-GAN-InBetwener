import torch
import torch.nn as nn
import os
import torch.nn.utils as nn_utils


class DiscriminatorDense(nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(DiscriminatorDense, self).__init__()
                
        self.conv_blocks = nn.Sequential(
            nn_utils.spectral_norm(nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn_utils.spectral_norm(nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn_utils.spectral_norm(nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn_utils.spectral_norm(nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn_utils.spectral_norm(nn.Conv2d(hidden_channels * 8, hidden_channels * 16, kernel_size=4, stride=2, padding=1, bias=False)),
        )

        self.fc = None
    
    def forward(self, data):
        x = self.conv_blocks(data)
        if self.fc is None:
            self.fc = nn.Linear(x.shape[1] * x.shape[2] * x.shape[3], 1).to(data.device)
            self.fc = nn_utils.spectral_norm(self.fc)
            x = x.view(x.shape[0], -1)
            x = self.fc(x)
        return x