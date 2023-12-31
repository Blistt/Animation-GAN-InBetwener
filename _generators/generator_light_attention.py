import torch
import torch.nn as nn
import os
import sys
from torch.nn import functional as F
import torch.nn.utils as nn_utils

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class GeneratorLightSpectral(nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(GeneratorLightSpectral, self).__init__()
        filter_size = 4
        stride_size = 2
                
        self.down_sample_blocks = nn.Sequential(
            nn_utils.spectral_norm(nn.Conv2d(input_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1, bias=False)),  # size
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn_utils.spectral_norm(nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False)),  # size/2
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn_utils.spectral_norm(nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False)),  # size/2
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.02, inplace=True),
            nn_utils.spectral_norm(nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=filter_size, stride=stride_size, padding=1, bias=False)),  # size/2
            nn.BatchNorm2d(hidden_channels * 8),
            nn.LeakyReLU(0.02, inplace=True)
            )
        
        self.up_sample_block = nn.Sequential(
            nn_utils.spectral_norm(nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, kernel_size=filter_size, stride=stride_size, padding=1, bias=False)),  # size*2
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU(0.02, inplace=True),
            nn_utils.spectral_norm(nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=filter_size, stride=stride_size, padding=1, bias=False)),  # size*2
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.02, inplace=True),
            nn_utils.spectral_norm(nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=filter_size, stride=stride_size, padding=1, bias=False)),  # size*2
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.02, inplace=True),
            nn_utils.spectral_norm(nn.ConvTranspose2d(hidden_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=False)),  # size
            nn.Sigmoid()
            )
        
        # self.self_attn1 = Self_Attn(hidden_channels * 8, 'relu')
    
    def forward(self, tensor0, tensor2):

        out = torch.cat((tensor0, tensor2), 1)  # @UndefinedVariable
        
        out_down = self.down_sample_blocks(out)
        out_up = self.up_sample_block(out_down)
        
        return out_up