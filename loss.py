'''
Created on Apr 14, 2020

@author: lab1323pc
'''

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import utils.calculator as cal


def get_gen_loss(preds, disc, real, adv_l, adv_lambda, l1=None, l2=None, l3=None, lamb1=None, lamb2=None, lamb3=None, device='cuda'):
    disc_pred_hat = disc(preds)
    gen_adv_loss = adv_l(disc_pred_hat, torch.ones_like(disc_pred_hat))
    gen_l1 = l1(real, preds)
    gen_l2 = l2(real, preds)
    gen_l3 = l3(real, preds)
    gen_loss = (gen_adv_loss * adv_lambda) + (gen_l1 * lamb1) + (gen_l2 * lamb2) + (gen_l3 * lamb3)
    return gen_loss



def gdl_loss(gen_frames, gt_frames, alpha=2, device='cuda:1'):
    filter_x = nn.Conv2d(1, 1, (1, 3), padding=(0, 1)).to(device)
    filter_y = nn.Conv2d(1, 1, (3, 1), padding=(1, 0)).to(device)

    gen_dx = filter_x(gen_frames)
    gen_dy = filter_y(gen_frames)
    gt_dx = filter_x(gt_frames)
    gt_dy = filter_y(gt_frames)

    grad_diff_x = torch.pow(torch.abs(gt_dx - gen_dx), alpha)
    grad_diff_y = torch.pow(torch.abs(gt_dy - gen_dy), alpha)

    grad_total = torch.stack([grad_diff_x, grad_diff_y])

    return torch.mean(grad_total)


class GDL(nn.Module):
    '''
    Gradient different loss function
    Target: reduce motion blur 
    '''

    def __init__(self, device='cuda:1'):
        super(GDL, self).__init__()
        self.device = device

    def forward(self, gen_frames, gt_frames):
        return gdl_loss(gen_frames, gt_frames, device=self.device)

    
class MS_SSIM(nn.Module):
    '''
    Multi scale SSIM loss. Refer from:
    - https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
    - https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    '''
    def __init__(self, device='cuda:1'):
        super(MS_SSIM, self).__init__()
        self.device = device

    def forward(self, gen_frames, gt_frames):
        return 1 - self._cal_ms_ssim(gen_frames, gt_frames)
        
    def _cal_ms_ssim(self, gen_tensors, gt_tensors):
        weights = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        if torch.cuda.is_available():
            weights = weights.to(self.device)
        
        gen = gen_tensors
        gt = gt_tensors
        levels = weights.shape[0]
        mcs = []
        win_size = 3 if gen_tensors.shape[3] < 256 else 11
        win = cal._fspecial_gauss_1d(size=win_size, sigma=1.5)
        win = win.repeat(gen.shape[1], 1, 1, 1)
        
        for i in range(levels):
            ssim_per_channel, cs = cal._ssim_tensor(gen, gt, data_range=1.0, win=win)
            
            if i < levels - 1: 
                mcs.append(torch.relu(cs))
                padding = (gen.shape[2] % 2, gen.shape[3] % 2)
                gen = F.avg_pool2d(gen, kernel_size=2, padding=padding)
                gt = F.avg_pool2d(gt, kernel_size=2, padding=padding)
        
        ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
        mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
        ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)
    
        return ms_ssim_val.mean()
