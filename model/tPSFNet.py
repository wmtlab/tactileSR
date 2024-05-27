#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import math
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class tPSFNet(nn.Module):
    def __init__(self, gama, perception_scale, size=(100, 100), device=None):
        super(tPSFNet, self).__init__()

        # Degradation model parameters
        self.gama = gama
        self.perception_scale = perception_scale
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # MLP layer for predicting alpha and beta
        self.MLP_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*3, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softplus(),
        )
        self._init_weights(self.MLP_layer)

        # Zero padding function
        self.zeroPad_func = nn.ZeroPad2d(padding=(48, 48, 48, 48))

        # PSF sdf
        self.PSF_sdf = torch.zeros([1, 1, 99, 99], device=device, dtype=torch.float32)
        center_point_x, center_point_y = math.floor(self.PSF_sdf.shape[-2] / 2), math.floor(self.PSF_sdf.shape[-1] / 2)
        self.PSF_sdf[0][0] = self._sdf(99, 99, (center_point_x, center_point_y), dtype='torch')
        self.PSF_sdf = 10 * (self.PSF_sdf - self.PSF_sdf.min()) / (self.PSF_sdf.max() - self.PSF_sdf.min())  # Scale to (0, 10)

        # Masking sdf
        self.LR_masking_sdf = torch.zeros([4, 4, 100, 100], device=self.device)
        mask_centre = np.zeros((4, 4, 2))
        for x_idx in range(4):
            for y_idx in range(4):
                mask_centre[x_idx][y_idx][0], mask_centre[x_idx][y_idx][1] = 12 + x_idx * 25, 12 + y_idx * 25
                self.LR_masking_sdf[x_idx][y_idx] = self._sdf(100, 100, mask_centre[x_idx][y_idx], dtype='torch')
        self.LR_masking_sdf = 10 * (self.LR_masking_sdf - self.LR_masking_sdf.min()) / (self.LR_masking_sdf.max() - self.LR_masking_sdf.min())  # Scale to (0, 10)

    def _init_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.03)

    def _sdf(self, shape_x, shape_y, centre, dtype='numpy'):
        if dtype == 'torch':
            masking = torch.zeros([shape_x, shape_y], device=self.device)
        else:
            masking = np.zeros((shape_x, shape_y))

        for x in range(shape_x):
            for y in range(shape_y):
                masking[x][y] = ((x - centre[0])**2 + (y - centre[1])**2) ** 0.5
        return masking

    def tactilePSF(self, alphaBeta, order=2):
        """
        alphaBeta: size: (1, 2)
        return PSF: size: (1, 1, H, W)
        """
        return alphaBeta[0, 0] * torch.exp(-self.PSF_sdf.data**order / (alphaBeta[0, 1]**2))

    def depth2tactile(self, depth, psf):
        disturbance_num = 1e-3
        depth_masking = depth > (depth.max() - disturbance_num)  # 接触面
        depth = self.zeroPad_func(depth)
        HR = F.conv2d(depth, psf, padding=1)

        #--- max ---#
        # HR[depth_masking] = HR.detach().max()
        
        #--- second max  ---#
        temp_var = HR.detach()
        temp_var[depth_masking] = 0
        HR[depth_masking] = temp_var.max()
        del temp_var

        return HR

    def forward(self, x, depth):
        """
        x: B, C, H_L, W_L
        depth: B, 1, H_H, W_H
        return: HR_tactile: B, 1, H_H, W_H
        """
        # assert x.device == self.device, f"Input LR tactile {x.device} should be in the same device with the model {self.device}!"
        assert x.shape[0] == depth.shape[0], "Batch size of LR tactile and depth should be the same!"

        LR_tactile_degrade = torch.zeros((depth.shape[0], 1, 4, 4), device=self.device)
        ret_HR_tactile = torch.zeros(depth.shape, device=self.device)
        ret_psf = torch.zeros((depth.shape[0], 1, 99, 99), device=self.device)
        ret_alphaBeta = torch.zeros((depth.shape[0], 1, 3), device=self.device)

        alphaBeta = self.MLP_layer(x)

        for i in range(x.shape[0]):
            psf = self.tactilePSF(alphaBeta[i:i+1])
            HR_tactile = self.depth2tactile(depth[i:i+1], psf)
            LR_tactile_degrade[i:i+1] = self.degradation_process(HR_tactile, alphaBeta[i:i+1])

            ret_alphaBeta[i:i+1] = alphaBeta[i:i+1]
            ret_psf[i:i+1] = psf
            ret_HR_tactile[i:i+1] = HR_tactile

        return ret_HR_tactile, LR_tactile_degrade, ret_psf, ret_alphaBeta

    def degradation_process(self, HR_tactile, alphaBeta, order=2):
        """
        HR_tactile: 1, 1, H_H, W_H
        LR_tactile_degrade: 1, 1, H_L, W_L
        """
        masking = torch.exp(-self.LR_masking_sdf.data**2 / alphaBeta[0, 2])
        masking = (masking - masking.min()) / (masking.max() - masking.min())  # Scale to (0, 1)

        LR_tactile_degrade = torch.zeros(1, 1, 4, 4, device=self.device)
        for i in range(4):
            for j in range(4):
                LR_tactile_degrade[0, 0, i, j] = torch.sum(HR_tactile[0][0] * masking[i][j]) * 1e-4
        return LR_tactile_degrade


if __name__ == "__main__":
    batch_size = 4
    contact_depth = torch.rand(batch_size, 1, 100, 100)
    LR_tactile = torch.rand(batch_size, 3, 4, 4)
    tPSFNet = tPSFNet(gama=0.5, perception_scale=None, device='cpu')
    HR_tactile, LR_tactile_degrade, _, _ = tPSFNet(LR_tactile, contact_depth)
    print(HR_tactile.shape)
    print(LR_tactile_degrade.shape)
    
    
    
    