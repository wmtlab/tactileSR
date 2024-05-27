#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import math
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSRB(nn.Module):
    # Multi-scale residual network for image super-resolution, 2018, ECCV
    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=1),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )

        self.conv_5_1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size_2, padding=2),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )

        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size_1, padding=1),
            nn.BatchNorm2d(n_feats * 2),
            nn.ReLU(inplace=True),
        )

        self.conv_5_2 = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size_2, padding=2),
            nn.BatchNorm2d(n_feats * 2),
            nn.ReLU(inplace=True),
        )

        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)
        
        self._init_network()

    def forward(self, x):
        input_1 = x
        output_3_1 = self.conv_3_1(input_1)
        output_5_1 = self.conv_5_1(input_1)
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.conv_3_2(input_2)
        output_5_2 = self.conv_5_2(input_2)
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return self.relu(output)

    def _init_network(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0.1)


class TactileSRCNN(nn.Module):
    def __init__(self):
        super(TactileSRCNN, self).__init__()
        self.msrb_layer = self._make_layer(MSRB, 6)
        self.input_zyx = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.upSample = nn.Upsample(
            scale_factor=10,
            mode='bilinear',
            align_corners=False
        )
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self._init_network()

    def _init_network(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0.1)

    def _make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out_zyx = self.upSample(x)
        out_zyx = self.input_zyx(out_zyx)
        out_zyx = self.msrb_layer(out_zyx)
        out_zyx = self.output(out_zyx)
        return out_zyx


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1).cuda()
		
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).cuda()
    
    gradients = torch.autograd.grad(outputs=d_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=fake,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True,
                                    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2,dim=1) - 1)**2).mean()
    return gradient_penalty


class Dis_Net(nn.Module):
    def __init__(self):
        super(Dis_Net, self).__init__()
        self.dis_input = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ) 
        self.residual_layer = self.make_layer(Leaky_Res_Block, 3)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(1, inplace=True),
            nn.MaxPool2d(5, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(1, inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Flatten(),
            nn.Linear(1024, 1),
            # nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.dis_input(x)
        x = self.residual_layer(x)
        return self.output(x)


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

class Leaky_Res_Block(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, strides=1):
        super(Leaky_Res_Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(1, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.block(x)
        return self.relu(out+x)

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())
    
    # -- input Data -- #
    batch_size = 4
    LR_tactile = torch.rand(batch_size, 3, 4, 4).cuda()
    tactileSR = TactileSRCNN(scale_factor=10, seqsCnt=1, axisCnt=3).cuda()
    out = tactileSR(LR_tactile)
    print(LR_tactile.shape)
    print(out.shape)
    
    disNet = Dis_Net().cuda()
    logical_out = disNet(out)
    print(logical_out.shape)
    
