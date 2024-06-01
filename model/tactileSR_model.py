#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""  
@author: Wu Bing
@date : 2022/10/30
"""

import numpy as np
import random
import math
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class TactileSR(nn.Module):
    """
    STSR, MTSR, ToH 2024
    """
    def __init__(self, scale_factor=10, seqsCnt=1, axisCnt=3, patternFeatureExtraLayerCnt=6, forceFeatureExtraLayerCnt=1):
        """ 
        scale_factor: 缩放因子
        seqsCnt     : 输入的seqs数目
        patternFeatureExtraLayerCnt: pattern 特征提取的层数
        """
        super(TactileSR, self).__init__()
        self.taxel_cnt = 4             # Xela tactile sensor has 4x4 taxel unit
        self.scale_factor = scale_factor
        self.seqsCnt = seqsCnt
        self.axisCnt = axisCnt
        
        self.patternFeatureExtra_layer = self.make_layer(MSRB, patternFeatureExtraLayerCnt)   # (BS, 64, 40, 40)
        self.forceFeatureExtra_layer   = self.make_layer(ResBlock, forceFeatureExtraLayerCnt)                         # (BS, 64, 40, 40) 
        
        self.inputLayer_pattern_list = nn.ModuleList()
        for seqIdx in range(self.seqsCnt):
            self.inputLayer_pattern_list.append(nn.Sequential(
                nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False),
                
                nn.Conv2d(in_channels=axisCnt, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            ))
        
        self.inputContact_layer= nn.Sequential(
            nn.Conv2d(in_channels=self.seqsCnt*64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
        )
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )
        
        self.input_layer_force = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=axisCnt, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
        )
        
        self._init_network()
    
    def forward(self, x):
        assert x.shape[1] == self.seqsCnt*self.axisCnt, "input channel should be same with seqsCnt x axisCnt!"
        
        # pattern Feature Extra
        InputpatternFeature = self.inputLayer_pattern_list[0](x[:, self.axisCnt*0:self.axisCnt*(0+1)])
        for seqIdx in range(1, self.seqsCnt):
            InputTappingFeature = self.inputLayer_pattern_list[seqIdx](x[:, self.axisCnt*seqIdx:self.axisCnt*(seqIdx+1)])
            InputpatternFeature = torch.cat((InputpatternFeature, InputTappingFeature), dim=1)
        patternFeature = self.patternFeatureExtra_layer(self.inputContact_layer(InputpatternFeature))
        
        # force Feature Extra
        forceFeature = self.forceFeatureExtra_layer(self.input_layer_force(x[:, :self.axisCnt]))
        
        # pattern Feature and force Feature fusion
        out = torch.cat((forceFeature, patternFeature), dim=1)
        out = self.output_layer(out)
        out = F.interpolate(out, size=(self.taxel_cnt*self.scale_factor, self.taxel_cnt*self.scale_factor), mode='bilinear', align_corners=False)
        return out
        
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer): 
            layers.append(block())
        return nn.Sequential(*layers)
    
    def _init_network(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0.1)


class TactileSRCNN(nn.Module):
    """
    TactileSRCNN & TactileSRGAN, IROS 2022
    """
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



class MSRB(nn.Module):
    """
    Multi-scale residual network for image super-resolution, 2018, ECCV
    """
    def __init__(self, n_feats=64):
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size_1, padding=1),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(True),
        )

        self.conv_5_1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size_2, padding=2),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(True),
        )

        self.conv_3_2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size_1, padding=1),
            nn.BatchNorm2d(n_feats*2),
            nn.ReLU(True),
        )

        self.conv_5_2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size_2, padding=2),
            nn.BatchNorm2d(n_feats*2),
            nn.ReLU(True),
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

    def _init_network(self,):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0.1)

class ResBlock(nn.Module):
    def __init__(self, n_feats=64):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_feats,n_feats,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(n_feats,n_feats,kernel_size=3,padding=1)
        
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)

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

