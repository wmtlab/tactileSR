#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


from irosModel import TactileSRCNN

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from model.tactileSRNet import calculationPSNR, calculationSSIM
from model.tPSFNet import setup_seed
from utility.loadTactileDataSet import TactileSRDataset

setup_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def evaluateSRnet():
    """ 
        评估 TactileSRCNN 和 TactileSRGAN 在该数据集下的性能
    """
    scale_factor = 10
    test_dataSize, test_currentIdx = tactileSRDataset_test.__len__() - 1, 0
    
    ## -- TactileSRCNN -- ##
    srcnn_pth = '/irosPth/srcnn/bs_4_lr_1e-4_wd_1e-1_srcnn_msrn/scale_factor_10/epoch_90.pth'
    # srcnn_pth = '/irosPth/srcnn/bs_4_lr_1e-4_wd_1e-1/scale_factor_10/epoch_90.pth'
    srcnn_model = torch.load(root_path+srcnn_pth).to(device)

    ## -- TactileSRGAN -- ##
    srgan_pth = '/irosPth/srgan/bs_4_lr_1e-4_wd_1e-1_dnet_2_srcnn_msrn/scale_factor_10/epoch_20.pth'
    # srgan_pth = '/irosPth/srgan/bs_4_lr_1e-4_wd_1e-1_dnet_2/scale_factor_10/epoch_40.pth'
    srgan_model = torch.load(root_path+srgan_pth).to(device)


    ## -- evaluate -- ## 
    all_psnr_cnn, all_ssim_cnn = 0, 0
    all_psnr_gan, all_ssim_gan = 0, 0
    for LR, HR in tqdm(test_loader):
        LR, HR = LR.to(device), HR.to(device)
        LR, HR = LR.type(torch.float32), HR.type(torch.float32) / HR_scale_num
        HR = F.interpolate(HR, size=(4*scale_factor, 4*scale_factor), mode='bilinear', align_corners=False)
        # LR = LR[:, :seqsCnt*axisCnt]
        out_srcnn = srcnn_model(LR)
        out_srgan = srgan_model(LR)

        LR_img = LR[0][2].cpu().detach()
        HR_img = HR[0][0].cpu().detach()
        SR_cnn = out_srcnn[0][0].cpu().detach()
        SR_gan = out_srgan[0][0].cpu().detach()
        
        tmp_psnr_cnn = calculationPSNR(SR_cnn, HR_img, maxValue=sensorMaxVaule)
        tmp_ssim_cnn = calculationSSIM(SR_cnn, HR_img)
        all_psnr_cnn += tmp_psnr_cnn
        all_ssim_cnn += tmp_ssim_cnn
        
        tmp_psnr_gan = calculationPSNR(SR_gan, HR_img, maxValue=sensorMaxVaule)
        tmp_ssim_gan = calculationSSIM(SR_gan, HR_img)
        all_psnr_gan += tmp_psnr_gan
        all_ssim_gan += tmp_ssim_gan
        
        test_currentIdx += 1

    print("[CNN] PSNR:{:.2f}, SSIM:{:.4f} | [GAN] PSNR:{:.2f}, SSIM:{:.4f}".
            format(all_psnr_cnn/test_currentIdx, all_ssim_cnn/test_currentIdx, all_psnr_gan/test_currentIdx, all_ssim_gan/test_currentIdx))

HR_scale_num = 10
sensorMaxVaule = 250 / HR_scale_num

## ----  load Dataset ---- ##
tactileSRDataset_test = TactileSRDataset(root_path + '/Dataset/SRdataset/order/SRdataset_validation.npy')
# tactileSRDataset_test = TactileSRDataset(root_path + '/Dataset/SRdataset/order/SRdataset_test.npy')
print('dataset size:',tactileSRDataset_test.__len__())
test_loader = DataLoader(tactileSRDataset_test, batch_size=1, shuffle=False)
evaluateSRnet()

""" 
模型评估结果

    srcnn_pth = '/irosPth/srcnn/bs_4_lr_1e-4_wd_1e-1_srcnn_msrn/scale_factor_10/epoch_90.pth'
    srgan_pth = '/irosPth/srgan/bs_4_lr_1e-4_wd_1e-1_dnet_2_srcnn_msrn/scale_factor_10/epoch_20.pth'
    # validation
    [CNN] PSNR:22.17, SSIM:0.8277 | [GAN] PSNR:22.05, SSIM:0.8241
    # test
    [CNN] PSNR:21.10, SSIM:0.7690 | [GAN] PSNR:21.54, SSIM:0.7876


"""


