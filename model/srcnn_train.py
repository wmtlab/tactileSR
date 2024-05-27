#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from irosModel import TactileSRCNN, SRCNN_MSRN

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from model.tactileSRNet import calculationPSNR, calculationSSIM
from model.tPSFNet import setup_seed
from utility.loadTactileDataSet import TactileSRDataset

setup_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

## ----------------------- config ---------------------------- ##
inin_lr, weight_decay = 0.0001, 1e-1
scheduler_StepLR_step_size, scheduler_StepLR_gama = 2, 0.8
train_batch_size, test_batch_size = 4, 1
HR_scale_num = 10
sensorMaxVaule = 250 / HR_scale_num
epochs = 101
scale_factor = 10          # 缩放因子 4, 6, 8 ,10
patternFeatureExtraLayerCnt = 6
################################################################

## --------------- 设置保存路径 --------------- ##
hyperParam = 'bs_4_lr_1e-4_wd_1e-1_srcnn_msrn'

save_dir = root_path + '/irosPth/srcnn/' + hyperParam
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
save_dir = save_dir + '/scale_factor_' + str(scale_factor)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
print("Pth save path : ", save_dir)

## ----------- DataSet -------------- ##
# train
tactileSRDataset_train = TactileSRDataset(root_path + '/Dataset/SRdataset/order/SRdataset_train.npy')
print('train dataset size:',tactileSRDataset_train.__len__())
train_loader = DataLoader(tactileSRDataset_train, batch_size=train_batch_size, shuffle=True)
train_size = tactileSRDataset_train.__len__()
# validation
tactileSRDataset_test = TactileSRDataset(root_path + '/Dataset/SRdataset/order/SRdataset_validation.npy')
print('test dataset size:',tactileSRDataset_test.__len__())
test_loader = DataLoader(tactileSRDataset_test, batch_size=test_batch_size, shuffle=False)
test_size = tactileSRDataset_test.__len__()


## ----------- Train Config -------------- ##
TactileSR = SRCNN_MSRN().to(device)
# TactileSR = TactileSRCNN(scale_factor=scale_factor, patternFeatureExtraLayerCnt=patternFeatureExtraLayerCnt).to(device)
optimizer = optim.Adam(TactileSR.parameters(), lr=inin_lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_StepLR_step_size, gamma=scheduler_StepLR_gama)
mse_loss = nn.MSELoss()

## ----------- log -------------- ##
# tensorBorad
writer = SummaryWriter(log_dir='logs/irosModel/srcnn/' + hyperParam + '/' + 'scale_factor_' + str(scale_factor))

for epoch in range(epochs):
    epoch_loss = 0
    for LR, HR in train_loader:
        LR, HR = LR.to(device), HR.to(device)
        LR, HR = LR.type(torch.float32), HR.type(torch.float32) / HR_scale_num
        HR = F.interpolate(HR, size=(4*scale_factor, 4*scale_factor), mode='bilinear', align_corners=False)
        out = TactileSR(LR)
            
        loss = mse_loss(out, HR)
        
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    scheduler.step()
    print("===> Epoch[{}]: learn rate:(1e-3)x{:.3f} Loss: {:.4f}".format(epoch, 1000*optimizer.state_dict()['param_groups'][0]['lr'], epoch_loss), end='  |  ')
    writer.add_scalar('train_loss', scalar_value=epoch_loss, global_step=epoch)
    
    test_loss, test_ssim_loss, test_psnr_loss = 0, 0, 0
    for LR, HR in test_loader:
        LR, HR = LR.to(device), HR.to(device)
        LR, HR = LR.type(torch.float32), HR.type(torch.float32) / HR_scale_num
        HR = F.interpolate(HR, size=(4*scale_factor, 4*scale_factor), mode='bilinear', align_corners=False)
        out = TactileSR(LR)
        
        loss = mse_loss(out, HR)
        
        test_loss += loss.item()
        
        tmp_psnr = calculationPSNR(out[0][0].detach(), HR[0][0].detach(), maxValue=sensorMaxVaule)
        tmp_ssim = calculationSSIM(out[0][0].detach(), HR[0][0].detach())
        test_psnr_loss += tmp_psnr
        test_ssim_loss += tmp_ssim
        
    print("test loss:{:.3f}, ssim loss:{:.3f}, psnr loss:{:.3f}".format(test_loss, (test_batch_size*test_ssim_loss)/test_size, (test_batch_size*test_psnr_loss)/test_size))
    writer.add_scalar('test_loss', scalar_value=test_loss, global_step=epoch)
    writer.add_scalar('test PSNR', scalar_value=(test_batch_size*test_psnr_loss)/test_size, global_step=epoch)
    writer.add_scalar('test SSIM', scalar_value=(test_batch_size*test_ssim_loss)/test_size, global_step=epoch)
    
    
    if epoch%5 == 0:
        save_file_name = '/epoch_' + str(epoch) 
        torch.save(TactileSR, save_dir + save_file_name + '.pth')

writer.close()



