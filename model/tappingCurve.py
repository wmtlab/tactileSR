#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
不同模型在tapping 过程中的曲线
"""

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utility.loadTactileDataSet import TactileSRDataset, TactileSRDataset_seq
from model.tactileSRNet import calculationPSNR, calculationSSIM
from model.tactileSRNet import TactileSR, TactileSR_noForceLayer
from model.tPSFNet import SSIM, setup_seed
from realDemo.realDemoDataProcess import SingleTappingDataByTime

setup_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

## -- 加载一个 tapping 过程的触觉序列 -- ## 
dataset_path = 'realDemo/data/singleTapping_contact_surface_3_tactile.npy'
# start_time, end_time = 1673250222, 1673250240
const_cnt = 30*6    # 30*6 | 
print('const_cnt: ', const_cnt)
start_time, end_time = 1673250215+const_cnt, 1673250235+const_cnt
dataset = SingleTappingDataByTime(dataset_path, scale_num=100, start_time=start_time, end_time=end_time)
time_list, data_list = dataset.getDataset()
print(len(data_list))

## -- 加载 模型 -- ## 
# STSR model        
STSR_model_pth = '/tactileSR_pth/bs_4_lr_1e-4_wd_1e-1/scale_factor_10/epoch_40.pth'
STSR_model = torch.load(root_path + STSR_model_pth).to(device)

# TactileSRCNN
tactileSRCNN_model_pth = '/irosPth/srcnn/bs_4_lr_1e-4_wd_1e-1_srcnn_msrn/scale_factor_10/epoch_90.pth'
tactileSRCNN_model = torch.load(root_path + tactileSRCNN_model_pth).to(device)

# TactileSRGAN
tactileSRGAN_model_pth = '/irosPth/srgan/bs_4_lr_1e-4_wd_1e-1_dnet_2_srcnn_msrn/scale_factor_10/epoch_20.pth'
tactileSRGAN_model = torch.load(root_path + tactileSRGAN_model_pth).to(device)

## -- 计算曲线 -- ##
tapping_time_list = []
LR_x_sum_list, LR_y_sum_list, LR_z_sum_list = [], [], []
STSR_model_sum_list, tactileSRCNN_model_sum_list, tactileSRGAN_model_sum_list = [], [], []

LR_x_max_list, LR_y_max_list, LR_z_max_list = [], [], []
STSR_model_max_list, tactileSRCNN_model_max_list, tactileSRGAN_model_max_list = [], [], []


for idx, LR in enumerate(data_list):
    # print("{}/{}, {}, {}".format(idx, time_list[idx], LR.shape, LR[2].sum()))
    LR_x = cv2.resize(LR[0], (40, 40), interpolation=cv2.INTER_LINEAR)
    LR_y = cv2.resize(LR[1], (40, 40), interpolation=cv2.INTER_LINEAR)
    LR_z = cv2.resize(LR[2], (40, 40), interpolation=cv2.INTER_LINEAR)
    LR_x_sum_list.append(LR_x.sum())
    LR_y_sum_list.append(LR_y.sum())
    LR_z_sum_list.append(LR_z.sum())
    
    LR_x_max_list.append(LR_x.max())
    LR_y_max_list.append(LR_y.max())
    LR_z_max_list.append(LR_z.max())
    
    tapping_time_list.append(time_list[idx])
    
    LR = torch.from_numpy(LR)
    LR = LR.type(torch.float32).cuda()
    LR = LR.unsqueeze(0)
    # input_data = input_data[:, :seqsCnt*axisCnt]
    
    out_stsr  = STSR_model(LR)
    out_srcnn = tactileSRCNN_model(LR)
    out_srgan = tactileSRGAN_model(LR)
    
    out_stsr_np  = out_stsr[0][0].cpu().detach().numpy()
    out_srcnn_np = out_srcnn[0][0].cpu().detach().numpy()
    out_srgan_np = out_srgan[0][0].cpu().detach().numpy()
    
    # out_np_sum              = abs(out_np.sum() - 2000)
    # out_noForceLayer_np_sum = out_noForceLayer_np.sum() - 1800
    
    # -- sum -- #
    hr_sum_scale = 0.5
    out_stsr_np_sum  = out_stsr_np.sum()  * hr_sum_scale
    # out_srcnn_np_sum = out_srcnn_np.sum() * hr_sum_scale
    # out_srgan_np_sum = out_srgan_np.sum() * hr_sum_scale
    out_srcnn_np_sum = out_srcnn_np.sum()
    out_srgan_np_sum = out_srgan_np.sum()
    
    STSR_model_sum_list.append(out_stsr_np_sum)
    tactileSRCNN_model_sum_list.append(out_srcnn_np_sum)
    tactileSRGAN_model_sum_list.append(out_srgan_np_sum)
    
    # -- max -- #
    hr_max_scale = 0.5
    out_stsr_np_max  = out_stsr_np.max()  * hr_max_scale
    out_srcnn_np_max = out_srcnn_np.max() * hr_max_scale
    out_srgan_np_max = out_srgan_np.max() * hr_max_scale
    
    STSR_model_max_list.append(out_stsr_np_max)
    tactileSRCNN_model_max_list.append(out_srcnn_np_max)
    tactileSRGAN_model_max_list.append(out_srgan_np_max)

    
tapping_time_list = np.array(tapping_time_list)
tapping_time_list -= tapping_time_list[0]
scale = 0.01
stepSize = 1

LR_x_sum_list = np.array(LR_x_sum_list)*scale* 1.4
LR_y_sum_list = np.array(LR_y_sum_list)*scale
LR_z_sum_list = np.array(LR_z_sum_list)*scale

STSR_model_sum_list         = np.abs(np.array(STSR_model_sum_list) * scale - 6.5)
tactileSRCNN_model_sum_list = np.abs(np.array(tactileSRCNN_model_sum_list) * scale - 30)
tactileSRGAN_model_sum_list = np.abs(np.array(tactileSRGAN_model_sum_list) * scale - 30)

# STSR_model_sum_list         = np.array(STSR_model_sum_list)*scale
# tactileSRCNN_model_sum_list = np.array(tactileSRCNN_model_sum_list)*scale
# tactileSRGAN_model_sum_list = np.array(tactileSRGAN_model_sum_list)*scale



## smooth ##
# n = 3
# LR_z_sum_list         = np.convolve(LR_z_sum_list, np.ones((n,))/n, mode='same')
# out_list              = np.convolve(out_list, np.ones((n,))/n, mode='same')
# out_noForceLayer_list = np.convolve(out_noForceLayer_list, np.ones((n,))/n, mode='same')

fontdict = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 20,
            }
plt.rc('font', **fontdict)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111)
# ax2 = ax1.twinx()


ax1.plot(tapping_time_list[::stepSize], LR_x_sum_list[::stepSize], color='thistle')
ax1.plot(tapping_time_list[::stepSize], LR_y_sum_list[::stepSize], color='violet')
ax1.plot(tapping_time_list[::stepSize], LR_z_sum_list[::stepSize], color='blue')

ax1.plot(tapping_time_list[::stepSize], STSR_model_sum_list[::stepSize], color='red')
ax1.plot(tapping_time_list[::stepSize], tactileSRCNN_model_sum_list[::stepSize], color='lightgreen')
ax1.plot(tapping_time_list[::stepSize], tactileSRGAN_model_sum_list[::stepSize], color='forestgreen')

ax1.set_ylim([0, 60])
ax1.set_xlim([0, 20])

# ax2.plot(tapping_time_list[::stepSize], LR_z_max_list[::stepSize], '--', color='blue')
# ax2.plot(tapping_time_list[::stepSize], out_max_list[::stepSize], '--', color='red')
# ax2.plot(tapping_time_list[::stepSize], out_max_noForceLayer_list[::stepSize], '--', color='green')
# ax2.set_ylim([0, 20])


# plt.savefig('out.png')
save_path = root_path + '/paperPlot/out/'
plt.savefig(save_path+"tappingCurve.png", transparent=True, dpi=1000, bbox_inches ='tight', pad_inches=0)
