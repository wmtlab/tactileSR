#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import os, sys
import cv2
from tqdm import tqdm
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from model.tPSFNet import tPSFNet
from utility.load_tactile_dataset import tPSFNetDataSet, singleTapSeqsDataset
from model.tPSFNet import tPSFNet
from config.default import tPSFNet_config, root_path
from utility.tools import select_gpu_with_least_used_memory, test_gpu
gpu_idx, device, _, _ = select_gpu_with_least_used_memory()

def loadDataset():
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    dataset_path = root_path+'/Dataset/SRdataset/SRdataset_disorder_test.npy'
    SRdataset = np.load(dataset_path, allow_pickle=True)
    
    ## -- 可视化 -- ## 
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])

    
    for SRdata in SRdataset:
        SRdata = SRdata.item()
        LR         = SRdata['LR']            # [1, 3, 4, 4]
        depth      = SRdata['depth']         # [1, 1, 100, 100]
        HR         = SRdata['HR']            # [1, 1, 100, 100]
        LR_degrade = SRdata['LR_degrade']    # [1, 1, 4, 4]
        alphaBeta  = SRdata['alphaBeta']     # [2, ]
        
        ax1.cla(), ax2.cla(), ax3.cla(), ax4.cla()
        
        LR, depth, HR, LR_degrade = LR.numpy(), depth.numpy(), HR.numpy(), LR_degrade.numpy()
        ax1.imshow(LR[2], vmin=0, vmax=13)
        ax2.imshow(LR_degrade[0], vmin=0, vmax=13)
        ax3.imshow(depth[0], vmin=0, vmax=1)
        ax4.imshow(HR[0], vmin=0, vmax=250)
        
        ax1.set_title('LR')
        ax2.set_title('LR_degrade')
        ax3.set_title('depth')
        ax4.set_title('HR')
        
        plt.savefig('out.png')
        


def generate_single_SRdataset():
    pth_dir = "/code/pth/tPSFNet_no_aug/checkpoints/epoch_1.pth"
    dataset_dir = os.path.join(root_path, 'data/rotateDataset')
    sample_cnt = 16
    batch_size = 1
    scale_num = 100
    
    tPSF_model = tPSFNet(gama=tPSFNet_config['gama'],
                    perception_scale=tPSFNet_config['perception_scale'], 
                    device=device,
                    )
    checkpoint = torch.load(pth_dir,  map_location=device)
    tPSF_model.load_state_dict(checkpoint['model'], strict=False)
    tPSF_model = tPSF_model.to(device)
    print("tPSF: gama={}, perception_scale={}".format(tPSF_model.gama, tPSF_model.perception_scale))

    ## -- 加载数据 -- ##
    test_idx       = [0+9*0, 6+9*0, 7+9*0, 8+9*0]
    validation_idx = [0+9*1, 6+9*1, 7+9*1, 8+9*1]
    train_idx      = [0+9*2, 6+9*2, 7+9*2, 8+9*2, 
                      0+9*3, 6+9*3, 7+9*3, 8+9*3, 
                      0+9*4, 6+9*4, 7+9*4, 8+9*4, 
                      0+9*5, 6+9*5, 7+9*5, 8+9*5, 
                      0+9*6, 6+9*6, 7+9*6, 8+9*6, 
                      0+9*7, 6+9*7, 7+9*7, 8+9*7, 
                      0+9*8, 6+9*8, 7+9*8, 8+9*8]

    # train
    tPSFNetDataSet_train = tPSFNetDataSet(dataset_dir, sample_cnt=sample_cnt, is_sample_idx=train_idx)
    print("train data len: ", tPSFNetDataSet_train.__len__())     # sample_cnt x 9 x 4 x 9
    data_loader_train = DataLoader(tPSFNetDataSet_train, batch_size=batch_size, shuffle=False)

    # test
    tPSFNetDataSet_test = tPSFNetDataSet(dataset_dir, sample_cnt=sample_cnt, is_sample_idx=test_idx)
    print("test data len: ", tPSFNetDataSet_test.__len__())
    data_loader_test = DataLoader(tPSFNetDataSet_test, batch_size=batch_size, shuffle=False)

    # validation
    tPSFNetDataSet_validation = tPSFNetDataSet(dataset_dir, sample_cnt=sample_cnt, is_sample_idx=validation_idx)
    print("validation data len: ", tPSFNetDataSet_validation.__len__())
    data_loader_validation = DataLoader(tPSFNetDataSet_validation, batch_size=batch_size, shuffle=False)
    
    ## -- 处理数据 -- ##
    SRdataset_train, SRdataset_validation, SRdataset_test = [], [], []
    # train
    for LR, depth in tqdm(data_loader_train):
        LR, depth = LR.to(device), depth.to(device)
        LR, depth = LR.type(torch.float32)/scale_num, depth.type(torch.float32)
        depth = depth.unsqueeze(1)
        HR_tactile, LR_tactile_degrade, ret_psf, ret_alphaBeta = tPSF_model(LR, depth)
        alphaBeta = ret_alphaBeta[0][0].cpu().detach()
        LR_ori_0, depth_ori_0, HR_ori_0, LR_degrade_ori_0 = LR[0].cpu().detach(), depth[0].cpu().detach(), HR_tactile[0].cpu().detach(), LR_tactile_degrade[0].cpu().detach()
        SRdataset_train.append([{'LR'         : LR_ori_0,          # (3, 4, 4) 
                                 'depth'      : depth_ori_0,       # (1, 100, 100)
                                 'HR'         : HR_ori_0,          # (1, 100, 100) 
                                 'LR_degrade' : LR_degrade_ori_0,  # (1, 4, 4)
                                 'alphaBeta'  : alphaBeta      
                                }])

    # validation
    for LR, depth in tqdm(data_loader_validation):
        LR, depth = LR.to(device), depth.to(device)
        LR, depth = LR.type(torch.float32)/scale_num, depth.type(torch.float32)
        depth = depth.unsqueeze(1)
        HR_tactile, LR_tactile_degrade, ret_psf, ret_alphaBeta = tPSF_model(LR, depth)
        alphaBeta = ret_alphaBeta[0][0].cpu().detach()
        LR_ori_0, depth_ori_0, HR_ori_0, LR_degrade_ori_0 = LR[0].cpu().detach(), depth[0].cpu().detach(), HR_tactile[0].cpu().detach(), LR_tactile_degrade[0].cpu().detach()
        SRdataset_validation.append([{'LR'         : LR_ori_0,          # (3, 4, 4) 
                                 'depth'      : depth_ori_0,       # (1, 100, 100)
                                 'HR'         : HR_ori_0,          # (1, 100, 100) 
                                 'LR_degrade' : LR_degrade_ori_0,  # (1, 4, 4)
                                 'alphaBeta'  : alphaBeta      
                                }])
    
    # test
    for LR, depth in tqdm(data_loader_test):
        LR, depth = LR.to(device), depth.to(device)
        LR, depth = LR.type(torch.float32)/scale_num, depth.type(torch.float32)
        depth = depth.unsqueeze(1)
        HR_tactile, LR_tactile_degrade, ret_psf, ret_alphaBeta = tPSF_model(LR, depth)
        alphaBeta = ret_alphaBeta[0][0].cpu().detach()
        LR_ori_0, depth_ori_0, HR_ori_0, LR_degrade_ori_0 = LR[0].cpu().detach(), depth[0].cpu().detach(), HR_tactile[0].cpu().detach(), LR_tactile_degrade[0].cpu().detach()
        SRdataset_test.append([{'LR'         : LR_ori_0,          # (3, 4, 4) 
                                 'depth'      : depth_ori_0,       # (1, 100, 100)
                                 'HR'         : HR_ori_0,          # (1, 100, 100) 
                                 'LR_degrade' : LR_degrade_ori_0,  # (1, 4, 4)
                                 'alphaBeta'  : alphaBeta      
                                }])
        
    print('train:', len(SRdataset_train)) 
    print('validation:', len(SRdataset_validation)) 
    print('test: ', len(SRdataset_test)) 
        
    ## -- 保存数据 -- ##
    save_path = os.path.join(root_path, 'data/SRdataset')
    save_name_train, save_name_validation,save_name_test = 'SRdataset_train.npy', 'SRdataset_validation.npy','SRdataset_test.npy'
    np.save(os.path.join(save_path, save_name_train), SRdataset_train)
    np.save(os.path.join(save_path, save_name_validation),  SRdataset_validation)
    np.save(os.path.join(save_path, save_name_test),  SRdataset_test)


if __name__ == "__main__":
    generate_single_SRdataset()
    
