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
from utility.load_tactile_dataset import tPSFNetDataSet
from config.default import tPSFNet_config, root_path
from utility.tools import select_gpu_with_least_used_memory, test_gpu
gpu_idx, device, _, _ = select_gpu_with_least_used_memory()

def generate_seqs_SRdataset():
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

    tPSFNetDataSet_train = tPSFNetDataSet(dataset_dir, sample_cnt=sample_cnt, is_sample_idx=[i for i in range(0, 81)])
    print(tPSFNetDataSet_train.__len__())


    data = []
    train_idx      = [2, 3, 4, 5, 6, 7, 8]
    validation_idx = [1]
    test_idx       = [0]
    
    SRdataset_train, SRdataset_validation, SRdataset_test = [], [], []
    
    for contact_idx in range(18):
        for trans_idx in range(9):
            for seqs_idx in range(sample_cnt):
                data_0  = tPSFNetDataSet_train[sample_cnt-1+sample_cnt*(0+trans_idx*9)+sample_cnt*81*contact_idx]
                data_5  = tPSFNetDataSet_train[sample_cnt-1+sample_cnt*(1+trans_idx*9)+sample_cnt*81*contact_idx]
                data_10 = tPSFNetDataSet_train[sample_cnt-1+sample_cnt*(2+trans_idx*9)+sample_cnt*81*contact_idx]
                data_15 = tPSFNetDataSet_train[sample_cnt-1+sample_cnt*(3+trans_idx*9)+sample_cnt*81*contact_idx]
                data_20 = tPSFNetDataSet_train[sample_cnt-1+sample_cnt*(4+trans_idx*9)+sample_cnt*81*contact_idx]
                data_25 = tPSFNetDataSet_train[sample_cnt-1+sample_cnt*(5+trans_idx*9)+sample_cnt*81*contact_idx]
                data_30 = tPSFNetDataSet_train[seqs_idx    +sample_cnt*(6+trans_idx*9)+sample_cnt*81*contact_idx]

                data_0_LR,  data_0_depth  = np.ascontiguousarray(data_0[0]),  np.ascontiguousarray(data_0[1])
                data_5_LR,  data_5_depth  = np.ascontiguousarray(data_5[0]),  np.ascontiguousarray(data_5[1])
                data_10_LR, data_10_depth = np.ascontiguousarray(data_10[0]), np.ascontiguousarray(data_10[1])
                data_15_LR, data_15_depth = np.ascontiguousarray(data_15[0]), np.ascontiguousarray(data_15[1])
                data_20_LR, data_20_depth = np.ascontiguousarray(data_20[0]), np.ascontiguousarray(data_20[1])
                data_25_LR, data_25_depth = np.ascontiguousarray(data_25[0]), np.ascontiguousarray(data_25[1])
                data_30_LR, data_30_depth = np.ascontiguousarray(data_30[0]), np.ascontiguousarray(data_30[1])

                data_0_LR, data_5_LR, data_10_LR, data_15_LR, data_20_LR, data_25_LR, data_30_LR = torch.from_numpy(data_0_LR)/scale_num, torch.from_numpy(data_5_LR)/scale_num, torch.from_numpy(data_10_LR)/scale_num, \
                                                                    torch.from_numpy(data_15_LR)/scale_num, torch.from_numpy(data_20_LR)/scale_num, torch.from_numpy(data_25_LR)/scale_num, torch.from_numpy(data_30_LR)/scale_num, 
                data_30_depth = torch.from_numpy(data_30_depth).unsqueeze(0)

                data_30_LR_in, data_30_depth_in = data_30_LR.unsqueeze(0), data_30_depth.unsqueeze(0)
                data_30_LR_in, data_30_depth_in = data_30_LR_in.to(device), data_30_depth_in.to(device)
                data_30_LR_in, data_30_depth_in = data_30_LR_in.type(torch.float32), data_30_depth_in.type(torch.float32)
                
                HR_tactile, LR_tactile_degrade, ret_psf, ret_alphaBeta = tPSF_model(data_30_LR_in, data_30_depth_in)
                alphaBeta = ret_alphaBeta[0][0].cpu().detach()
                HR_ori_30, LR_degrade_ori_30 = HR_tactile[0].cpu().detach(), LR_tactile_degrade[0].cpu().detach()

                # data_LR = torch.cat((data_0_LR, data_5_LR, data_10_LR, data_15_LR, data_20_LR, data_25_LR, data_30_LR), dim=0)
                data_LR = torch.cat((data_30_LR, data_25_LR, data_20_LR, data_15_LR, data_10_LR, data_5_LR, data_0_LR), dim=0)

                if trans_idx in validation_idx: 
                    SRdataset_validation.append([{'LR'  : data_LR,            # (21, 4, 4) 
                                                 'depth': data_30_depth, # (1, 100, 100)
                                                 'HR'   : HR_ori_30,     # (1, 100, 100)
                                                 }])
                    
                elif trans_idx in test_idx: 
                    SRdataset_test.append([{'LR'   : data_LR,            # (21, 4, 4) 
                                            'depth': data_30_depth, # (1, 100, 100)
                                            'HR'   : HR_ori_30,     # (1, 100, 100)
                                            }])
 
                else: 
                    SRdataset_train.append([{'LR'   : data_LR,            # (21, 4, 4) 
                                             'depth': data_30_depth, # (1, 100, 100)
                                             'HR'   : HR_ori_30,     # (1, 100, 100)
                                            }])

    print('train:', len(SRdataset_train)) 
    print('validation:', len(SRdataset_validation)) 
    print('test: ', len(SRdataset_test))     
    
    save_path = os.path.join(root_path, 'data/SeqsDataset')
    save_name_train, save_name_validation,save_name_test = 'SRdataset_train_32.npy', 'SRdataset_validation_32.npy','SRdataset_test_32.npy'
    np.save(os.path.join(save_path, save_name_train), SRdataset_train)
    np.save(os.path.join(save_path, save_name_validation),  SRdataset_validation)
    np.save(os.path.join(save_path, save_name_test),  SRdataset_test)    

if __name__ == "__main__":
    generate_seqs_SRdataset()
