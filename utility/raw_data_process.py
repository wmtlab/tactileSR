#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""  原始数据处理。
@author: Wu Bing
@date : 2022/10/17
"""
import os, sys
import numpy as np
from tqdm import tqdm
import cv2

# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# from config.default import data_process_config

""" 
每个接触面有 3x3 个不同位置, 4 (0 deg, 30 deg, 60 deg, 90 deg) 中不同的接触角度. 共有 36 种不同的位姿

tPSFNet 训练的数据集: 每个接触面只选择一种位置 idx=16.
tSRNet  训练的数据集: 所有
"""


def getContactTactileSeqs(tactile_LRs, theshold_scale=0.1, sample_count=-1):
    """ 返回满足阈值的触觉序列
    Arg:
        tactile_LRs: (seqs, 48)
    Return:
        LR_x, LR_y, LR_z, LRs
        LR_x : (seqs, 4, 4)
        LRs : (seqs, 3, 4, 4)
    """
    ret_LR_x, ret_LR_y, ret_LR_z, ret_LR = [], [], [], []
    
    
    # 获取 最大压力值
    tactile_z_max, max_idx = 0, 0
    for i in range(tactile_LRs.shape[0]):
        LR = tactile_LRs[i, :].reshape(16,3)
        LR_x, LR_y, LR_z = LR[:, 0].reshape(4, 4), LR[:, 1].reshape(4, 4), LR[:, 2].reshape(4, 4)
        if LR_z.sum() > tactile_z_max:
            tactile_z_max = LR_z.sum()
            max_idx = i
    
    # 获取满足 阈值的触觉序列
    for i in range(max_idx):
        LR = tactile_LRs[i, :].reshape(16,3)
        LR_x, LR_y, LR_z = LR[:, 0].reshape(4, 4), LR[:, 1].reshape(4, 4), LR[:, 2].reshape(4, 4)
        if LR_z.sum() > tactile_z_max*theshold_scale:
            ret_LR_x.append(np.flip(LR_x, axis=0))
            ret_LR_y.append(np.flip(LR_y, axis=0))
            ret_LR_z.append(np.flip(LR_z, axis=0))
            
            # ret_LR_x.append(LR_x)
            # ret_LR_y.append(LR_y)
            # ret_LR_z.append(LR_z)
            
            ret_LR.append([np.flip(LR_x, axis=0), np.flip(LR_y, axis=0), np.flip(LR_z, axis=0)])
                
    ret_LR_x, ret_LR_y, ret_LR_z, ret_LR = np.array(ret_LR_x), np.array(ret_LR_y), np.array(ret_LR_z), np.array(ret_LR)
    
    # 选择 sample_count 个 数据
    if sample_count > 0:
        sample_count = min(sample_count, ret_LR_z.shape[0])
        sample_idx = np.linspace(0, ret_LR_z.shape[0]-1, sample_count).astype(np.int16)
        ret_LR_x, ret_LR_y, ret_LR_z, ret_LR = ret_LR_x[sample_idx, :, :], ret_LR_y[sample_idx, :, :], ret_LR_z[sample_idx, :, :], ret_LR[sample_idx, :, :] 
    
    return ret_LR_x, ret_LR_y, ret_LR_z, ret_LR

def depth2tactile(dataset):
    for data in dataset:
        data['depth'] = data['depth'] * data['LR'][2].sum() * 0.01
    return dataset

def augmentData(ret_dataset):
    # TODO: 加高斯噪声
    aug_ret_dataset = []
    for data in ret_dataset:
        # original data
        aug_ret_dataset.append({
            'LR': data['LR'],
            'depth' : data['depth']
        })
        
        # ori 90 
        LR_ori90 = np.array([np.rot90(data['LR'][0], 1), 
                             np.rot90(data['LR'][1], 1), 
                             np.rot90(data['LR'][2], 1)])
        depth_ori90 = np.rot90(data['depth'], 1)
        aug_ret_dataset.append({
                'LR': LR_ori90,
                'depth' : depth_ori90
        })
        
        # ori 180
        LR_ori180 = np.array([np.rot90(data['LR'][0], 2), 
                              np.rot90(data['LR'][1], 2), 
                              np.rot90(data['LR'][2], 2)])
        depth_ori180 = np.rot90(data['depth'], 2)
        aug_ret_dataset.append({
                'LR': LR_ori180,
                'depth' : depth_ori180
        })
        
        # ori 270
        LR_ori270 = np.array([np.rot90(data['LR'][0], 3), 
                              np.rot90(data['LR'][1], 3), 
                              np.rot90(data['LR'][2], 3)])
        depth_ori270 = np.rot90(data['depth'], 3)
        aug_ret_dataset.append({
                'LR': LR_ori270,
                'depth' : depth_ori270
        })
    return aug_ret_dataset

def loadRawDataset(dataset_filepath, sample_cnt, is_sample_idx, idx_threshold_scale=0.3, depth_pixel=100, is_aug_data=False):
    """ 加载原始数据
        sample_cnt          : 整个tapping过程中, 选择几个frame
        is_sample_idx       : 是否只选择特定 idx 的数据
        idx_threshold_scale : 选取的最小frame是最大frame的倍数. 设值过小可能由于噪声的存在导致不准确
    """
    dataset = []
    raw_data = np.load(dataset_filepath, allow_pickle=True)
    for data in raw_data:
        data = data.item()
        sample_idx = data['sample_idx']
        # print(sample_idx)
        tactile_depth = data['depth']                              # (depth_pixel, depth_pixel)
        tactile_depth[tactile_depth>(tactile_depth.min()*0.5+tactile_depth.max()*0.5)]    = 1
        tactile_depth[~(tactile_depth>(tactile_depth.min()*0.5+tactile_depth.max()*0.5))] = 0
        tactile_depth = cv2.resize(tactile_depth, (depth_pixel, depth_pixel), cv2.INTER_LINEAR)
        
        tactile_LRs = np.array(data['LRs'])       # (seqs, 48)
        _, _, _, LR_seqs = getContactTactileSeqs(tactile_LRs, theshold_scale=idx_threshold_scale, sample_count=sample_cnt)
        
        # print("idx:{}, raw_LR_seqs shape: {}, LR_Seqs shape : {}".format(sample_idx, tactile_LRs.shape, LR_seqs.shape))
        
        for idx in range(LR_seqs.shape[0]):
            dataset.append({
                'LR': LR_seqs[idx],
                'depth' : tactile_depth
            })

    ret_dataset = []
    if isinstance(is_sample_idx, list):
        for idx in is_sample_idx:
            assert idx>=0, "sample index should >= 0 !"
            ret_dataset += dataset[idx*sample_cnt:(idx+1)*sample_cnt]
    else:
        ret_dataset = dataset
    
    if is_aug_data:
        ret_dataset = augmentData(ret_dataset)       # 数据增强
    return ret_dataset


def loadSeqDataset_SR(dataset_filePath, sample_cnt, idx_threshold_scale=0.3, depth_pixel=100):
    """ 
    不同的移动位置 + 不同的旋转角度
    -----------------------> y
    |   [ 0~ 3] [ 4~ 7] [ 8~11]
    |   [12~15] [16~19] [20~23]
    |   [24~27] [28~31] [32~35]
    |
    x
    
    """
    rotateCnt = 2              # 旋转需要的seqs
    dataset = []
    raw_data = np.load(dataset_filePath, allow_pickle=True)
    print(len(raw_data))
    for i in range(9):
        for j in range(3):
            data_rot0  = raw_data[4*i + j].item()
            data_rot30 = raw_data[4*i + j+1].item()
            
            tactile_depth = data_rot30['depth']                           # (depth_pixel, depth_pixel)
            tactile_depth[tactile_depth>(tactile_depth.min()*0.5+tactile_depth.max()*0.5)]    = 1
            tactile_depth[~(tactile_depth>(tactile_depth.min()*0.5+tactile_depth.max()*0.5))] = 0
            tactile_depth = cv2.resize(tactile_depth, (depth_pixel, depth_pixel), cv2.INTER_LINEAR)
        
            tactile_LRs_rot0 = np.array(data_rot0['LRs'])       # (seqs, 48)
            _, _, _, LR_rot0 = getContactTactileSeqs(tactile_LRs_rot0, theshold_scale=idx_threshold_scale, sample_count=sample_cnt)
            LR_rot0 = LR_rot0[-1]
            
            tactile_LRs = np.array(data_rot30['LRs'])       # (seqs, 48)
            _, _, _, LR_seqs = getContactTactileSeqs(tactile_LRs, theshold_scale=idx_threshold_scale, sample_count=sample_cnt)

            for idx in range(LR_seqs.shape[0]):
                dataset.append({
                    'LR_0': LR_rot0,
                    'LR_1': LR_seqs[idx],
                    'depth' : tactile_depth
                })
        
        data_rot0  = raw_data[4*i + 2].item()
        data_rot30 = raw_data[4*i + 1].item()
        
        tactile_depth = data_rot30['depth']                           # (depth_pixel, depth_pixel)
        tactile_depth[tactile_depth>(tactile_depth.min()*0.5+tactile_depth.max()*0.5)]    = 1
        tactile_depth[~(tactile_depth>(tactile_depth.min()*0.5+tactile_depth.max()*0.5))] = 0
        tactile_depth = cv2.resize(tactile_depth, (depth_pixel, depth_pixel), cv2.INTER_LINEAR)
    
        tactile_LRs_rot0 = np.array(data_rot0['LRs'])       # (seqs, 48)
        _, _, _, LR_rot0 = getContactTactileSeqs(tactile_LRs_rot0, theshold_scale=idx_threshold_scale, sample_count=sample_cnt)
        LR_rot0 = LR_rot0[-1]
        
        tactile_LRs = np.array(data_rot30['LRs'])       # (seqs, 48)
        _, _, _, LR_seqs = getContactTactileSeqs(tactile_LRs, theshold_scale=idx_threshold_scale, sample_count=sample_cnt)

        for idx in range(LR_seqs.shape[0]):
            dataset.append({
                'LR_0': LR_rot0,
                'LR_1': LR_seqs[idx],
                'depth' : tactile_depth
            })  
    print(len(dataset))    
    return dataset    

if __name__ == "__main__":
    root_path = "/code"
    dataset_dir = root_path + '/data/rotateDataset/'
    file_path = dataset_dir + 'C.npy'
    loadRawDataset(dataset_filepath=file_path, sample_cnt=16, is_sample_idx=10)
    
    # [0, 6, 7, 8]
    
    """
    all_dataset = []
    print(root_path)
    for root, ds, fs, in os.walk(dataset_dir):
        for f in fs:
            obj_name, suffix_dot = os.path.splitext(f)
            if suffix_dot == '.npy':
                dataset_file = dataset_dir + '/' + f
                print(dataset_file)
                dataset = loadSeqDataset_SR(dataset_filePath=dataset_file, sample_cnt=16)
                all_dataset = all_dataset + dataset

    print(len(all_dataset))
    np.save(root_path + '/Dataset/SeqsDataset/' + 'seqs_rotate_2_depth.npy', all_dataset)
    
    os._exit(0)
    """

    # dataset = loadSeqDataset_SR(dataset_filePath=dataset_dir+'/P.npy', sample_cnt=16)

    # import matplotlib.pyplot as plt
    # fig = plt.figure(tight_layout=True)
    # ax1 = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)

    # for data in dataset:
    #     # data = data.item()
    #     LR_0 = data['LR_0'] / 100
    #     LR_1 = data['LR_1'] / 100
    #     tactile_depth = data['depth']

    #     LR_vmin, LR_vmax = 0, 12
    #     HR_vmin, HR_vmax = 0, 1

    #     # print(LR_1[2])
    #     ax1.cla(), ax2.cla(), ax3.cla()

    #     ax1.imshow(LR_0[2], vmin=LR_vmin, vmax=LR_vmax)
    #     ax2.imshow(LR_1[2], vmin=LR_vmin, vmax=LR_vmax)
    #     ax3.imshow(tactile_depth, vmin=HR_vmin, vmax=HR_vmax)

    #     plt.savefig('out.png')
