#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
from torch.utils.data import Dataset
import numpy as np

from utility.raw_data_process import loadRawDataset

class TactileDataSet(Dataset):
    def __init__(self, ):
        pass

    def __getitem__(self, index):
        pass
    
    def __len__(self,):
        pass


class tPSFNetDataSet(Dataset):
    def __init__(self, dataset_dir, sample_cnt, is_sample_idx, is_aug_data=True):
        self.dataset = []
        for root, ds, fs, in os.walk(dataset_dir):
            for f in fs:
                obj_name, suffix_dot = os.path.splitext(f)
                if suffix_dot == '.npy':
                    dataset_file = os.path.join(root, f)
                    self.dataset = self.dataset + loadRawDataset(dataset_file, sample_cnt=sample_cnt, is_sample_idx=is_sample_idx, is_aug_data=is_aug_data)
        
                   
    def __getitem__(self, idx):
        return np.ascontiguousarray(self.dataset[idx]['LR']), np.ascontiguousarray(self.dataset[idx]['depth'])
    
    def __len__(self,):
        return len(self.dataset)


class TactileSRDataset(Dataset):
    def __init__(self, dataset_dir):
        self.SRdataset = np.load(dataset_dir, allow_pickle=True)
        
    def __getitem__(self, idx):
        return np.ascontiguousarray(self.SRdataset[idx].item()['LR']), np.ascontiguousarray(self.SRdataset[idx].item()['HR'])
    
    def __len__(self,):
        return len(self.SRdataset)



class TactileSRDataset_seq(Dataset):
    def __init__(self, dataset_dir):
        self.SRdataset = np.load(dataset_dir, allow_pickle=True)
        
    def __getitem__(self, idx):
        return np.ascontiguousarray(self.SRdataset[idx].item()['LR']), np.ascontiguousarray(self.SRdataset[idx].item()['HR'])
    
    def __len__(self,):
        return len(self.SRdataset)


class singleTapSeqsDataset(Dataset):
    """ 
        一次tapping过程采集的数据.
        这些数据Depth 都是一样的
    """
    def __init__(self, dataset_file, is_sample_idx=6, sample_cnt=10):
        self.dataset = loadRawDataset(dataset_file, sample_cnt=sample_cnt, is_sample_idx=is_sample_idx)

    def __getitem__(self, idx):
        return np.ascontiguousarray(self.dataset[idx]['LR']), np.ascontiguousarray(self.dataset[idx]['depth'])

    
    def __len__(self,):
        return len(self.dataset)


if __name__ == "__main__":
    root_path = '/code'
    dataset_dir = os.path.join(root_path, 'data/rotateDataset')
    tPSFNetDataSet_train = tPSFNetDataSet(dataset_dir, sample_cnt=10, is_sample_idx=[16])
    print(tPSFNetDataSet_train.__len__())

