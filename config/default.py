import os
import sys
import numpy as np
import argparse

root_path = '/code'

common_config = {
    'root_path'    : root_path,
    'random_seed'  : 42,
    'deterministic': False,
}

tPSFNet_config = {
    'train_batch_size'      : 256,
    'test_batch_size'       : 8,
    'scale_num'             : 100,
    'gama'                  : 1.4,
    'perception_scale'      : None,
    'loss_scale'            : 1e-1,
    'lr'                    : 1e-4,
    'lr_scheduler_step_size': 1,
    'checkpoint_period'     : 1,
    'lr_scheduler_gamma'    : 0.8,
    'weight_decay'          : 1e-5,
    'epochs'                : 51,
    'sample_cnt'            : 32,
    'dataset_dir'           : os.path.join(root_path, 'data/rotateDataset'),
    'save_dir'              : os.path.join(root_path, 'pth/tPSFNet'),
    'is_aug_data'           : True,
    
    ## TODO: lr warmup param
    'warmup_t'       : 1,
    'warmup_by_epoch': True,
    'warmup_mode'    : 'fix',
    'warmup_init_lr' : 1e-5,
    'warmup_factor'  : 1e-4,
    
    ## inference test param
    'inference_test' : True,
    'inference_index' : 36,
    'inference_seqs_length': 64,
    'test_dataset_dir_1' : os.path.join(root_path, 'data/rotateDataset/I.npy'),
    'test_dataset_dir_2' : os.path.join(root_path, 'data/rotateDataset/P.npy'),
}
tPSFNet_config = {**common_config, **tPSFNet_config}


tactileSR_config = {
    'train_batch_size'           : 4,
    'test_batch_size'            : 1,
    'lr'                         : 1e-1,
    'weight_decay'               : 1e-1,
    'lr_scheduler_step_size'     : 1,
    'lr_scheduler_gamma'         : 0.8,
    'HR_scale_num'               : 10,
    'sensorMaxVaule_factor'      : 250,       # calucate the PSNR, maxVaule = sensorMaxVaule_factor /  HR_scale_num
    'epochs'                     : 101,
    'scale_factor'               : 10,
    'patternFeatureExtraLayerCnt': 6,

}

tactileSR_config = {**common_config, **tactileSR_config}



