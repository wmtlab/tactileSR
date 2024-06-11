import os
import sys
import numpy as np
import argparse

root_path = '/code'

common_config = {
    'root_path'    : root_path,
    'random_seed'  : 42,
    'deterministic': False,
    'scale_num'    : 100,
}

########################################################

tPSFNet_config = {
    'train_batch_size'      : 256,
    'test_batch_size'       : 8,
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
    'save_dir'              : os.path.join(root_path, 'pth/tPSFNet_no_aug'),
    'is_aug_data'           : False,
    
    ## inference test param
    'inference_test' : True,
    'inference_index' : 36,
    'inference_seqs_length': 64,
    'test_dataset_dir_1' : os.path.join(root_path, 'data/rotateDataset/I.npy'),
    'test_dataset_dir_2' : os.path.join(root_path, 'data/rotateDataset/P.npy'),
}
tPSFNet_config = {**common_config, **tPSFNet_config}

########################################################

tactileSR_config = {
    'train_batch_size'      : 32,
    'test_batch_size'       : 8,
    'lr'                    : 1e-3,
    'weight_decay'          : 1e-2,
    'lr_scheduler_step_size': 2,
    'lr_scheduler_gamma'    : 0.8,
    'checkpoint_period'     : 1,
    'HR_scale_num'          : 10,
    'sensorMaxVaule_factor' : 250,  # calucate the PSNR, maxVaule = sensorMaxVaule_factor /  HR_scale_num
    'epochs'                : 51,

    'warmup_t'       : 2000,
    'warmup_by_epoch': True,
    'warmup_mode'    : 'auto',
    'warmup_init_lr' : 1e-5,
    'warmup_factor'  : 1e-4,
    
    'scale_factor'               : 10,
    'seqsCnt'                    : 1,
    'axisCnt'                    : 3,
    'patternFeatureExtraLayerCnt': 6,
    'forceFeatureExtraLayerCnt'  : 1,
    
    'inference_test' : True,
    
    'save_dir'         : os.path.join(root_path, 'pth/tactileSR_single'),
    'train_dataset_dir': os.path.join(root_path, 'data/SRdataset/SRdataset_train.npy'),
    'test_dataset_dir' : os.path.join(root_path, 'data/SRdataset/SRdataset_test.npy'),
    'val_dataset_dir'  : os.path.join(root_path, 'data/SRdataset/SRdataset_validation.npy'),

}
tactileSR_config = {**common_config, **tactileSR_config}

########################################################
tactileSeqs_config = tactileSR_config.copy()
tactileSeqs_config.update({
    'seqsCnt': 7,               # seqs length <= 7
    'axisCnt': 3,
    
    'lr'          : 1e-4,
    'weight_decay': 1e-2,
    'epochs'      : 51,
    
    
    'load_checkpoint_dir' : os.path.join(root_path, 'pth/tactileSR_single/checkpoints/epoch_50.pth'),
    
    'save_dir'         : os.path.join(root_path, 'pth/tactileSeqs_seq_7'),
    'train_dataset_dir': os.path.join(root_path, 'data/SeqsDataset/SRdataset_train_32.npy'),
    'test_dataset_dir' : os.path.join(root_path, 'data/SeqsDataset/SRdataset_test_32.npy'),
    'val_dataset_dir'  : os.path.join(root_path, 'data/SeqsDataset/SRdataset_validation_32.npy'),
})


# TODO: change to another file
# GPU management
from utility.tools import select_gpu_with_least_used_memory, test_gpu
gpu_idx, device, _, _ = select_gpu_with_least_used_memory()
print(f"Selected GPU Index:{gpu_idx}, device:{device}") 
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
