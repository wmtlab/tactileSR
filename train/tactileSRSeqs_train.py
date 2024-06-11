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
from sklearn.metrics import mean_squared_error
import logging

ROOT_PATH = "/code"
sys.path.append(ROOT_PATH)

from cpu import ConfigArgumentParser, Trainer, save_args, set_random_seed, setup_logger
from cpu import EvalHook, HookBase
from cpu.trainer import Trainer, logger
from cpu.misc import set_random_seed

from config.default import tactileSR_config, tactileSeqs_config, root_path, device
from model.tactileSR_model import TactileSR, TactileSRCNN
from utility.load_tactile_dataset import TactileSRDataset_seq
from utility.tools import calculationPSNR, calculationSSIM

from train.tactileSR_train import Trainer_tactileSR, eval_func, InferenceHook_tactileSR


def build_dataloader(config):
    tactileSRDataset_train = TactileSRDataset_seq(config['train_dataset_dir'])
    tactileSRDataset_test = TactileSRDataset_seq(config['test_dataset_dir'])
    
    train_loader = DataLoader(tactileSRDataset_train, batch_size=config['train_batch_size'], shuffle=True)
    test_loader = DataLoader(tactileSRDataset_test, batch_size=config['test_batch_size'], shuffle=False)
    
    print('train dataset size:',tactileSRDataset_train.__len__())
    print('test dataset size:',tactileSRDataset_test.__len__())
    return train_loader, test_loader


def model_param_init(singleSR_config,  seqsSR_config, seqsSR_model):
    load_checkpoint_dir = seqsSR_config['load_checkpoint_dir']
    checkpoint = torch.load(load_checkpoint_dir, map_location=device)
    
    singleSR_model = TactileSR(scale_factor       =singleSR_config['scale_factor'],
                      seqsCnt                     = singleSR_config['seqsCnt'],
                      axisCnt                     = singleSR_config['axisCnt'],
                      patternFeatureExtraLayerCnt = singleSR_config['patternFeatureExtraLayerCnt'],
                      forceFeatureExtraLayerCnt   = singleSR_config['forceFeatureExtraLayerCnt']
                    ).to(device)
    singleSR_model.load_state_dict(checkpoint['model'], strict=False)
    singleSR_model = singleSR_model.to(device)
    
    seqsSR_model.patternFeatureExtra_layer = singleSR_model.patternFeatureExtra_layer
    seqsSR_model.forceFeatureExtra_layer   = singleSR_model.forceFeatureExtra_layer

    return seqsSR_model
    

def main(config):
    set_random_seed(config['random_seed'])
    train_loader, test_loader = build_dataloader(config)

    # tactileSR_model
    model = TactileSR(scale_factor=config['scale_factor'],
                      seqsCnt                     = config['seqsCnt'],
                      axisCnt                     = config['axisCnt'],
                      patternFeatureExtraLayerCnt = config['patternFeatureExtraLayerCnt'],
                      forceFeatureExtraLayerCnt   = config['forceFeatureExtraLayerCnt']
                    ).to(device)
    
    optimizer    = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_scheduler_step_size'], gamma=config['lr_scheduler_gamma'])
    
    model = model_param_init(tactileSR_config, tactileSeqs_config, model)
    
    trainer = Trainer_tactileSR(config            = config,
                                model             = model,
                                optimizer         = optimizer,
                                lr_scheduler      = lr_scheduler,
                                data_loader       = train_loader,
                                max_epochs        = config['epochs'],
                                work_dir          = config['save_dir'],
                                checkpoint_period = config['checkpoint_period'],
                                )
    
    trainer.register_hooks([
            EvalHook(1, lambda: eval_func(model, test_loader, config)),
        ])
    
    if config['inference_test']:
        trainer.register_hooks([
            InferenceHook_tactileSR(test_loader, config),
        ])

    trainer.train(auto_resume=False)


if __name__ == "__main__":
    main(tactileSeqs_config)


