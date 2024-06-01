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

ROOT_PATH = "/code"
sys.path.append(ROOT_PATH)

from cpu import ConfigArgumentParser, Trainer, save_args, set_random_seed, setup_logger
from cpu import EvalHook, HookBase
from cpu.trainer import Trainer
from cpu.misc import set_random_seed

from config.default import tactileSR_config, root_path, device
from model.tactileSR_model import TactileSR, TactileSRCNN
from utility.load_tactile_dataset import TactileSRDataset
from utility.tools import calculationPSNR, calculationSSIM

class Trainer_tactileSR(Trainer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model     = kwargs.get('model')
        self.HR_scale_num = config['HR_scale_num']
        self.scale_factor = config['scale_factor']
        self.criterion = nn.MSELoss()

    
    def train_cal_loss(self, batch):
        LR, HR = batch
        LR, HR = LR.to(device), HR.to(device)
        LR, HR = LR.type(torch.float32), HR.type(torch.float32) / self.HR_scale_num
        HR = F.interpolate(HR, size=(4*self.scale_factor, 4*self.scale_factor), mode='bilinear', align_corners=False)
        out = self.model(LR)
        
        loss = self.criterion(out, HR)
        loss_dict = {'total_loss': loss}
        return loss, loss_dict


def build_dataloader(config):
    tactileSRDataset_train = TactileSRDataset(config['train_dataset_dir'])
    tactileSRDataset_test = TactileSRDataset(config['test_dataset_dir'])
    
    train_loader = DataLoader(tactileSRDataset_train, batch_size=config['train_batch_size'], shuffle=True)
    test_loader = DataLoader(tactileSRDataset_test, batch_size=config['test_batch_size'], shuffle=False)
    
    print('train dataset size:',tactileSRDataset_train.__len__())
    print('test dataset size:',tactileSRDataset_test.__len__())
    return train_loader, test_loader


def eval_func(model, test_loader, config):
    HR_scale_num   = config['HR_scale_num']
    sensorMaxVaule = config['sensorMaxVaule_factor']
    scale_factor   = config['scale_factor']
    
    test_ave_loss, test_ave_ssim_loss, test_ave_psnr_loss = 0, 0, 0
    mse_loss_func = nn.MSELoss()
    
    model.eval()
    for LR, HR in test_loader:
        LR, HR = LR.to(device), HR.to(device)
        LR, HR = LR.type(torch.float32), HR.type(torch.float32) / HR_scale_num
        HR = F.interpolate(HR, size=(4*scale_factor, 4*scale_factor), mode='bilinear', align_corners=False)
        out = model(LR)
        
        mse_loss = mse_loss_func(out, HR)
        test_ave_loss += mse_loss.item()

        ## TODO: can not handle batch size > 1
        batch_ssim_loss, batch_psnr_loss = 0, 0
        for i in range(out.shape[0]):
            psnr_loss = calculationPSNR(out[i].detach(), HR[i].detach(), maxValue=sensorMaxVaule)
            ssim_loss = calculationSSIM(out[i].detach(), HR[i].detach())
            batch_psnr_loss += psnr_loss
            batch_ssim_loss += ssim_loss
        test_ave_ssim_loss += batch_ssim_loss / out.shape[0]  
        test_ave_psnr_loss += batch_psnr_loss / out.shape[0]
        
    test_ave_loss /= len(test_loader)
    test_ave_ssim_loss /= len(test_loader)
    test_ave_psnr_loss /= len(test_loader)
    
    print(f"==> [test] loss: {test_ave_loss:.4f}, SSIM: {test_ave_ssim_loss:.4f}, PSNR: {test_ave_psnr_loss:.4f}")


class InferenceHook_tPSF(HookBase):
    def __init__(self, dataloader, config):
        self.dataloader = dataloader
    
    def after_epoch(self):
        model = self.trainer.model
        cur_epoch = self.trainer.cur_epoch
        
        ## model info
        scale_factor = model.scale_factor
        seqsCnt      = model.seqsCnt
        axisCnt      = model.axisCnt
        
        print("   -----------------------------------------------------   ")
        print(f"SRmodel config: scale_factor:{scale_factor}, seqsCnt:{seqsCnt}, axisCnt:{axisCnt}")
        print("   -----------------------------------------------------   ")


    def inference_func(self, model, test_loader, config, save_name):
        HR_scale_num   = config['HR_scale_num']
        sensorMaxVaule = config['sensorMaxVaule_factor']
        scale_factor   = config['scale_factor']
        
        test_ave_loss, test_ave_ssim_loss, test_ave_psnr_loss = 0, 0, 0
        mse_loss_func = nn.MSELoss()
        
        model.eval()
        for LR, HR in test_loader:
            LR, HR = LR.to(device), HR.to(device)
            LR, HR = LR.type(torch.float32), HR.type(torch.float32) / HR_scale_num
            HR = F.interpolate(HR, size=(4*scale_factor, 4*scale_factor), mode='bilinear', align_corners=False)
            out = model(LR)
            
            mse_loss = mse_loss_func(out, HR)




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
    
    # tactileSRCNN_model = TactileSRCNN(scale_factor=config['scale_factor'])
    optimizer    = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_scheduler_step_size'], gamma=config['lr_scheduler_gamma'])
    
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
    
    trainer.train(auto_resume=False)
    
        
if __name__ == "__main__":
    main(tactileSR_config)



