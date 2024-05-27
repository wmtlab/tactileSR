import logging
import os, sys

import torch
import torch.nn as nn
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT_PATH = "/code"
sys.path.append(ROOT_PATH)
from cpu import ConfigArgumentParser, Trainer, save_args, set_random_seed, setup_logger
from cpu import InferenceHook, EvalHook, HookBase
from cpu.trainer import Trainer
from cpu.misc import set_random_seed

from config.default import tPSFNet_config, root_path
from model.tPSFNet import tPSFNet
from utility.load_tactile_dataset import tPSFNetDataSet, singleTapSeqsDataset
from utility.tools import select_gpu_with_least_used_memory, test_gpu
from utility.tools import calculationSSIM 

gpu_idx, device, _, _ = select_gpu_with_least_used_memory()
print(f"Selected GPU Index:{gpu_idx}, device:{device}") 
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'


def build_dataloader(config):
    tPSFNetDataSet_train = tPSFNetDataSet(config['dataset_dir'], sample_cnt=config['sample_cnt'], is_sample_idx=[i for i in range(5, 81)], is_aug_data=config['is_aug_data'])
    tPSFNetDataSet_test = tPSFNetDataSet(config['dataset_dir'], sample_cnt=config['sample_cnt'], is_sample_idx=[i for i in range(0, 5)], is_aug_data=config['is_aug_data'])
    print('train dataset size:',tPSFNetDataSet_train.__len__())
    print('test dataset size:',tPSFNetDataSet_test.__len__())
    
    train_loader = DataLoader(tPSFNetDataSet_train, batch_size=tPSFNet_config['train_batch_size'], shuffle=True)
    test_loader = DataLoader(tPSFNetDataSet_test, batch_size=tPSFNet_config['test_batch_size'], shuffle=False)
    
    if config['inference_test'] is False:
        return train_loader, test_loader, None, None
    
    # inference dataset
    testDataset_1          = singleTapSeqsDataset(config['test_dataset_dir_1'], [config['inference_index']], config['inference_seqs_length'])
    testDataset_2          = singleTapSeqsDataset(config['test_dataset_dir_2'], [config['inference_index']], config['inference_seqs_length'])
    inference_dataloader_1 = DataLoader(testDataset_1, batch_size=1, shuffle=False)
    inference_dataloader_2 = DataLoader(testDataset_2, batch_size=1, shuffle=False)
    
    return train_loader, test_loader, inference_dataloader_1, inference_dataloader_2


def eval_func(model, test_loader, config):
    model.eval()
    mse_loss_ave, ssim_ave = 0, 0
    for batch in test_loader:
        LR, depth = batch
        LR, depth = LR.to(device), depth.to(device)
        LR, depth = LR.type(torch.float32)/config['scale_num'], depth.type(torch.float32)
        depth     = depth.unsqueeze(1)
        HR_tactile, LR_tactile_degrade, ret_psf, ret_alphaBeta = model(LR, depth)

        LR_z       = LR[0][2].cpu().detach().numpy()
        LR_degrade = LR_tactile_degrade[0][0].cpu().detach().numpy()
        ssim_loss  = calculationSSIM(LR_degrade, LR_z)
        mse_loss   = mean_squared_error(LR_degrade, LR_z)
        
        ssim_ave     += ssim_loss
        mse_loss_ave += mse_loss
    
    ssim_ave = ssim_ave/len(test_loader)
    mse_loss_ave = mse_loss_ave/len(test_loader)
    
    print(f"mse_loss_ave:{mse_loss_ave}, ssim_ave:{ssim_ave}")


class InferenceHook_tPSF(HookBase):
    def __init__(self, test_loader_1, test_loader_2):
        self.test_loader_1 = test_loader_1
        self.test_loader_2 = test_loader_2
        
    def after_epoch(self):
        model = self.trainer.model
        cur_epoch = self.trainer.cur_epoch
        
        inference_result_dir = os.path.join(self.trainer.work_dir, 'inference_result')
        if not os.path.exists(inference_result_dir):
            os.makedirs(inference_result_dir)
        
        save_name = os.path.join(inference_result_dir, 'epoch_{}.png'.format(cur_epoch))
        self.inference_func(model, self.test_loader_1, self.test_loader_2, cur_epoch, save_name)
    
    
    def seqs_result(self, model, dataloder, scale_num=100):
        depth_list, LR_z_list, HR_list, LR_degrade_list, PSF_list = [], [], [], [], []
        alpha_list, beta_list, force_list = [], [], []
        cnt = 0
        for LR, depth in dataloder:
            LR, depth = LR.to(device), depth.to(device)
            LR, depth = LR.type(torch.float32)/scale_num, depth.type(torch.float32)
            depth = depth.unsqueeze(1)
            
            HR_tactile, LR_tactile_degrade, ret_psf, ret_alphaBeta = model(LR, depth)
            
            # only use the first data of the batch
            depth         = depth[0].cpu().detach().numpy()
            LR_z          = LR[0][2].cpu().detach().numpy()
            HR            = HR_tactile[0].cpu().detach().numpy()
            LR_degrade    = LR_tactile_degrade[0][0].cpu().detach().numpy()
            psf_img       = ret_psf[0][0].cpu().detach().numpy()
            ret_alphaBeta = ret_alphaBeta[0][0].cpu().detach().numpy()
            
            ssim_loss = calculationSSIM(LR_degrade, LR_z)
            mse_loss = mean_squared_error(LR_degrade, LR_z)
            
            depth_list.append(depth)
            LR_z_list.append(LR_z)
            HR_list.append(HR)
            LR_degrade_list.append(LR_degrade)
            PSF_list.append(psf_img)
            
            alpha_list.append(ret_alphaBeta[0])
            beta_list.append(ret_alphaBeta[1])
            force_list.append(LR_z.sum())
            # print("alphaBeta:{}".format(ret_alphaBeta))
            # print("LR max:{:3f},  LR_degrade max:{:3f}, HR max:{:3f} | LR sum:{:3f}, LR_degrade sum:{:3f}, HR sum:{:3f}, MSE: {:.3f}, ssim:{:.3f}".format(LR_z.max(), LR_degrade.max(), HR.max(), \
                                                            # LR_z.sum(), LR_degrade.sum(), HR.sum()*16/10000, mse_loss, ssim_loss))
            cnt += 1

        return depth_list, LR_z_list, HR_list, LR_degrade_list, PSF_list, alpha_list, beta_list, force_list
    
    def inference_func(self, model, test_loader_1, test_loader_2, config, save_name=None):
        ## -- plot init -- ##
        fig = plt.figure(figsize=(10, 6),tight_layout=True)
        gs = gridspec.GridSpec(2, 4)
        ax1 = fig.add_subplot(gs[0:2, 1:4])
        ax2 = ax1.twinx()
        ax3 = fig.add_subplot(gs[0,0])
        ax4 = fig.add_subplot(gs[1,0])
        xx = np.linspace(start=0, stop=99, num=100)
        yy = np.linspace(start=0, stop=99, num=100)
        X, Y = np.meshgrid(xx, yy)
        
        depth_list_1, LR_z_list_1, HR_list_1, LR_degrade_list_1, PSF_list_1, alpha_list_1, beta_list_1, force_list_1 \
                            = self.seqs_result(model, test_loader_1)
        alpha_list_1, beta_list_1, force_list_1 = np.array(alpha_list_1), np.array(beta_list_1), np.array(force_list_1)
        ax1.plot(force_list_1, alpha_list_1, color='red', label=r'pattern1_$\alpha$')
        ax2.plot(force_list_1, beta_list_1, '--', color='red', label=r'pattern1_$\beta$')
        ax3.imshow(depth_list_1[-1][0])
        ax3.set_title('pattern1')
        
        depth_list_2, LR_z_list_2, HR_list_2, LR_degrade_list_2, PSF_list_2, alpha_list_2, beta_list_2, force_list_2 \
                            = self.seqs_result(model, test_loader_2)
        alpha_list_2, beta_list_2, force_list_2 = np.array(alpha_list_2), np.array(beta_list_2), np.array(force_list_2)
        ax1.plot(force_list_2, alpha_list_2, color='blue', label=r'pattern2_$\alpha$')
        ax2.plot(force_list_2, beta_list_2, '--', color='blue', label=r'pattern2_$\beta$')
        ax4.imshow(depth_list_2[-1][0])
        ax4.set_title('pattern2')
        
        ax1.set_ylim([0.8, 1.5])
        ax2.set_ylim([0.8, 1.5])
        ax1.set_ylabel(r'$\alpha$')
        ax2.set_ylabel(r'$\beta$')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        if save_name is None:
            plt.savefig('out.png')
        else:
            plt.savefig(save_name)
        plt.close()
        del fig
        
        
        
class Trainer_tPSF(Trainer):
    def __init__(self, scale_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model     = kwargs.get('model')
        self.criterion = nn.MSELoss()
        self.scale_num = scale_num
    
    def train_cal_loss(self, batch):
        LR, depth = batch
        LR, depth = LR.to(device), depth.to(device)
        LR, depth = LR.type(torch.float32)/self.scale_num, depth.type(torch.float32)
        depth = depth.unsqueeze(1)
        HR_tactile, LR_tactile_degrade, ret_psf, ret_alphaBeta = self.model(LR, depth)
        
        loss      = self.criterion(LR[:, 2:3], LR_tactile_degrade)
        loss_dict = {'total_loss': loss}
        
        return loss, loss_dict


def main(config):
    set_random_seed(config['random_seed'])
    train_loader, test_loader, test_loader_1, test_loader_2 = build_dataloader(config)
    
    tPSF_model = tPSFNet(gama=config['gama'], 
                         perception_scale=config['perception_scale'], 
                         device=device).to(device)
    
    optimizer = optim.Adam(tPSF_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = StepLR(optimizer, step_size=config['lr_scheduler_step_size'], gamma=config['lr_scheduler_gamma'])

    
    trainer = Trainer_tPSF(scale_num         = config['scale_num'],
                           model             = tPSF_model,
                           optimizer         = optimizer,
                           lr_scheduler      = lr_scheduler,
                           data_loader       = train_loader,
                           max_epochs        = config['epochs'],
                           work_dir          = config['save_dir'],
                           checkpoint_period = config['checkpoint_period'],
                           ##  lr warmup param
                        #    warmup_t       = config['warmup_t'],
                        #    warmup_mode    = config['warmup_mode'],
                        #    warmup_init_lr = config['warmup_init_lr'],
                        #    warmup_factor  = config['warmup_factor'],
                           )
    
    trainer.register_hooks([
            EvalHook(1, lambda: eval_func(tPSF_model, test_loader, config)),
        ])
    
    if config['inference_test']:
        trainer.register_hooks([
            InferenceHook_tPSF(test_loader_1=test_loader_1, test_loader_2=test_loader_2),
        ])

    trainer.train(auto_resume=False)


def inference_func(model, test_loader_1, test_loader_2, config):
    
    ## -- plot init -- ##
    fig = plt.figure(figsize=(10, 6),tight_layout=True)
    gs = gridspec.GridSpec(2, 4)
    ax1 = fig.add_subplot(gs[0:2, 1:4])
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(gs[0,0])
    ax4 = fig.add_subplot(gs[1,0])
    xx = np.linspace(start=0, stop=99, num=100)
    yy = np.linspace(start=0, stop=99, num=100)
    X, Y = np.meshgrid(xx, yy)
    
    depth_list_1, LR_z_list_1, HR_list_1, LR_degrade_list_1, PSF_list_1, alpha_list_1, beta_list_1, force_list_1 \
                            = seqs_result(model, test_loader_1)

    alpha_list_1, beta_list_1, force_list_1 = np.array(alpha_list_1), np.array(beta_list_1), np.array(force_list_1)
    ax1.plot(force_list_1, alpha_list_1, color='red', label=r'pattern1_$\alpha$')
    ax2.plot(force_list_1, beta_list_1, '--', color='red', label=r'pattern1_$\beta$')
    ax3.imshow(depth_list_1[-1][0])
    ax3.set_title('pattern1')
    
    depth_list_2, LR_z_list_2, HR_list_2, LR_degrade_list_2, PSF_list_2, alpha_list_2, beta_list_2, force_list_2 \
                            = seqs_result(model, test_loader_2)
                            
    alpha_list_2, beta_list_2, force_list_2 = np.array(alpha_list_2), np.array(beta_list_2), np.array(force_list_2)
    ax1.plot(force_list_2, alpha_list_2, color='blue', label=r'pattern2_$\alpha$')
    ax2.plot(force_list_2, beta_list_2, '--', color='blue', label=r'pattern2_$\beta$')
    ax4.imshow(depth_list_2[-1][0])
    ax4.set_title('pattern2')

    plt.savefig('out.png')
    plt.close()
    del fig
    

def seqs_result(model, dataloder, scale_num=100):
    depth_list, LR_z_list, HR_list, LR_degrade_list, PSF_list = [], [], [], [], []
    alpha_list, beta_list, force_list = [], [], []
    cnt = 0
    for LR, depth in dataloder:
        LR, depth = LR.to(device), depth.to(device)
        LR, depth = LR.type(torch.float32)/scale_num, depth.type(torch.float32)
        depth = depth.unsqueeze(1)
        
        HR_tactile, LR_tactile_degrade, ret_psf, ret_alphaBeta = model(LR, depth)
        
        # only use the first data of the batch
        depth         = depth[0].cpu().detach().numpy()
        LR_z          = LR[0][2].cpu().detach().numpy()
        HR            = HR_tactile[0].cpu().detach().numpy()
        LR_degrade    = LR_tactile_degrade[0][0].cpu().detach().numpy()
        psf_img       = ret_psf[0][0].cpu().detach().numpy()
        ret_alphaBeta = ret_alphaBeta[0][0].cpu().detach().numpy()
        
        ssim_loss = calculationSSIM(LR_degrade, LR_z)
        mse_loss = mean_squared_error(LR_degrade, LR_z)
        
        depth_list.append(depth)
        LR_z_list.append(LR_z)
        HR_list.append(HR)
        LR_degrade_list.append(LR_degrade)
        PSF_list.append(psf_img)
        
        alpha_list.append(ret_alphaBeta[0])
        beta_list.append(ret_alphaBeta[1])
        force_list.append(LR_z.sum())
        # print("alphaBeta:{}".format(ret_alphaBeta))
        print("LR max:{:3f},  LR_degrade max:{:3f}, HR max:{:3f} | LR sum:{:3f}, LR_degrade sum:{:3f}, HR sum:{:3f}, MSE: {:.3f}, ssim:{:.3f}".format(LR_z.max(), LR_degrade.max(), HR.max(), LR_z.sum(), LR_degrade.sum(), HR.sum()*16/10000, mse_loss, ssim_loss))
        cnt += 1

    return depth_list, LR_z_list, HR_list, LR_degrade_list, PSF_list, alpha_list, beta_list, force_list


def test_tPSF(config):
    scale_num = config['scale_num']
    test_batch_size = 1
    
    test_dataset_filepath_1 = os.path.join(root_path, 'data/rotateDataset/I.npy')
    test_dataset_filepath_2 = os.path.join(root_path, 'data/rotateDataset/P.npy')
    tPSF_model_path = os.path.join(root_path, 'pth/tPSFNet/checkpoints/epoch_5.pth')
    
    tPSF_model = tPSFNet(gama=config['gama'],
                    perception_scale = tPSFNet_config['perception_scale'],
                    device           = device,
                )
    
    checkpoint = torch.load(tPSF_model_path, map_location=device)
    tPSF_model.load_state_dict(checkpoint['model'], strict=False)
    tPSF_model = tPSF_model.to(device)

    
    testDataset_1 = singleTapSeqsDataset(test_dataset_filepath_1, [80], 64)
    testDataset_2 = singleTapSeqsDataset(test_dataset_filepath_2, [64], 64)
    test_loader_1 = DataLoader(testDataset_1, batch_size=test_batch_size, shuffle=False)
    test_loader_2 = DataLoader(testDataset_2, batch_size=test_batch_size, shuffle=False)
    
    print("dataset_1 size: ", testDataset_1.__len__())
    print("dataset_2 size: ", test_loader_2.__len__())

    inference_func(tPSF_model, test_loader_1, test_loader_2, config)


if __name__ == "__main__":
    main(tPSFNet_config)
    # test_tPSF(tPSFNet_config)


