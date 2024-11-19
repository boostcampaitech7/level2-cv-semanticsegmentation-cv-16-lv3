# src/utils.py
import os
import os.path as osp
import wandb
import torch
import random
import warnings
import numpy as np
import albumentations as A

import argparse
import torch.optim as optim

from tqdm.auto import tqdm
from trainer import Trainer
from dataset import XRayDataset
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from code.loss_functions.loss_selector import LossSelector
from code.scheduler.scheduler_selector import SchedulerSelector
from code.models.model_selector import ModelSelector
# Load configuration
from omegaconf import OmegaConf



#이미지.png 랑 라벨.json을 가져옵니다.
def setup(cfg): 
    # 이미지 파일명
    fnames = {
        osp.relpath(osp.join(root, fname), start=cfg.train.image_root)
        for root, _, files in os.walk(cfg.train.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }

    # label json 파일명
    labels = {
        osp.relpath(osp.join(root, fname), start=cfg.train.label_root)
        for root, _, files in os.walk(cfg.train.label_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".json"
    }

    return np.array(sorted(fnames)), np.array(sorted(labels))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    #GPU 에서 결과의 재현성 보장여부
    torch.backends.cudnn.deterministic = True 
    
    #cuDNN최적화된 알고리즘 선택
    torch.backends.cudnn.benchmark = False



def set_wandb(configs):
    wandb.login(key=configs.wandb.api_key)
    
    wandb.init(
        entity=configs.wandb.team_name, #팀  wandb page생기면.
        project=configs.wandb.project_name,
        name=configs.wandb.exp_name, #진행하는 실험의 이름? 뭔지 모르겠음.
        config={
                'model': configs.model.name,
                'resize': configs.image_size,
                'batch_size': configs.train.train_batch_size,
                'loss_name': configs.loss.name,
                'scheduler_name': configs.scheduler.name,
                'learning_rate': configs.train.lr,
                'epoch': configs.max_epoch
            }
    )
    # Sweep 활성화 여부를 확인
    if configs.wandb.use_sweep:
        # wandb sweep에서 전달된 파라미터를 가져옴
        config_sweep = wandb.config
        
        # Sweep에서 전달된 값으로 configs 업데이트 (수정된 부분)
        configs.train.lr = config_sweep.get("train_lr", configs.train.lr)  
        configs.train.train_batch_size = config_sweep.get("train_batch_size", configs.train.train_batch_size)  
        configs.max_epoch = config_sweep.get("max_epoch", configs.max_epoch)  
        configs.model.name = config_sweep.get("model_name", configs.model.name)  
        configs.model.parameters.encoder_name = config_sweep.get("model_encoder_name", configs.model.parameters.encoder_name)  
        configs.model.parameters.encoder_weights = config_sweep.get("model_encoder_weight",configs.model.parameters.encoder_weights)
        configs.loss.name = config_sweep.get("loss_name", configs.loss.name)  
        configs.scheduler.name = config_sweep.get("scheduler_name", configs.scheduler.name)  
  


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for segmentation model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate for optimizer")
    parser.add_argument("--val_every", type=int, default=None, help="Frequency of validation")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    return parser.parse_args()