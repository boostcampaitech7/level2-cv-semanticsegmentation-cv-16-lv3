# src/utils.py
import os
import os.path as osp
import wandb
import torch
import random
import warnings
import numpy as np
import albumentations as A
import yaml
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
    """
    WandB 설정 및 Sweep 초기화
    """
    wandb.login(key=configs.wandb.api_key)
    
    if configs.wandb.use_sweep:
        # Sweep 설정 로드
        with open(configs.wandb.sweep_path, "r") as sweep_file:
            config_sweep = yaml.safe_load(sweep_file)

        # Sweep ID 생성
        sweep_id = wandb.sweep(config_sweep, project=configs.wandb.project_name)
        return sweep_id
    else:
        # 일반 WandB 초기화
        wandb.init(
            entity=configs.wandb.team_name,
            project=configs.wandb.project_name,
            name=configs.wandb.exp_name,
            config={
                "model": configs.model.name,
                "resize": configs.image_size,
                "batch_size": configs.train.train_batch_size,
                "loss_name": configs.loss.name,
                "scheduler_name": configs.scheduler.name,
                "learning_rate": configs.train.lr,
                "epoch": configs.max_epoch,
            },
        )
        return None


def sweep_train(configs):
    """
    Sweep에서 하이퍼파라미터 설정에 따라 학습 실행
    """
    # WandB 초기화 (sweep 실행 시)
    wandb.init()
    # WandB Sweep 설정 값 가져오기
    configs.train.lr = wandb.config.get("train_lr", configs.train.lr)
    configs.train.train_batch_size = wandb.config.get("train_batch_size", configs.train.train_batch_size)
    configs.max_epoch = wandb.config.get("max_epoch", configs.max_epoch)
    configs.model.name = wandb.config.get("model_name", configs.model.name)
    configs.model.parameters.encoder_name = wandb.config.get("model_encoder_name", configs.model.parameters.encoder_name)
    configs.model.parameters.encoder_weights = wandb.config.get("model_encoder_weight", configs.model.parameters.encoder_weights)
    configs.loss.name = wandb.config.get("loss_name", configs.loss.name)
    configs.scheduler.name = wandb.config.get("scheduler_name", configs.scheduler.name)

    # 학습 시작
    from train import main
    main(configs)


def print_trainable_parameters(model):
    # model.parameters()로 파라미터를 가져와서 그 중에서 gradient를 계산할 수 있는 파라미터만 출력
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {total_trainable_params}')

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