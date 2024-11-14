# src/train.py
import os
import yaml
import wandb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import albumentations as A
import datetime
import argparse
from omegaconf import OmegaConf
import gc
# torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Import custom modules
from datasets import XRayDataset
from utils import set_seed,save_model, parse_args
from config_loader import load_config
from train_loader import get_data_loaders
from models import get_model
from utils import set_seed, save_model, parse_args
from loss_function import dice_coef
from validation import validation

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Ignore new version release of albumentations
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


def train(model, data_loader, val_loader, criterion, optimizer,use_wandb =False):
    print(f'Start training..')   
    
    n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(EPOCHS):
        model.train()
        # Initialize train_loss at the start of each epoch
        train_loss = 0.0

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            outputs = model(images)['out']
            
            # loss를 계산합니다.
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                
        # Epoch당 평균 train loss를 기록
        avg_train_loss = train_loss / len(data_loader)
        if use_wandb:
            wandb.log({"train_loss": avg_train_loss}, step=epoch + 1)
             
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                #에포크 번호를  파일명에 추가해서 각 에포크의 최적 모델을 저장
                save_model(model, file_name=f"{SAVED_DIR}/best_model_epoch_{epoch + 1}.pt")
        # Garbage collection 및 GPU 메모리 캐시 정리
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Set random seed
    set_seed(config.training.random_seed)
    
    # Initialize WandB
    if args.use_wandb:
        wandb.init(project=config.project_name, name=config.wandb_name)
    
    # Load data loaders and classes
    train_loader, valid_loader, CLASSES, SAVED_DIR, FILE_NAME = get_data_loaders(config)
    
    # Training parameters
    EPOCHS = config.training.epochs
    VAL_EVERY = config.training.val_every
    
    # Model setup
    model = get_model(config.model.name, len(CLASSES), config.model.pretrained)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        multi_gpu = True
    else:
        print("Using single GPU or CPU for training")
        multi_gpu = False

    model = model.to(device)

    #Wrap the model with DataParallel if multiple GPUs are available
    if multi_gpu:
        model = nn.DataParallel(model)
        print(f"model use {torch.cuda.device_count()} multi gpu ")
    else:
        print("model not use multi GPU")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=1e-6)
    
    # Start training
    train(model, train_loader, valid_loader, criterion, optimizer, use_wandb=False)


