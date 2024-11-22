from omegaconf import OmegaConf
import os
import os.path as osp
import gc
import torch
import random
import warnings
import numpy as np
import albumentations as A
import argparse
import torch.optim as optim
import wandb

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from trainer import Trainer
from dataset import XRayDataset

from code.loss_functions.loss_selector import LossSelector
from code.scheduler.scheduler_selector import SchedulerSelector
from code.models.model_selector import ModelSelector
from code.utils.utils import set_seed, set_wandb, setup, print_trainable_parameters, sweep_train
from code.utils.split_data import split_image_into_patches


# warnings.filterwarnings('ignore')
    
def main(cfg):
    #wandb 설정.
    set_wandb(cfg)
    
    #이미지 파일명 설정.
    set_seed(cfg.seed)
    
    #이미지png, 라벨.json으로 가져옴.
    fnames, labels = setup(cfg)

    #A모듈에서 aug 에 해당하는 변환함수 가져옴.
    transform = []
    for aug, params in cfg.transform.items():
        if params.get("use", True):
            new_params = {k: v for k, v in params.items() if k != "use"}
            transform.append(getattr(A, aug)(**new_params))

    train_dataset = XRayDataset(fnames,
                                labels,
                                cfg.train.image_root,
                                cfg.train.label_root,
                                fold=cfg.validation.val_fold,
                                transforms=transform,
                                is_train=True,
                                channel_1=cfg.train.channel_1,
                                patch_size= cfg.train.patch_size
                                )
    
    valid_dataset = XRayDataset(fnames,
                                labels,
                                cfg.train.image_root,
                                cfg.train.label_root,
                                fold=cfg.validation.val_fold,
                                transforms=transform,
                                is_train=False,
                                channel_1=cfg.train.channel_1,
                                patch_size=cfg.train.patch_size 
                                )
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=None if cfg.train.patch_size else cfg.train.train_batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=False if cfg.train.patch_size else True,  # patch_size 활성화 시 drop_last=False
    )

    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=None if cfg.train.patch_size else cfg.validation.val_batch_size,
        shuffle=False,
        num_workers=cfg.validation.num_workers,
         drop_last=False,  # Validation은 drop_last=False로 유지
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"you use {device}")

    # model 선택
    model_selector = ModelSelector()
    if cfg.model.parameters.lora_use:
        model = model_selector.get_model(cfg.model.name, **cfg.model.parameters)
        # 어떻게 lora가 적용되는지 확인
        # for name, param in model.named_parameters():
        #     print(name, param.shape, param.requires_grad)
        print_trainable_parameters(model)
    else:
        model = model_selector.get_model(cfg.model.name, **cfg.model.parameters)

    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
        print(f"multi {torch.cuda.device_count()} use")

    model.to(device)

    # optimizer는 고정
    optimizer = optim.Adam(params=model.parameters(),
                           lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    
    # scheduler 선택
    scheduler_selector = SchedulerSelector(optimizer)
    scheduler = scheduler_selector.get_scheduler(cfg.scheduler.name, **cfg.scheduler.parameters)
    
    # loss 선택
    loss_selector = LossSelector()
    criterion = loss_selector.get_loss(cfg.loss.name, **cfg.loss.parameters)

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=valid_loader,
        threshold=cfg.validation.threshold,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        max_epoch=cfg.max_epoch,
        save_dir=cfg.save_dir,
        val_interval=cfg.validation.val_interval
    )

    trainer.train()


if __name__=='__main__':
    #갈비지 컬렉터를 비웁니다.
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    
    #다른 config 를 사용할려면 터미널에 --config 붙이시면 됩니다.ex)python train.py --configs/alternative.yaml
    parser.add_argument("--config", type=str, default="configs/config.yaml")

    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
        print(cfg) #본인이 무엇을 config 에 정의했는지 알 수
    
    # WandB 설정 및 Sweep ID 반환
    sweep_id = set_wandb(cfg)

    if cfg.wandb.use_sweep and sweep_id:
        # Sweep 실행: wandb.agent로 sweep_train 실행
        wandb.agent(sweep_id, function=lambda: sweep_train(cfg))  # count=1로 지정하면 하나씩 실행
    else:
        # 일반 실행
        main(cfg)
    
