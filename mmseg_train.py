import os
import argparse
import logging
import numpy as np
import gc
import torch
from omegaconf import OmegaConf

from tqdm.auto import tqdm
from mmseg.registry import DATASETS, TRANSFORMS, MODELS, METRICS
from mmseg.models.segmentors import EncoderDecoderWithoutArgmax
from mmseg.models.decode_heads import  FCNHeadWithoutAccuracy, SegformerHeadWithoutAccuracy
from mmseg.models.utils.wrappers import resize
from mmseg.evaluation.metrics import DiceMetric
from mmseg.models.losses import LossByFeatMixIn
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmseg.datasets.XRayDataset import CLASSES,IND2CLASS,IMAGE_COUNT_TRAIN
from mmengine.config import Config, DictAction
from mmengine.logging import print_log

def main(config):

    # load config
    # cfg = Config.fromfile(args.config)
    cfg = Config.fromfile(config.model_config)
    cfg.launcher = "none"
    cfg.work_dir = config.save_dir

    # resume training
    # cfg.resume = config.resume
    

    new_size = int(config.image_size)
    new_size = (new_size,new_size)
    by_epoch = config.by_epoch
    
    cfg.model.data_preprocessor.size = new_size
    cfg.log_processor.by_epoch = by_epoch
    cfg.train_pipeline[2].scale = new_size


    cfg.val_pipeline[1].scale = new_size
    cfg.test_pipeline[1].scale = new_size

    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
    cfg.train_dataloader.batch_size = config.train.train_batch_size
    cfg.train_dataloader.num_workers = config.train.num_workers

    cfg.val_dataloader.dataset.pipeline = cfg.val_pipeline
    cfg.val_dataloader.batch_size = config.validation.val_batch_size
    cfg.val_dataloader.num_workers = config.validation.num_workers
    cfg.test_dataloader.dataset.pipeline = cfg.val_pipeline

    # opt_type = 'AdamW'
    # cfg.optimizer.type = opt_type
    cfg.optimizer=dict(
    betas=(
        0.9,
        0.999,
    ),
    lr=config.train.lr,
    type='AdamW',
    weight_decay=config.train.weight_decay)
    cfg.optim_wrapper.optimizer = cfg.optimizer

    for sch in cfg.param_scheduler:
         sch.by_epoch = by_epoch


    max_epoch= config.train.max_epoch
    iter_per_epoch = int(IMAGE_COUNT_TRAIN*0.8/config.train.train_batch_size)
    total_iter = iter_per_epoch*max_epoch
    val_interval= config.validation.val_interval * iter_per_epoch
    cfg.train_cfg= dict(type='IterBasedTrainLoop', max_iters=total_iter, val_interval=val_interval)
    # dict(
    # type='EpochBasedTrainLoop',
    # max_epochs=max_iters,  # 총 epoch 수
    # val_interval=val_interval  # 검증 주기 (매 epoch마다)
    # )

    
    # # 배치마다
    logger_interval = int(iter_per_epoch/4)
    cfg.default_hooks.logger.interval = logger_interval
    cfg.default_hooks.logger.log_metric_by_epoch = by_epoch
    cfg.default_hooks.checkpoint.by_epoch = by_epoch
    cfg.default_hooks.checkpoint.interval = iter_per_epoch
    # cfg.default_hooks.timer.type = 'EpochTimerHook'
    # cfg.work_dir='mmseg_results'

    if config.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        print(optim_wrapper)
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

  
    for cf in cfg:
        pass
        # print(f'{cf}: {cfg[cf]}')
        # print()
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mmseg_config.yaml")

    args = parser.parse_args()    
    with open(args.config, 'r') as f:
        config = OmegaConf.load(f)
        # print(config)
        main(config)
        
    
