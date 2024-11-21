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
from mmsegmentation.configs._base_.datasets.XRayDataset import CLASSES,IND2CLASS,IMAGE_COUNT
from mmengine.config import Config, DictAction
from mmengine.logging import print_log


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train a segmentor')
#     parser.add_argument('--config', default="./mmsegmentation/my_configs/demo_xray.py")
#     parser.add_argument('--work-dir', help='the dir to save logs and models')
#     parser.add_argument(
#         '--resume',
#         action='store_true',
#         default=False,
#         help='resume from the latest checkpoint in the work_dir automatically')
#     parser.add_argument(
#         '--amp',
#         action='store_true',
#         default=True,
#         help='enable automatic-mixed-precision training')
#     parser.add_argument(
#         '--cfg-options',
#         nargs='+',
#         action=DictAction,
#         help='override some settings in the used config, the key-value pair '
#         'in xxx=yyy format will be merged into config file. If the value to '
#         'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
#         'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
#         'Note that the quotation marks are necessary and that no white space '
#         'is allowed.')
#     parser.add_argument(
#         '--launcher',
#         choices=['none', 'pytorch', 'slurm', 'mpi'],
#         default='none',
#         help='job launcher')
#     # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
#     # will pass the `--local-rank` parameter to `tools/train.py` instead
#     # of `--local_rank`.
#     parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
#     args = parser.parse_args()
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(args.local_rank)

#     return args

def main(config):
    # load config
    # print(config)
    # for i in config:
    #     print(i)
    #     print()

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
    cfg.test_dataloader.dataset.pipeline = cfg.test_pipeline

    # opt_type = 'AdamW'
    # cfg.optimizer.type = opt_type
    cfg.optimizer.lr = config.train.lr
    cfg.optimizer.weight_decay = config.train.weight_decay
    cfg.optim_wrapper.optimizer = cfg.optimizer

    for sch in cfg.param_scheduler:
         sch.by_epoch = by_epoch


    max_epoch= config.train.max_epoch
    iter_per_epoch = int(IMAGE_COUNT*0.8/config.train.train_batch_size)
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
        
    
