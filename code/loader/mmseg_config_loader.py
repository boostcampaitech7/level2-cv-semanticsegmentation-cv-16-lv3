import os
import logging

from argparse import ArgumentParser
from mmengine.config import Config
from omegaconf import OmegaConf
from mmseg.datasets.XRayDataset import IMAGE_COUNT_TRAIN
from mmengine.logging import print_log
from mmengine.config import  DictAction
def get_config(is_train=True):
    parser = ArgumentParser(description="Training script for segmentation model")
    parser.add_argument("--config", type=str, default="configs/mmseg_config.yaml", help="Path to config file")
    parser.add_argument("--model_config", type=str, default=None, help="Path to model config file")
    parser.add_argument("--see", type=bool, default=None, help="print config file")
    config_args, _ = parser.parse_known_args()
    with open(config_args.config, 'r') as f:
        config = OmegaConf.load(f)

    MC = os.path.join('mmsegmentation/my_configs', config_args.model_config if config_args.model_config else config.model_config)
    if not os.path.exists(MC):
        raise FileNotFoundError(f"Configuration file '{MC}' not found.")
    cfg = Config.fromfile(MC)
    
    parser.add_argument("--size", type=int, default=None, help="Size of image for data preprocessiog")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers for data loading")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for optimizer")
    parser.add_argument("--val_every", type=int, default=None, help="Frequency of validation")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to save dir")
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument( '--out',type=str,help='The directory to save output prediction for offline evaluation')
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument('--show-dir', help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument('--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument('--cfg-options',nargs='+',action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    c = parser.parse_args()

    cfg.launcher = "none"
    cfg.work_dir = c.save_dir if c.save_dir else config.save_dir

    # resume training
    cfg.resume = os.path.join('mmseg_results',c.checkpoint if c.checkpoint else config.checkpoint)

    size = int(c.size if c.size else config.image_size)
    size = (size, size)
    by_epoch = config.by_epoch

    cfg.model.data_preprocessor.size = size
    cfg.log_processor.by_epoch = by_epoch
    cfg.train_pipeline[2].scale = size


    cfg.val_pipeline[1].scale = size
    cfg.test_pipeline[1].scale = size

    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
    batch_size = c.batch_size if c.batch_size else config.train.batch_size 
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = c.num_workers if c.num_workers else config.train.num_workers 

    cfg.val_dataloader.dataset.pipeline = cfg.val_pipeline
    cfg.val_dataloader.batch_size = config.validation.batch_size
    cfg.val_dataloader.num_workers = config.validation.num_workers
    cfg.test_dataloader.dataset.pipeline = cfg.val_pipeline
    cfg.test_dataloader.batch_size = config.test.batch_size
    cfg.test_dataloader.num_workers = config.test.num_workers

    cfg.optimizer=dict(
    betas=(0.9, 0.999,),
    lr= c.lr if c.lr else config.train.lr,    
    type='AdamW',
    weight_decay=config.train.weight_decay
    )
    cfg.optim_wrapper.optimizer = cfg.optimizer

    for sch in cfg.param_scheduler:
         sch.by_epoch = by_epoch
    
    max_epoch= c.epochs if c.epochs else config.train.max_epoch
    iter_per_epoch = int(IMAGE_COUNT_TRAIN*0.8/batch_size)
    total_iter = iter_per_epoch*max_epoch
    val_interval= (c.val_every if c.val_every else config.validation.val_interval) * iter_per_epoch
    cfg.train_cfg= dict(type='IterBasedTrainLoop', max_iters=total_iter, val_interval=val_interval)

    logger_interval = int(iter_per_epoch/4)
    cfg.default_hooks.logger.interval = logger_interval
    cfg.default_hooks.logger.log_metric_by_epoch = by_epoch
    cfg.default_hooks.checkpoint.by_epoch = by_epoch
    cfg.default_hooks.checkpoint.interval = iter_per_epoch

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

    if c.see == True:
        print("parser input :",c)
        print("cfg : ",cfg)
        cfg.show = True
    else: 
        cfg.show = False
        
    if is_train:
        log_config = dict(
            interval=10,  # 로깅 주기 (10 iterations마다 로깅)
            hooks=[
                dict(
                    type='WandbLoggerHook',  # Wandb 로깅을 위한 Hook
                    init_kwargs=dict(
                        project=config.wandb.project_name,  # Wandb 프로젝트 이름
                        entity=config.wandb.entity_name    # Wandb 사용자 이름
                    ),
                    log_checkpoint=True,  # 체크포인트 로깅 여부
                    log_checkpoint_metadata=True,  # 체크포인트 메타데이터 로깅 여부
                   # num_eval_images=100,  # 평가 샘플링 이미지 수
                ),
            ]
        )

        # `cfg`에 log_config 설정 추가
        cfg.log_config = log_config
    return cfg,c