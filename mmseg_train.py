import os
import json
from collections import  defaultdict

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from mmseg.registry import DATASETS, TRANSFORMS, MODELS, METRICS
from mmseg.models.segmentors import EncoderDecoderWithoutArgmax

from mmseg.models.decode_heads import  FCNHeadWithoutAccuracy, SegformerHeadWithoutAccuracy
from mmseg.models.utils.wrappers import resize
from mmseg.evaluation.metrics import DiceMetric
from mmseg.models.losses import LossByFeatMixIn

from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner, load_checkpoint

from mmsegmentation.configs._base_.datasets.XRayDataset import CLASSES,IND2CLASS
import argparse
from mmengine.config import Config, DictAction

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    # load config
    cfg = Config.fromfile("./mmsegmentation/my_configs/demo_xray.py")
    cfg.launcher = "none"
    cfg.work_dir = "mmseg_results"

    # resume training
    cfg.resume = False

    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()

if __name__ == '__main__':
    main()
