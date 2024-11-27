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
from mmengine.runner import Runner, load_checkpoint
from mmseg.datasets.XRayDataset import CLASSES,IND2CLASS,IMAGE_COUNT_TRAIN
from mmengine.config import Config, DictAction
from mmengine.logging import print_log

from code.loader.mmseg_config_loader import get_config
def main():

    cfg,_ = get_config(is_train=True)
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()
   


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    main()
    
