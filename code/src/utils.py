# src/utils.py
import os
import random
import numpy as np
import torch
import yaml
import argparse
# Load configuration
from omegaconf import OmegaConf

config_path = os.path.join('configs', 'config.yaml')
config = OmegaConf.load(config_path)

SAVED_DIR = config.output.checkpoint_dir

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, file_name='fcn_resnet50_best_model.pt'):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)



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