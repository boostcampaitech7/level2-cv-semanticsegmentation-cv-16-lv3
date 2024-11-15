# src/config_loader.py
import os
from omegaconf import OmegaConf

def load_config(args):
    # Read config file
    config_path = os.path.join('configs', 'config.yaml')
    config = OmegaConf.load(config_path)
    
    # Override config with argparse arguments
    config.training.num_workers = args.num_workers or config.training.num_workers
    config.training.batch_size = args.batch_size or config.training.batch_size
    config.training.epochs = args.epochs or config.training.epochs
    config.training.learning_rate = args.learning_rate or config.training.learning_rate
    config.training.val_every = args.val_every or config.training.val_every

    return config
