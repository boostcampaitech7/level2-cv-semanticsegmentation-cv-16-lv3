import os
import albumentations as A
from torch.utils.data import DataLoader
from datasets import XRayDataset

def get_data_loaders(config):
    IMAGE_ROOT_TRAIN = config.data.image_root_train
    LABEL_ROOT_TRAIN = config.data.label_root_train
    IMAGE_ROOT_TEST = config.data.image_root_test
    CLASSES = config.data.classes
    SAVED_DIR = config.output.checkpoint_dir
    FILE_NAME = config.output.name
    
    # Create datasets and data loaders
    tf = A.Resize(512, 512)
    train_dataset = XRayDataset(is_train=True, transforms=tf)
    valid_dataset = XRayDataset(is_train=False, transforms=tf)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        drop_last=False,
    )

    return train_loader, valid_loader, CLASSES, SAVED_DIR, FILE_NAME