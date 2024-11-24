import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import cv2
import random
from tqdm import tqdm
from albumentations import Compose, Resize
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import GroupKFold
import shutil

IMAGE_ROOT = "/data/ephemeral/home/level2-cv-semanticsegmentation-cv-16-lv3/data/train/DCM"
LABEL_ROOT = "/data/ephemeral/home/level2-cv-semanticsegmentation-cv-16-lv3/data/train/outputs_json"
TEST_IMAGE_ROOT = "/data/ephemeral/home/level2-cv-semanticsegmentation-cv-16-lv3/data/test/DCM"
output_dir = "./clahe_dataset"

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply((image * 255).astype(np.uint8))  
    return image_clahe

class Dataset:
    def __init__(self, image_root, label_root=None, transforms=None):
        self.image_root = image_root
        self.label_root = label_root

        self.filenames = sorted([
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _, files in os.walk(image_root)
            for fname in files if os.path.splitext(fname)[1].lower() == ".png"
        ])
        if label_root:
            self.labelnames = sorted([
                os.path.relpath(os.path.join(root, fname), start=label_root)
                for root, _, files in os.walk(label_root)
                for fname in files if os.path.splitext(fname)[1].lower() == ".json"
            ])
        else:
            self.labelnames = None

        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_name = self.filenames[idx]
        image_path = os.path.join(self.image_root, image_name)

        label_name = self.labelnames[idx] if self.labelnames else None
        label_path = os.path.join(self.label_root, label_name) if self.labelnames else None

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0

        if self.transforms:
            result = self.transforms(image=image)
            image = result["image"]

        return image_name, image, label_name, label_path

def save_clahe_dataset(dataset, dataset_type="train", include_json=True):
    dcm_subdir = os.path.join(output_dir, dataset_type, "DCM")
    json_subdir = os.path.join(output_dir, dataset_type, "outputs_json") if include_json else None


    for image_name, image, label_name, label_path in tqdm(dataset, desc=f"Processing {dataset_type} dataset"):
        dcm_save_path = os.path.join(dcm_subdir, os.path.dirname(image_name))
        os.makedirs(dcm_save_path, exist_ok=True)
        
        image_clahe = apply_clahe(image)
        image_save_path = os.path.join(dcm_save_path, os.path.basename(image_name))
        cv2.imwrite(image_save_path, image_clahe)

        if include_json and label_path:
            json_save_path = os.path.join(json_subdir, os.path.dirname(label_name))
            os.makedirs(json_save_path, exist_ok=True)
            shutil.copy(label_path, os.path.join(json_save_path, os.path.basename(label_name)))

resize_tf = Resize(512, 512)

train_dataset = Dataset(IMAGE_ROOT, LABEL_ROOT, transforms=resize_tf)
test_dataset = Dataset(TEST_IMAGE_ROOT, transforms=resize_tf)

save_clahe_dataset(train_dataset, dataset_type="train", include_json=True)
save_clahe_dataset(test_dataset, dataset_type="test", include_json=False)
