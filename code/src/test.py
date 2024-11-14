# src/inference.py

import os
import yaml
import json
import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
from omegaconf import OmegaConf

#warning 무시
import warnings
warnings.filterwarnings("ignore")

# NO_ALBUMENTATIONS_UPDATE 환경 변수 설정
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A


# Load configuration
config_path = os.path.join('configs', 'config.yaml')
config = OmegaConf.load(config_path)
print(config)

# Data paths and classes
IMAGE_ROOT_TEST = config.data.image_root_test
CLASSES = config.data.classes
IND2CLASS = {i: v for i, v in enumerate(CLASSES)}

# Inference parameters
TEST_BATCH_SIZE = config.test.batch_size
TEST_NUM_WORKERS = config.test.num_workers
TEST_MODEL = config.output.name
RESIZE_HEIGHT = config.test.resize_height
RESIZE_WIDTH = config.test.resize_width
ORIG_HEIGHT = config.test.original_height
ORIG_WIDTH = config.test.original_width
THRESHOLD = config.test.threshold

# Output paths
SAVED_DIR = config.output.checkpoint_dir
OUTPUT_CSV = config.output.output_csv

# Create directories if they don't exist
os.makedirs(SAVED_DIR, exist_ok=True)

model = torch.load(os.path.join(SAVED_DIR, TEST_MODEL))

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT_TEST)
    for root, _dirs, files in os.walk(IMAGE_ROOT_TEST)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}
# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE로 인코딩된 결과를 mask map으로 복원합니다.

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


# Define Dataset Class for Inference
class XRayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.filenames = _filenames
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT_TEST, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  
        
        image = torch.from_numpy(image).float()
            
        return image, image_name

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)['out']
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

tf = A.Resize(512, 512)

# Create test dataset and loader
test_dataset = XRayInferenceDataset(transforms=tf)
test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=TEST_NUM_WORKERS,
    drop_last=False
)

rles, filename_and_class = test(model, test_loader)

preds = []
for rle in rles[:len(CLASSES)]:
    pred = decode_rle_to_mask(rle, height=2048, width=2048)
    preds.append(pred)

preds = np.stack(preds, 0)

# Prepare submission DataFrame
classes, filename = zip(*[x.split("_") for x in filename_and_class])
image_name = [os.path.basename(f) for f in filename]

df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"Submission saved to {OUTPUT_CSV}")
