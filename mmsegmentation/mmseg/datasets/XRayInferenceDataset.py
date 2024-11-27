import os
import cv2
import torch
import numpy as np

from mmseg.datasets import BaseSegDataset
from mmcv.transforms import BaseTransform

IMAGE_ROOT_TEST = "data/test/DCM/"

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT_TEST)
    for root, _dirs, files in os.walk(IMAGE_ROOT_TEST)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

class XRayInferenceDataset():
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
        image = image.transpose(2, 0, 1)    # make channel first

        image = torch.from_numpy(image).float()

        return image, image_name
    

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    '''
    mask = mask.cpu().numpy()
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

