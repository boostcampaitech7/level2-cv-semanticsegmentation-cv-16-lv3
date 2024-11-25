import os
import cv2
import json
import torch
import numpy as np
import os.path as osp
import albumentations as A

from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from code.utils.split_data import split_image_into_patches
from code.utils.copy_paste import copy_paste

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

class XRayDataset(Dataset):
    def __init__(self, fnames, labels, image_root, label_root, fold=0, transforms=None, is_train=True, channel_1=False, patch_size=False, copy_k = 0):
        self.transforms = A.Compose(transforms)
        self.is_train = is_train
        self.image_root = image_root
        self.label_root = label_root
        self.validation_fold = fold
        self.class2ind = {v: i for i, v in enumerate(CLASSES)}
        self.ind2class = {v: k for k, v in self.class2ind.items()}
        self.num_classes = len(CLASSES)
        self.channel_1 = channel_1
        self.patch_size = patch_size
        self.copy_k = copy_k
        
        groups = [osp.dirname(fname) for fname in fnames]
        
        # dummy label
        ys = [0] * len(fnames)
        
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(fnames, ys, groups)):
            if self.is_train:
                if i == self.validation_fold:
                    continue        
                filenames += list(fnames[y])
                labelnames += list(labels[y])
    
            else:
                if i != self.validation_fold:
                    continue
                filenames = list(fnames[y])
                labelnames = list(labels[y])

        
        self.fnames = filenames
        self.labels = labelnames
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, item):
        image_name = self.fnames[item]
        image_path = osp.join(self.image_root, image_name)
        
        if self.channel_1:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # 채널을 유지하면서 이미지 읽기
            image = image[..., np.newaxis] # (2048, 2048) -> (2048, 2048, 1) 차원 추가
        else:
            image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labels[item]
        label_path = osp.join(self.label_root, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = self.class2ind[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        # k가 0이 아니고, train모드일때만 copy and paste 추가
        if self.copy_k != 0 and self.is_train:
            image, label = copy_paste(self.image_root, self.label_root, self.fnames, self.labels, image, label,
                                      CLASSES, self.copy_k)
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # 패치 크기가 설정된 경우 이미지를 패치로 분리
        if self.patch_size:
            image = split_image_into_patches(image, self.patch_size)  # (num_patches, C, patch_size, patch_size)
            label = split_image_into_patches(label, self.patch_size)  # (num_patches, C, patch_size, patch_size)

            # 데이터 타입 확인 후 변환
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()  # float32 변환
            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(label).float()
        else:
            # 채널 순서 변경
            image = image.transpose(2, 0, 1)    # channel first 포맷으로 변경
            label = label.transpose(2, 0, 1)
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()

        return image, label 
    

class XRayInferenceDataset(Dataset):
    def __init__(self, fnames, image_root, transforms=None):
        self.fnames = np.array(sorted(fnames))
        self.image_root = image_root
        self.transforms = transforms
        self.ind2class = {i: v for i, v in enumerate(CLASSES)}
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, item):
        image_name = self.fnames[item]
        image_path = osp.join(self.image_root, image_name)
        
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