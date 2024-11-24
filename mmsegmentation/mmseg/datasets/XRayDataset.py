import os
import json

import numpy as np
import cv2
from sklearn.model_selection import GroupKFold

from mmseg.registry import DATASETS, TRANSFORMS, MODELS, METRICS
from mmseg.datasets import BaseSegDataset

from mmcv.transforms import BaseTransform
import warnings
warnings.filterwarnings("ignore", message=".*Please pay attention your ground truth.*")

IMAGE_ROOT_TRAIN = "data/train/DCM/"
IMAGE_ROOT_TEST = "data/test/DCM/"
LABEL_ROOT = "data/train/outputs_json/"


CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

pngs_train = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT_TRAIN)
    for root, _dirs, files in os.walk(IMAGE_ROOT_TRAIN)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons_train = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}
IMAGE_COUNT_TRAIN = len(pngs_train)
pngs_train = sorted(pngs_train)

jsons_train = sorted(jsons_train)

pngs_test = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT_TEST)
    for root, _dirs, files in os.walk(IMAGE_ROOT_TEST)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}
pngs_test = sorted(pngs_test)

@DATASETS.register_module()
class XRayDataset(BaseSegDataset):
    def __init__(self, is_train, **kwargs):       
        self.is_train = is_train   
        super().__init__(**kwargs)

    def load_data_list(self):

        _filenames = np.array(pngs_train)
        _labelnames = np.array(jsons_train)

        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]

        # dummy label
        ys = [0 for fname in _filenames]

        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if self.is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
                
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])

                # skip i > 0
                break

        data_list = []
        for i, (img_path, ann_path) in enumerate(zip(filenames, labelnames)):
            data_info = dict(
                img_path=os.path.join(IMAGE_ROOT_TRAIN, img_path),
                seg_map_path=os.path.join(LABEL_ROOT, ann_path),
            )
            data_list.append(data_info)
        return data_list


@DATASETS.register_module()
class XRayDatasetTest(BaseSegDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data_list(self):

        _filenames = np.array(pngs_test)
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]

        # dummy label
        ys = [0 for fname in _filenames]

        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)

        filenames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):     
            filenames += list(_filenames[y])

        data_list = []
        for i, img_path in enumerate(filenames):
            data_info = dict(
                img_path=os.path.join(IMAGE_ROOT_TEST, img_path),
            )
            data_list.append(data_info)
        return data_list

@TRANSFORMS.register_module()
class LoadXRayAnnotations(BaseTransform):
    def transform(self, result):
        label_path = result["seg_map_path"]
        if 'scale' in result:
            image_size = result['scale']

        else:
            image_size = (1024, 1024)
        scale_x = image_size[1] / result['ori_shape'][1]
        scale_y = image_size[0] / result['ori_shape'][0] 


        # process a label of shape (H, W, NC)
        label_shape = image_size + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"]).astype(np.float64)
 
            points[:, 0] *= scale_x  # x 좌표 변환
            points[:, 1] *= scale_y  # y 좌표 변환
            points = points.astype(np.int32)
 
            # polygon to mask
            class_label = np.zeros(image_size, dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        result["gt_seg_map"] = label
        return result
    
@TRANSFORMS.register_module()
class TransposeAnnotations(BaseTransform):
    def transform(self, result):
        result["gt_seg_map"] = np.transpose(result["gt_seg_map"], (2, 0, 1))

        return result