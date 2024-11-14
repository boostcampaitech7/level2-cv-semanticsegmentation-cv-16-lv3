import cv2
import os
import numpy as np
import json
import argparse
import torch
import streamlit as st
from torch.utils.data import Dataset
from typing import List

@st.cache_data
def get_arguments():
    """Return the values of CLI params"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", default="images")
    parser.add_argument("--image_width", default=400, type=int)
    args = parser.parse_args()
    return getattr(args, "image_folder"), getattr(args, "image_width")


@st.cache_data
def get_images_list(path_to_folder: str) -> list:
    """Return the list of images from folder
    Args:
        path_to_folder (str): absolute or relative path to the folder with images
    """
    image_names_list = [
        x for x in os.listdir(path_to_folder) if x[-3:] in ["jpg", "peg", "png"]
    ]
    return image_names_list


@st.cache_data
def load_image(image_name: str, path_to_folder: str, bgr2rgb: bool = True):
    """Load the image
    Args:
        image_name (str): name of the image
        path_to_folder (str): path to the folder with image
        bgr2rgb (bool): converts BGR image to RGB if True
    """
    path_to_image = os.path.join(path_to_folder, image_name)
    image = cv2.imread(path_to_image)
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_dataset(image_root:str, label_root:str, classes:List[str]) -> Dataset:
    """
    이미지를 로드하고 XRayDataset 클래스를 생성합니다.

    Parameters:
    - image_root (str): 이미지 파일이 저장된 경로
    - label_root (str): 라벨 파일(.json)이 저장된 경로
    - classes (list): 이미지 내 클래스 목록

    Returns:
    - XRayDataset 객체: 데이터셋 객체로서 이미지와 라벨을 포함
    """
    # 파일 목록 가져오기
    pngs = {
    os.path.relpath(os.path.join(root, fname), start=image_root)
    for root, _dirs, files in os.walk(image_root)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
            }
    
    jsons = {
    os.path.relpath(os.path.join(root, fname), start=label_root)
    for root, _dirs, files in os.walk(label_root)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
            }
    pngs = sorted(pngs)
    jsons = sorted(jsons)

    # CLASS2IND 매핑 생성
    class2ind = {v: i for i, v in enumerate(classes)}
    # 데이터 경로와 클래스 설정
    IMAGE_ROOT = "../code/data/train/DCM"
    LABEL_ROOT = "../code/data/train/outputs_json"
    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

    # XRayDataset 인스턴스 생성 및 반환
    return XRayDataset(pngs, jsons, class2ind, IMAGE_ROOT,LABEL_ROOT)

class XRayDataset(Dataset):
    def __init__(self, pngs: List[str], jsons: List[str], class2ind, image_root: str, label_root: str, is_train=True, transforms=None):
        self.filenames = pngs
        self.labelnames = jsons
        self.class2ind = class2ind  # class2ind를 인스턴스 변수로 저장
        self.image_root = image_root
        self.label_root = label_root
        self.is_train = is_train
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(self.class2ind), )
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
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tensor 변환
        image = image.transpose(2, 0, 1)  # channel first 포맷으로 변경
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label

def upload_image(bgr2rgb: bool = True):
    """Uoload the image
    Args:
        bgr2rgb (bool): converts BGR image to RGB if True
    """
    file = st.sidebar.file_uploader(
        "Upload your image (jpg, jpeg, or png)", ["jpg", "jpeg", "png"]
    )
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@st.cache_data
def load_augmentations_config(
    placeholder_params: dict, path_to_config: str = "configs/augmentations.json"
) -> dict:
    """Load the json config with params of all transforms
    Args:
        placeholder_params (dict): dict with values of placeholders
        path_to_config (str): path to the json config file
    """
    with open(path_to_config, "r") as config_file:
        augmentations = json.load(config_file)
    for name, params in augmentations.items():
        params = [fill_placeholders(param, placeholder_params) for param in params]
    return augmentations
    


def fill_placeholders(params: dict, placeholder_params: dict) -> dict:
    """Fill the placeholder values in the config file
    Args:
        params (dict): original params dict with placeholders
        placeholder_params (dict): dict with values of placeholders
    """
    # TODO: refactor
    if "placeholder" in params:
        placeholder_dict = params["placeholder"]
        for k, v in placeholder_dict.items():
            if isinstance(v, list):
                params[k] = []
                for element in v:
                    if str(element) in placeholder_params:
                        params[k].append(placeholder_params[element])
                    else:
                        params[k].append(element)
            else:
                if v in placeholder_params:
                    params[k] = placeholder_params[v]
                else:
                    params[k] = v
        params.pop("placeholder")
    return params


def get_params_string(param_values: dict) -> str:
    """Generate the string from the dict with parameters
    Args:
        param_values (dict): dict of "param_name" -> "param_value"
    """
    params_string = ", ".join(
        [k + "=" + str(param_values[k]) for k in param_values.keys()]
    )
    return params_string


def get_placeholder_params(image):
    return {
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "image_half_width": int(image.shape[1] / 2),
        "image_half_height": int(image.shape[0] / 2),
    }


def select_transformations(augmentations: dict, interface_type: str) -> list:
    # in the Simple mode you can choose only one transform
    if interface_type == "Simple":
        transform_names = [
            st.sidebar.selectbox(
                "Select a transformation:", sorted(list(augmentations.keys()))
            )
        ]
    # in the professional mode you can choose several transforms
    elif interface_type == "Professional":
        transform_names = [
            st.sidebar.selectbox(
                "Select transformation №1:", sorted(list(augmentations.keys()))
            )
        ]
        while transform_names[-1] != "None":
            transform_names.append(
                st.sidebar.selectbox(
                    f"Select transformation №{len(transform_names) + 1}:",
                    ["None"] + sorted(list(augmentations.keys())),
                )
            )
        transform_names = transform_names[:-1]
    return transform_names


def show_random_params(data: dict, interface_type: str = "Professional"):
    """Shows random params used for transformation (from A.ReplayCompose)"""
    if interface_type == "Professional":
        st.subheader("Random params used")
        random_values = {}
        for applied_params in data["replay"]["transforms"]:
            random_values[
                applied_params["__class_fullname__"].split(".")[-1]
            ] = applied_params["params"]
        st.write(random_values)
