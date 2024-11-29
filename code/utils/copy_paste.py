from PIL import Image, ImageDraw
import os
import os.path as osp
import numpy as np
import cv2
import random
import json

def copy_paste(image_root, label_root, fnames, labels, image, label, CLASSES, copy_k):
    class2ind = {v: i for i, v in enumerate(CLASSES)}

    randoms = random.choices([i for i in range(len(fnames))], k=copy_k)

    for i in randoms:
        target_image = (
            cv2.imread(os.path.join(image_root, fnames[i])) / 255.0
        )
        target_label_path = os.path.join(label_root, labels[i])
        with open(target_label_path, "r") as f:
            target_annotations = json.load(f)
        target_annotations = target_annotations["annotations"]
        for ann in target_annotations:
            target_c = ann["label"]
            index_c = class2ind[target_c]
            # 복붙할 class 선택
            if index_c in [19, 20, 25, 26]: # 'Trapezium', 'Trapezoid', 'Triquetrum', 'Pisiform'
                points_chk = np.array([tuple(map(int, point)) for point in ann["points"]])
                points = np.array(ann["points"])
                max_coords = np.max(points_chk, axis=0)
                min_coords = np.min(points_chk, axis=0)

                # 복사 위치 계산
                h, w = image.shape[:2]
                copy_height = max_coords[1] - min_coords[1]
                copy_width = max_coords[0] - min_coords[0]

                x = random.randint(0, w - copy_width)
                y = random.randint(0, h - copy_height)

                # 마스크 생성
                img = Image.new("L", (w, h), 0)
                points_list = [tuple(map(int, p)) for p in points]
                ImageDraw.Draw(img).polygon(points_list, outline=0, fill=1)  
                mask = np.array(img)

                image[y : y + copy_height, x : x + copy_width, ...] = target_image[min_coords[1] : max_coords[1], min_coords[0] : max_coords[0], ...]

                ori_label = label[..., index_c]

                ori_label[y : y + copy_height, x : x + copy_width] = mask[min_coords[1] : max_coords[1], min_coords[0] : max_coords[0]]

                label[..., index_c] = ori_label
    return image, label