import os
import matplotlib.pyplot as plt
import torch
from utils.data_loader import DataLoader
from PIL import Image
import segmentation_models_pytorch as smp
from matplotlib.patches import Polygon, Patch
import numpy as np
import cv2

from code.models.base_model import UnetModel, DeepLabV3PlusModel,UnetPlusPlus, DeepLabV3PlusModel_channel0

PALETTE = [
    (255, 0, 0), (119, 11, 32), (0, 0, 142), (50, 50, 255), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (50, 100, 200), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (255, 30, 90), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
class_to_color = {cls: tuple(color) for cls, color in zip(CLASSES, PALETTE)}

class PredViz:
    def __init__(self, data_dir, user_selected_id):
        self.data_dir = data_dir
        self.user_selected_id = user_selected_id
        self.device = "cpu"
        self.model_classes = {
            "Unet" : UnetModel,
            "DeepLabV3Plus": DeepLabV3PlusModel,
            "DeepLabV3PlusModel_channel0": DeepLabV3PlusModel_channel0,
            "UnetPlusPlus": UnetPlusPlus
        }

    # 모델 select 버튼 활성화를 위한 선택지 return
    def select_model(self):
        return ['Unet', 'DeepLabV3Plus', 'DeepLabV3PlusModel_channel0', 'UnetPlusPlus']
    
    # 해당 모델의 pt파일 선택지 제공
    def select_pt(self, model):
        pt_list = os.listdir(os.path.join('../checkpoints', model))
        return pt_list
    
    # 모델 및 학습된 가중치 load
    def model_load(self, model_name, pt):
        checkpoint_path = os.path.join('../checkpoints', model_name, pt)
        model = torch.load(checkpoint_path, map_location=self.device)
        model.to(self.device)
        return model

    # 실제 GT와 prediction 시각화
    def prediction_viz(self, model):
        model.eval()
        imgs_path = os.path.join("../data/train/DCM", self.user_selected_id)
        data_loader = DataLoader("../data/")
        
        fig, ax = plt.subplots(2,2, figsize = (12,12))
        ax = ax.flatten()

        for idx, img_path in enumerate(os.listdir(imgs_path)):
            json_path = data_loader.get_json_path(os.path.join(self.user_selected_id,img_path))
            annotations = data_loader.load_json(json_path)["annotations"]

            # 모델에 넣을 이미지
            img = cv2.imread(os.path.join(imgs_path,img_path))
            img = img / 255.0
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()

            # 시각화용 이미지
            viz_img = Image.open(os.path.join(imgs_path,img_path)).convert('RGB')
            print(f'img_size{viz_img.size}')
            
            img = img.unsqueeze(0)

            # 모델 prediction 반환
            with torch.no_grad():
                outputs = model(img)
                outputs = torch.sigmoid(outputs)
                outputs = outputs.squeeze(0)
                outputs = (outputs > 0.5).detach().cpu()

            ax[idx * 2].imshow(viz_img)
            ax[idx * 2].set_title(f"GT: {self.user_selected_id}:{img_path}")
            ax[idx * 2].axis("off")

            # GT 시각화
            for annotation in annotations:
                points = annotation["points"]
                label = annotation["label"]
                color = [c / 255.0 for c in class_to_color.get(label, (0, 0, 0))]
                polygon = Polygon(points, closed=True, linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
                ax[idx * 2].add_patch(polygon)

            ax[idx * 2 + 1].imshow(viz_img)
            ax[idx * 2 + 1].set_title(f"Prediction {self.user_selected_id}:{img_path}")
            ax[idx * 2 + 1].axis("off")

            # prediction 시각화
            outputs_np = outputs.cpu().numpy()
            for class_idx, class_name in enumerate(CLASSES):
                class_mask = outputs_np[class_idx]
                if class_idx == 0:
                    print(f'pred_size : {class_mask.shape}')

                x_indices, y_indices = np.where(class_mask)
                points = [[y, x] for x, y in zip(x_indices, y_indices)]
                
                if not points:
                    continue

                else:
                    color = [c / 255.0 for c in class_to_color.get(class_name, (0, 0, 0))]
                    polygon = Polygon(points, closed=True, linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
                    ax[idx * 2 + 1].add_patch(polygon)

            # 범례 추가
            legend_patches = [Patch(color=[c / 255.0 for c in color], label=cls) for cls, color in class_to_color.items()]
            fig.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, title="Classes"
                   ,title_fontsize=20, prop={'size': 15}
                   )
            
        return fig