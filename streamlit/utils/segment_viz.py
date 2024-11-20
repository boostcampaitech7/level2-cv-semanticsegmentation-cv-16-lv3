## segment_viz.py ##
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
import random
from io import BytesIO
from utils.data_loader import DataLoader
from utils.inference_viz import decode_rle_to_mask
import numpy as np

# 시각화를 위한 팔레트와 class 정의
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

def viz(user_selected_id=None, cnt = '1', for_gt = True, df = False): # 4개를 볼때, legend가 중복되어 4개가 나오는 것을 막기 위해 cnt인자를 추가
    data_loader = DataLoader("../data/")
    pngs = data_loader.get_image_list()
    available_ids = list(set(img_path.split('/')[0] for img_path in pngs))

    # ID로 이미지 선택
    if user_selected_id is None or user_selected_id not in available_ids:
        user_selected_id = random.choice(available_ids)

    selected_images = [img for img in pngs if img.startswith(user_selected_id)] # ID/o39939.png
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), constrained_layout = True) # constrained_layout 추가하여 여백을 줄임
    ax = ax.flatten()

    if selected_images:
        for idx, img_path in enumerate(selected_images[:2]):  # 최대 4개의 이미지만 시각화
            full_image_path = os.path.join(data_loader.images_dir, img_path)
            full_json_path = data_loader.get_json_path(img_path)
            annotations = data_loader.load_json(full_json_path)["annotations"]
            
            try:
                image = data_loader.load_image(img_path).convert('RGB')
                if for_gt is True:
                    ax[idx * 2].imshow(image)
                    ax[idx * 2].axis('off')
                    ax[idx * 2].set_title(f"Selected Image: {user_selected_id}"
                                        , fontsize=20
                                        )
                    
                    ax[idx * 2 + 1].imshow(image)
                    ax[idx * 2 + 1].axis('off')
                    ax[idx * 2 + 1].set_title(f"Selected Image with Label:{user_selected_id}"
                                            , fontsize = 20
                                            )

                    for annotation in annotations:
                        points = annotation["points"]
                        label = annotation["label"]
                        color = [c / 255.0 for c in class_to_color.get(label, (0, 0, 0))]
                        polygon = Polygon(points, closed=True, linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
                        ax[idx * 2 + 1].add_patch(polygon)
                    
                elif for_gt is False:
                    ax[idx * 2].imshow(image)
                    ax[idx * 2].axis('off')
                    ax[idx * 2].set_title(f"Selected Image with Label:{user_selected_id}"
                                            , fontsize = 20
                                            )

                    for annotation in annotations:
                        points = annotation["points"]
                        label = annotation["label"]
                        color = [c / 255.0 for c in class_to_color.get(label, (0, 0, 0))]
                        polygon = Polygon(points, closed=True, linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
                        ax[idx * 2].add_patch(polygon)

                    ax[idx * 2 + 1].imshow(image)
                    ax[idx * 2 + 1].axis('off')
                    ax[idx * 2 + 1].set_title(f"Selected Image with Pred:{user_selected_id}"
                                            , fontsize = 20
                                            )
                    csv_img_name = [i.split('/')[1] for i in selected_images]
                    for idx, img in enumerate(csv_img_name):
                        selected_img_df = df[df['image_name'].apply(lambda x: x in img)]
                        for _, row in selected_img_df.iterrows():
                            label = row["class"]
                            color = [c / 255.0 for c in class_to_color.get(label)]
                            mask = decode_rle_to_mask(row['rle'], 2048, 2048)

                            # mask에서 1인 값들의 좌표 추출
                            points = np.where(mask == 1) 

                            #### scatter용 #### 
                            y_coords, x_coords = points[0], points[1]  # y, x 좌표 분리

                            # 기존에는 polygon으로 plot했지만 너무 느려서 scatter로 변경, 또한 이유는 모르겠지만, 색도 어둡게 나옴
                            ax[idx * 2 + 1].scatter(x_coords, y_coords, s=0.1, color=color, label=label, alpha=0.4)
            
            except FileNotFoundError:
                print(f"Error: File not found at {full_image_path}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

        legend_patches = [Patch(color=[c / 255.0 for c in color], label=cls) for cls, color in class_to_color.items()]
        if cnt == '4': # cnt인자가 "4"일 때
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight") # fig를 png로 저장 및 여백 최소화
            plt.close(fig)  # fig를 닫아 메모리를 해제
            return buf, legend_patches # 이미지와 legend patch를 반환(추후 multi_viz에서 하나의 legend만 만들기 위해서)
        
        else:
            # 이전과 동일
            fig.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, title="Classes"
                   ,title_fontsize=20, prop={'size': 15}
                   )
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight") # fig를 png로 저장 및 여백 최소화
            plt.close(fig)  # fig를 닫아 메모리를 해제
            return buf
            
    else:
        print(f"No images found for ID {user_selected_id}")
        return None
