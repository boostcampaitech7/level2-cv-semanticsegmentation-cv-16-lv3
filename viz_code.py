import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
import json
import random

def viz(user_selected_id = None, IMAGE_ROOT = "data/train/DCM", JSON_ROOT = "data/train/outputs_json"):
    # 시각화를 위한 팔레트를 설정
    PALETTE = [
        (255, 0, 0), (119, 11, 32), (0, 0, 142), (50, 50, 255), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (50, 100, 200), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (255, 30, 90), (165, 42, 42),
        (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
        (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
        (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
    ]

    # class명 정의
    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]
    
    # class와 팔레트의 색상을 하나씩 매핑
    class_to_color = {cls: tuple(color) for cls, color in zip(CLASSES, PALETTE)}

    # PNG 파일 경로들을 리스트로 저장
    pngs = [
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _dirs, files in os.walk(IMAGE_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".png"
    ]

    # ID로 이미지 선택
    def find_images_by_id(prefix_id):
        return [img_path for img_path in pngs if os.path.dirname(img_path).startswith(prefix_id)]


    random_id = [i.split('/')[0] for i in pngs]
    
    if user_selected_id is None:
        user_selected_id = random.choice(random_id)  # user_selected_id를 입력 하지 않으면 Random한 이미지 생성
    selected_images = find_images_by_id(user_selected_id)
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax = ax.flatten()

    if selected_images:
        # 선택된 모든 이미지를 시각화
        for idx, img_path in enumerate(selected_images):
            full_json_path = os.path.join(JSON_ROOT, f"{img_path.split('.')[0]}.json")
            full_image_path = os.path.join(IMAGE_ROOT, img_path)  
            try:
                image = Image.open(full_image_path).convert('RGB')
                ax[idx * 2].imshow(image)
                ax[idx * 2].axis('off')
                ax[idx * 2].set_title(f"Selected Image {user_selected_id}")

                with open(full_json_path, 'r') as f:
                    annotations = json.load(f)["annotations"]

                ax[idx * 2 + 1].imshow(image)
                ax[idx * 2 + 1].axis('off')
                ax[idx * 2 + 1].set_title(f"Selected Image With Label {user_selected_id}")

                for annotation in annotations:
                    points = annotation["points"]
                    label = annotation["label"]
                    color = [c / 255.0 for c in class_to_color.get(label)]

                    polygon = Polygon(points, closed=True, linewidth=2, edgecolor = 'black', facecolor=color, alpha = 0.7)
                    ax[idx * 2 + 1].add_patch(polygon)
            except FileNotFoundError:
                print(f"Error: File not found at {full_image_path}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        legend_patches = [Patch(color=[c / 255.0 for c in color], label=cls) for cls, color in class_to_color.items()]
        fig.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(0.9, 0.5), ncol=1, title="Classes")
        plt.show()
    else:
        print(f"No images found for ID {user_selected_id}")
