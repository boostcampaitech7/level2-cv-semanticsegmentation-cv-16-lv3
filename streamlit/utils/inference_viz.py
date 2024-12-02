import os
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import DataLoader
from matplotlib.patches import Polygon, Patch
from PIL import Image

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

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

class InferenceViz():
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def load_csv_list(self):
        csv_list = ["Select csv file"]
        for output_csv in os.listdir(self.output_dir):
            if output_csv.endswith('.csv'):
                csv_list.append(output_csv)
        return csv_list

    def inference_viz(self, df, user_selected_id, alpha):
        data_loader = DataLoader('../data')
        pngs = data_loader.get_test_img_list() 
        selected_images = [i for i in pngs if i.startswith(user_selected_id)] 
        csv_img_name = [i.split('/')[1] for i in selected_images] 

        fig, ax = plt.subplots(1, 2, figsize=(12, 12), dpi = 50, constrained_layout = True)

        image1 = Image.open(os.path.join('../data/test/DCM', selected_images[0])).convert('RGB')
        image2 = Image.open(os.path.join('../data/test/DCM', selected_images[1])).convert('RGB')

        ax[0].imshow(image1)
        ax[1].imshow(image2)

        for idx, img in enumerate(csv_img_name):
            selected_img_df = df[df['image_name'] == img]
            ax[idx].set_title(f'{selected_images[idx]}')
            ax[idx].axis('off')
            for _, row in selected_img_df.iterrows():
                label = row["class"]
                color = [c / 255.0 for c in class_to_color.get(label)]
                mask = decode_rle_to_mask(row['rle'], 2048, 2048)

                # mask에서 1인 값들의 좌표 추출
                points = np.where(mask == 1) 

                #### scatter용 #### 
                y_coords, x_coords = points[0], points[1]  # y, x 좌표 분리

                # 기존에는 polygon으로 plot했지만 너무 느려서 scatter로 변경, 또한 이유는 모르겠지만, 색도 어둡게 나옴
                ax[idx].scatter(x_coords, y_coords, s=1/2048, color=color, label=label, alpha=alpha)
                #### scatter용 ####

                #### Polygon용 ####
                # points = list(zip(points[1], points[0]))  
                
                # # Polygon 그리기
                # polygon = Polygon(points, closed=True, linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
                # ax[idx].add_patch(polygon)
                #### Polygon용 ####

        legend_patches = [Patch(color=[c / 255.0 for c in color], label=cls) for cls, color in class_to_color.items()]
        
        fig.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, title="Classes"
                   ,title_fontsize=15, prop={'size': 12}
                   )
        return fig