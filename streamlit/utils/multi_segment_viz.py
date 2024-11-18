from utils.segment_viz import viz
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

class MultiViz():
    def __init__(self, data_dir, user_selected_ids):
        self.data_dir = data_dir
        self.user_selected_ids = user_selected_ids # list로 출력할 이미지 ID를 받음

    def multi_viz(self):
        figs, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
        axes = axes.flatten()

        for i in range(0, 8, 2):
            fig, legend_patches = viz(data_dir=self.data_dir, user_selected_id= self.user_selected_ids[i], cnt = '4')
            # viz의 return이 이미지임으로 이미지를 open

            # viz의 결과를 다시 시각화
            img = Image.open(fig)
            axes[i//2].imshow(img)
            axes[i//2].axis('off')
        figs.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, title="Classes"
        ,title_fontsize=20, prop={'size': 15}
        )

        # 이미지로 return하기 위한 처리
        bufs = BytesIO()
        figs.savefig(bufs, format="png", bbox_inches="tight") 
        plt.close(figs)
        return bufs