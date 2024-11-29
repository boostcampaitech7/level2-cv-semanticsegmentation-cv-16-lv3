## main.py ##
import streamlit as st

# 기본 설정
st.set_page_config(
    page_title="Segment_viz 및 Point Cloud 뷰어",
    layout="wide"
)
from pages.EDA_viewer import main as eda_viewer_main
from utils.data_loader import DataLoader
from utils.prediction_viewer import prediction_viewer
from utils.segment_viewr import segment_viewer
from utils.inference_viewer import inference_viewer

# 사이드바에 페이지 선택 추가
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Segment Viewer", "Point Cloud Viewer", "Inference Viewer","Prediction Viewer"])

data_dir = DataLoader("../data/")
image_list = [i.split('/')[0] for i in data_dir.get_image_list()]
max_length = len(image_list)

test_image_list = [i.split('/')[0] for i in data_dir.get_test_img_list()]
test_max_length = len(test_image_list)

def main():
    # 선택된 페이지에 따라 실행할 기능 선택
    if page == "Segment Viewer":
        segment_viewer(image_list, max_length)
            
    elif page == "Point Cloud Viewer":
        st.title("Point Cloud Viewer")
        eda_viewer_main()
        # 여기에 Point Cloud Viewer 관련 추가 기능 구현 가능

    elif page == "Inference Viewer":
        # train과 test를 둘 다 사용할 수 있게 해놨는데 train은 csv파일이 없어 실행은 못해봤습니다.
        # 우선 test에 대한 csv(우리가 리더보드에 제출하는 csv파일)은 실행됩니다.
        # 추가적으로 폴더 구조는 level2-cv-semanticsegmentation/output/test에 csv파일이 있어야합니다
        # train도 같이 구현하기 폴더 구조를 살짝 변경해야 합니다.
        inference_viewer(image_list, test_image_list, max_length, test_max_length)

    elif page == "Prediction Viewer":
        # 이 기능을 사용하고 싶으시면, streamlit 폴더 안에 code폴더를 복사해서 붙여 넣어야 합니다.(streamlit/code)
        prediction_viewer(image_list, max_length)

if __name__ == "__main__":
    main()
