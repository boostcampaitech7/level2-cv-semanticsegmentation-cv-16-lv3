## main.py ##
import streamlit as st

# 기본 설정
st.set_page_config(
    page_title="Segment_viz 및 Point Cloud 뷰어",
    layout="wide"
)
from pages.EDA_viewer import main as eda_viewer_main
from utils.segment_viz import viz  # Import the viz function
from utils.data_loader import DataLoader
import sys
import os


# 사이드바에 페이지 선택 추가
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Segment Viewer", "Point Cloud Viewer"])

def main():
    # 선택된 페이지에 따라 실행할 기능 선택
    if page == "Segment Viewer":
        st.title("Segment Viewer")
        
        
        data_dir = DataLoader("../data/")
        user_selected_id = st.text_input("Enter User Selected ID (Leave blank for random selection):", "")

        # 시각화 함수 호출
        fig = viz(data_dir=data_dir, user_selected_id=user_selected_id if user_selected_id else None)
        
        # Streamlit에서 그래프 표시
        if fig is not None:
            st.pyplot(fig)
        else:
            st.write(f"No images found for ID {user_selected_id}")
            
    elif page == "Point Cloud Viewer":
        st.title("Point Cloud Viewer")
        eda_viewer_main()
        # 여기에 Point Cloud Viewer 관련 추가 기능 구현 가능

if __name__ == "__main__":
    main()
