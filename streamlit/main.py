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
from utils.multi_segment_viz import MultiViz
import sys
import os


# 사이드바에 페이지 선택 추가
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Segment Viewer", "Point Cloud Viewer"])

# index를 사용하여 next버튼과 prev버튼 시, 이미지 변환
if "index" not in st.session_state:
    st.session_state.index = 0

# cnt(1개를 볼지 4개로 볼지)가 변환될 때마다, index 0으로 초기화
def reset_index():
    st.session_state.index = 0

def main():
    # 선택된 페이지에 따라 실행할 기능 선택
    if page == "Segment Viewer":
        # 1개, 4개를 선택
        cnt = st.sidebar.selectbox('Select Image Number:', ['1', '4'], on_change=reset_index)
        st.title("Segment Viewer")

        data_dir = DataLoader("../data/")
        # 이미지 이름 list 및 최대 index설정
        image_list = [i.split('/')[0] for i in data_dir.get_image_list()]
        max_length = len(image_list)

        # 1개만 볼때,
        if cnt == "1":
            # 1개만 볼 때는 찾아서 볼 수 있게 남겨둠
            user_selected_id = st.text_input("Enter User Selected ID (Leave blank for random selection):", "")
            
            # 이전, 다음 이미지 버튼
            next_clicked = st.button("Next")
            prev_clicked = st.button("Prev")

            # 버튼 클릭시
            if next_clicked:
                if st.session_state.index + 2 < max_length: # 최대 조건 충족 시,
                    st.session_state.index += 2 # image_list에 동일한 Image_id가 2개씩 이기에 2씩 이돈
                else:
                    st.write("No next images available.")
            elif prev_clicked:
                if st.session_state.index > 0:
                    st.session_state.index -= 2
                else:
                    st.write("No previous images available.")
            
            # 특정 이미지를 입력 시
            if user_selected_id != "":
                fig = viz(data_dir=data_dir, user_selected_id = user_selected_id)
                # Streamlit에서 그래프 표시
                if fig is not None:
                    st.image(fig, width=680)
                    st.session_state.index = image_list.index(user_selected_id) # 해당 이미지의 index로 이동시킴
                else:
                    st.write(f"No images found for ID {user_selected_id}")
            
            # 이미지 입력 없을 시
            else:
                # 시각화 함수 호출
                fig = viz(data_dir=data_dir, user_selected_id = image_list[st.session_state.index])
                # Streamlit에서 그래프 표시
                if fig is not None:
                    # st.pyplot(fig)로는 사이즈 변경이 안되어서 st.image로 변경(figsize를 바꿔봐도 안됨)
                    st.image(fig, width=680)
                else:
                    st.write(f"No images found for ID {user_selected_id}")

        # 이미지 4개씩 보려고 할 때
        if cnt == '4':
            next_clicked = st.button("Next")
            prev_clicked = st.button("Prev")

            if next_clicked:
                if st.session_state.index + 8 < max_length:
                    st.session_state.index += 8
                else:
                    st.write("No next images available.")
            elif prev_clicked:
                if st.session_state.index > 0:
                    st.session_state.index -= 8
                else:
                    st.write("No previous images available.")
                    
            user_selected_ids = image_list[st.session_state.index : st.session_state.index + 8]
            multi_viz = MultiViz(data_dir = data_dir, user_selected_ids = user_selected_ids)
            multi_fig = multi_viz.multi_viz()
            st.image(multi_fig, width = 680)
        
            
    elif page == "Point Cloud Viewer":
        st.title("Point Cloud Viewer")
        eda_viewer_main()
        # 여기에 Point Cloud Viewer 관련 추가 기능 구현 가능

if __name__ == "__main__":
    main()
