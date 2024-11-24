from utils.inference_viz import InferenceViz
import pandas as pd
import streamlit as st
from utils.segment_viz import viz
import os

# 인덱스 초기화 함수
def reset_index_for_test():
    if "test_index_inf" in st.session_state:
        st.session_state.test_index_inf = 0

def reset_index_for_train():
    if "train_index_inf" in st.session_state:
        st.session_state.train_index_inf = 0


def inference_viewer(train_image_list, test_image_list, train_max_length, test_max_length):
    if "test_index_inf" not in st.session_state:
        st.session_state.test_index_inf = 0

    if "train_index_inf" not in st.session_state:
        st.session_state.train_index_inf = 0

    # csv파일을 1번만 불러오기 위해 사용
    if "test_csv" not in st.session_state:
        st.session_state.test_csv = None

    if "test_csv_path" not in st.session_state:
        st.session_state.test_csv_path = None

    if "train_csv" not in st.session_state:
        st.session_state.train_csv = None

    if "train_csv_path" not in st.session_state:
        st.session_state.train_csv_path = None

    data_type = st.sidebar.selectbox('Select Train or Test', ['Test', 'Train']) # Test데이터의 inference를 볼 일이 더 많을 거 같기에 순서 바꿈
    if data_type == 'Train':
        st.title('Train Inference Viewer') #이름 오타 수정
        resultloader = InferenceViz('../output/train')
        csv_list = resultloader.load_csv_list()
        selected_csv = st.sidebar.selectbox('Select CSV Files', csv_list, on_change=reset_index_for_train)

        if selected_csv.endswith('.csv'):
            next_clicked = st.button("Next")
            prev_clicked = st.button("Prev")
            if next_clicked:
                if st.session_state.train_index_inf + 2 < train_max_length: 
                    st.session_state.train_index_inf += 2 
                else:
                    st.write("No next images available.")
            elif prev_clicked:
                if st.session_state.train_index_inf > 0:
                    st.session_state.train_index_inf -= 2
                else:
                    st.write("No previous images available.")

            # slider를 추가하여 밝기를 조절하는 기능 추가
            selected_alpha = st.sidebar.slider("Select Scatter Alpha", min_value = 0.1, max_value = 1.0, value= 0.7, step= 0.1)
            if st.session_state.train_csv_path is None:
                st.session_state.train_csv_path = os.path.join('../output/train', selected_csv)
        
            if st.session_state.train_csv is None and st.session_state is not None:
                st.session_state.train_csv = pd.read_csv(st.session_state.train_csv_path)

            img = viz(train_image_list[st.session_state.train_index_inf], cnt = '1', for_gt = False, df = st.session_state.train_csv, alpha = selected_alpha)
            st.image(img, width= 680) # viz에서는 image를 반환하므로 pyplot에서 image로 변환
        else:
            st.write('Select CSV')
    
    elif data_type == 'Test':
        st.title("Test Inference Viewer")
        resultloader = InferenceViz('../output/test')
        csv_list = resultloader.load_csv_list()
        selected_csv = st.sidebar.selectbox('Select CSV Files', csv_list, on_change=reset_index_for_test)
        if selected_csv.endswith('.csv'):
            next_clicked = st.button("Next")
            prev_clicked = st.button("Prev")
            if next_clicked:
                if st.session_state.test_index_inf + 2 < test_max_length: 
                    st.session_state.test_index_inf += 2 
                else:
                    st.write("No next images available.")
            elif prev_clicked:
                if st.session_state.test_index_inf > 0:
                    st.session_state.test_index_inf -= 2
                else:
                    st.write("No previous images available.")

            # slider를 추가하여 밝기를 조절하는 기능 추가
            selected_alpha = st.sidebar.slider("Select Scatter Alpha", min_value = 0.1, max_value = 1.0, value= 0.7, step= 0.1)
            if st.session_state.test_csv_path is None:
                st.session_state.test_csv_path = os.path.join('../output/test', selected_csv)
        
            if st.session_state.test_csv is None and st.session_state is not None:
                st.session_state.test_csv = pd.read_csv(st.session_state.test_csv_path)

            img = resultloader.inference_viz(st.session_state.test_csv, test_image_list[st.session_state.test_index_inf], alpha = selected_alpha)
            st.pyplot(img)
        else:
            st.write('Select CSV')