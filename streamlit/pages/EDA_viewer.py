## streamlit/pages/EDA_viewer.py ##
import sys
import os
import streamlit as st
import numpy as np
from utils.data_loader import DataLoader

# Feat: #14, commit:, 24.11.14
from utils.mask_generator import PointCloudGenerator
st.title("EDA 뷰어 📊")

def main():
    # 데이터 로더 초기화
    data_loader = DataLoader("../data/")
    
    # 이미지 파일 목록과 페어 가져오기
    image_files = data_loader.get_image_list()
    image_pairs = data_loader.get_image_pairs(image_files)
    
    if not image_pairs:
        st.error("유효한 이미지 페어를 찾을 수 없습니다.")
        return

    # 전체 데이터셋의 JSON 데이터 수집
    left_jsons = []
    right_jsons = []
    sample_shape = None

    for pair_key in image_pairs:
        # Left 이미지 처리
        json_path_l = data_loader.get_json_path(image_pairs[pair_key]['L'])
        if os.path.exists(json_path_l):
            json_data_l = data_loader.load_json(json_path_l)
            left_jsons.append(json_data_l)
            if sample_shape is None:
                sample_image = data_loader.load_image(image_pairs[pair_key]['L'])
                sample_shape = np.array(sample_image).shape

        # Right 이미지 처리
        json_path_r = data_loader.get_json_path(image_pairs[pair_key]['R'])
        if os.path.exists(json_path_r):
            json_data_r = data_loader.load_json(json_path_r)
            right_jsons.append(json_data_r)

    # 클래스별 포인트 클라우드 생성
    st.header("클래스별 포인트 클라우드 시각화")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("왼손 클래스별 분포")
        left_clouds = PointCloudGenerator.create_class_point_cloud(left_jsons, sample_shape)
        for class_name, cloud in left_clouds.items():
            if cloud.any():  # 비어있지 않은 클래스만 표시
                st.image(cloud, caption=f"Left - {class_name}", use_column_width=True)
    
    with col2:
        st.subheader("오른손 클래스별 분포")
        right_clouds = PointCloudGenerator.create_class_point_cloud(right_jsons, sample_shape)
        for class_name, cloud in right_clouds.items():
            if cloud.any():  # 비어있지 않은 클래스만 표시
                st.image(cloud, caption=f"Right - {class_name}", use_column_width=True)

    # 기존의 이미지 페어 뷰어도 유지
    st.header("이미지 페어 뷰어")
    
if __name__ == "__main__":
    main()