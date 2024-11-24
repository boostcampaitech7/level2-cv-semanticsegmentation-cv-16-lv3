import streamlit as st
from utils.pred_viz import PredViz

def prediction_viewer(image_list, max_length):
    # """
    # 기본적으로 base_model에 있는 모델 전제하에 진행했습니다.

    # check point 폴더 구조
    # checkpoint
    # |_ Unet
    # |   |_ pt파일들
    # |
    # |_ DeepLabV3Plus
    # |   |_ pt파일들
    # |
    # |_ UnetPlusPlus
    # |   |_ pt파일들
    # """
    if "pred_index" not in st.session_state:
        st.session_state.pred_index = 0
    st.title('Prediction Viewer')
    next_clicked = st.button("Next")
    prev_clicked = st.button("Prev")

    # 버튼 클릭시
    if next_clicked:
        if st.session_state.pred_index + 2 < max_length:
            st.session_state.pred_index += 2
        else:
            st.write("No next images available.")
    elif prev_clicked:
        if st.session_state.pred_index > 0:
            st.session_state.pred_index -= 2
        else:
            st.write("No previous images available.")
    pred_viz = PredViz("../data/", image_list[st.session_state.pred_index])
    # 모델 선택
    selected_model_name = st.sidebar.selectbox("Select Model", options=pred_viz.select_model())

    if selected_model_name: # 선택된 모델이 있을 경우, 그 모델의 pt파일들을 불러옵니다.
        selected_pt = st.sidebar.selectbox("Select PT File", options = pred_viz.select_pt(selected_model_name))
        selected_model = pred_viz.model_load(selected_model_name, selected_pt)
        viz_with_predict = pred_viz.prediction_viz(selected_model)
        st.pyplot(viz_with_predict)