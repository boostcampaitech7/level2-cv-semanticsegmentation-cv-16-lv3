import streamlit as st
from utils.multi_segment_viz import MultiViz
from utils.segment_viz import viz  # Import the viz function

# index를 사용하여 next버튼과 prev버튼 시, 이미지 변환
if "index" not in st.session_state:
    st.session_state.index = 0

# cnt(1개를 볼지 4개로 볼지)가 변환될 때마다, index 0으로 초기화
def reset_index():
    st.session_state.index = 0

def segment_viewer(image_list, max_length):
    # 1개, 4개를 선택
    cnt = st.sidebar.selectbox('Select Image Number:', ['1', '4'], on_change=reset_index) # 몇개씩 볼지 바꿀 때마다 index초기화
    st.title("Segment Viewer")
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
            fig = viz(user_selected_id = user_selected_id)
            # Streamlit에서 그래프 표시
            if fig is not None:
                st.image(fig, width=680)
                st.session_state.index = image_list.index(user_selected_id) # 해당 이미지의 index로 이동시킴
            else:
                st.write(f"No images found for ID {user_selected_id}")
        
        # 이미지 입력 없을 시
        else:
            # 시각화 함수 호출
            fig = viz(user_selected_id = image_list[st.session_state.index])
            # Streamlit에서 그래프 표시
            if fig is not None:
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
        multi_viz = MultiViz(user_selected_ids = user_selected_ids)
        multi_fig = multi_viz.multi_viz()
        st.image(multi_fig, width = 680)