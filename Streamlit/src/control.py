import streamlit as st

# TODO: rename and refactor everything


def select_num_interval( # 특정 숫자 범위를 선택하는 슬라이더를 생성
    param_name: str, limits_list: list, defaults, n_for_hash, **kwargs
):
    st.sidebar.subheader(param_name)
    min_max_interval = st.sidebar.slider(
        "",
        limits_list[0],
        limits_list[1],
        defaults,
        key=hash(param_name + str(n_for_hash)), #유일한 문자열을 만들고 이를 해시값으로 변환
    )
    return min_max_interval


def select_several_nums( # 여러 개의 숫자 선택 슬라이더를 생성하는 함수
    param_name, subparam_names, limits_list, defaults_list, n_for_hash, **kwargs
):
    st.sidebar.subheader(param_name)
    result = []
    assert len(limits_list) == len(defaults_list)
    assert len(subparam_names) == len(defaults_list)

    for name, limits, defaults in zip(subparam_names, limits_list, defaults_list):
        result.append(
            st.sidebar.slider(
                name,
                limits[0],
                limits[1],
                defaults,
                key=hash(param_name + name + str(n_for_hash)),
            )
        )
    return tuple(result)


def select_min_max( # 최소값과 최대값을 지정하되, 둘 간의 차이가 min_diff 조건을 만족하도록 조정하는 기능
    param_name, limits_list, defaults_list, n_for_hash, min_diff=0, **kwargs
):
    assert len(param_name) == 2
    result = list(
        select_num_interval(
            " & ".join(param_name), limits_list, defaults_list, n_for_hash
        )
    )
    if result[1] - result[0] < min_diff:
        diff = min_diff - result[1] + result[0]
        if result[1] + diff <= limits_list[1]:
            result[1] = result[1] + diff
        elif result[0] - diff >= limits_list[0]:
            result[0] = result[0] - diff
        else:
            result = limits_list
    return tuple(result)


def select_RGB(param_name, n_for_hash, **kwargs): #RGB 색상을 선택하는 슬라이더
    result = select_several_nums(
        param_name,
        subparam_names=["Red", "Green", "Blue"],
        limits_list=[[0, 255], [0, 255], [0, 255]],
        defaults_list=[0, 0, 0],
        n_for_hash=n_for_hash,
    )
    return tuple(result)


def replace_none(string):# 문자열이 "None"일 경우 이를 실제 None으로 변환하는 간단한 함수
    if string == "None":
        return None
    else:
        return string


def select_radio(param_name, options_list, n_for_hash, **kwargs):# 여러 옵션 중 하나를 선택할 수 있는 라디오 버튼을 제공
    st.sidebar.subheader(param_name)
    result = st.sidebar.radio("", options_list, key=hash(param_name + str(n_for_hash)))
    return replace_none(result)


def select_checkbox(param_name, defaults, n_for_hash, **kwargs): # 체크박스를 통해 True/False 값을 선택할 수 있는 기능을 제공
    st.sidebar.subheader(param_name)
    result = st.sidebar.checkbox(
        "True", defaults, key=hash(param_name + str(n_for_hash))
    )
    return result


# dict from param name to function showing this param
param2func = {
    "num_interval": select_num_interval,
    "several_nums": select_several_nums,
    "radio": select_radio,
    "rgb": select_RGB,
    "checkbox": select_checkbox,
    "min_max": select_min_max,
}
