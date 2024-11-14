import os
import streamlit as st
import albumentations as A
import torch

from utils import (
    load_augmentations_config,
    get_arguments,
    get_placeholder_params,
    select_transformations,
    show_random_params,
    load_dataset,

)

from visuals import (
    select_image,
    show_credentials,
    show_docstring,
    get_transormations_params,
    label2rgb
)



# 데이터 경로와 클래스 설정
IMAGE_ROOT = "../code/data/train/DCM"
LABEL_ROOT = "../code/data/train/outputs_json"
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

import streamlit as st
import numpy as np
from PIL import Image
import streamlit as st



def main():
    st.title("X-Ray Image Segmentation with Albumentations")
    st.sidebar.subheader("Settings")

    # Load dataset
    dataset = load_dataset(IMAGE_ROOT, LABEL_ROOT, CLASSES)
    
    # Image navigation
    if "index" not in st.session_state:
        st.session_state.index = 0
    
    index = st.sidebar.slider("Select Image Index", 0, len(dataset) - 1, 0)
    # Update session_state.index when slider moves
    if index != st.session_state.index:
        st.session_state.index = index
    
    # Navigation buttons
    col1, col2 = st.sidebar.columns([1, 1])
    if col1.button("Previous"):
        st.session_state.index = max(0, st.session_state.index - 1)
    if col2.button("Next"):
        st.session_state.index = min(len(dataset) - 1, st.session_state.index + 1)

    index = st.session_state.index
    original_image, label = dataset[index]

    # Update slider to reflect the button change
    st.session_state.slider = st.session_state.index

    original_image, label = dataset[st.session_state.index]

    # Display images side by side
    st.subheader("Original Image and Label")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image.permute(1, 2, 0).numpy(), caption="Original Image", width=256)
    with col2:
        st.image(label2rgb(label.numpy()), caption="Label Image", width=256)

if __name__ == "__main__":
    main()

# # Streamlit UI
# def main():
#     st.title("X-Ray Image Segmentation with Albumentations")
#     st.sidebar.subheader("Settings")

#     # Load dataset
#     dataset = load_dataset(IMAGE_ROOT, LABEL_ROOT, CLASSES)
#     # Select Image
#     index = st.sidebar.slider("Select Image Index", 0, len(dataset) - 1, 0)
#     original_image, label = dataset[index]

#     # Display original image and label
#     st.subheader("Original Image and Label")
#     st.image(original_image.permute(1, 2, 0).numpy(), caption="Original Image", width=512)
#     st.image(label2rgb(label.numpy()), caption="Label Image", width=512)


# if __name__ == "__main__":
#     main()


###초기 코드들###
# def main():
#     # get CLI params: the path to images and image width
#     path_to_images, width_original = get_arguments()

#     if not os.path.isdir(path_to_images):
#         st.title("There is no directory: " + path_to_images)
#     else:
#         # select interface type
#         interface_type = st.sidebar.radio(
#             "Select the interface mode", ["Simple", "Professional"]
#         )

#         # select image
#         status, image = select_image(path_to_images, interface_type)
#         if status == 1:
#             st.title("Can't load image")
#         if status == 2:
#             st.title("Please, upload the image")
#         else:
#             # image was loaded successfully
#             placeholder_params = get_placeholder_params(image)

#             # load the config
#             augmentations = load_augmentations_config(
#                 placeholder_params, "configs/augmentations.json"
#             )

#             # get the list of transformations names
#             transform_names = select_transformations(augmentations, interface_type)

#             # get parameters for each transform
#             transforms = get_transormations_params(transform_names, augmentations)

#             try:
#                 # apply the transformation to the image
#                 data = A.ReplayCompose(transforms)(image=image)
#                 error = 0
#             except ValueError:
#                 error = 1
#                 st.title(
#                     "The error has occurred. \
#                 Most probably you have passed wrong set of parameters. \
#                 Check transforms that change the shape of image."
#                 )

#             # proceed only if everything is ok
#             if error == 0:
#                 augmented_image = data["image"]
#                 # show title
#                 st.title("Demo of Albumentations")

#                 # show the images
#                 width_transformed = int(
#                     width_original / image.shape[1] * augmented_image.shape[1]
#                 )

#                 st.image(image, caption="Original image", width=width_original)
#                 st.image(
#                     augmented_image,
#                     caption="Transformed image",
#                     width=width_transformed,
#                 )

#                 # comment about refreshing
#                 st.write("*Press 'R' to refresh*")

#                 # random values used to get transformations
#                 show_random_params(data, interface_type)

#                 # print additional info
#                 for transform in transforms:
#                     show_docstring(transform)
#                     st.code(str(transform))
#                 show_credentials()

#                 # adding generic privacy policy
#                 if "GA" in os.environ:
#                     st.markdown(
#                         (
#                             "[Privacy policy]"
#                             + (
#                                 "(https://htmlpreview.github.io/?"
#                                 + "https://github.com/IliaLarchenko/"
#                                 + "albumentations-demo/blob/deploy/docs/privacy.html)"
#                             )
#                         )
#                     )


# if __name__ == "__main__":
#     main()
