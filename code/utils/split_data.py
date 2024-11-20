#code/utils/split_data.py
import numpy as np
import torch

def split_image_into_patches(image, patch_size):
    """
    이미지를 non-overlapping patches로 나눕니다.
    
    Args:
        image (np.ndarray): 입력 이미지 (H, W, C) 또는 (H, W).
        patch_size (int): 패치의 크기 (정사각형 기준).
    
    Returns:
        patches (torch.Tensor): 나뉜 패치들. shape: (num_patches, C, patch_size, patch_size)
    """
    # 이미지 크기 확인
    if len(image.shape) == 2:  # grayscale 이미지
        image = np.expand_dims(image, axis=-1)  # 채널 추가
    
    h, w, c = image.shape
    assert h % patch_size == 0 and w % patch_size == 0, f"이미지 크기 ({h}, {w})가 패치 크기 ({patch_size})로 나누어 떨어지지 않습니다."
        
    
    patches = []
    
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch)
    
    patches = np.array(patches,dtype =np.float32)  # (num_patches, patch_size, patch_size, C)
    patches = np.transpose(patches, (0, 3, 1, 2))  # (num_patches, C, patch_size, patch_size) => 딥러닝 모델에 입력하는 텐서 형식에 맞추기 위함(Batch, Channels, Height,width)
    return torch.from_numpy(patches)

