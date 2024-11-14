import matplotlib.pyplot as plt
import wandb

def visualize_segmentation(images, pred_masks, true_masks, classes, step=None, use_wandb=False):
    """
    입력 이미지, 예측 마스크, 실제 마스크를 시각화하여 wandb에 로그합니다.
    
    Args:
        images (Tensor): 배치의 입력 이미지 텐서.
        pred_masks (Tensor): 예측된 마스크 텐서.
        true_masks (Tensor): 실제 마스크 텐서.
        classes (list): 클래스 이름 리스트.
        step (int, optional): 현재 스텝(에포크) 번호.
        use_wandb (bool): wandb 사용 여부.
    """
    fig, axes = plt.subplots(len(images), 3, figsize=(15, 5 * len(images)))
    
    for i, (image, pred_mask, true_mask) in enumerate(zip(images.cpu(), pred_masks, true_masks)):
        image = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)로 변환
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title("Input Image")
        
        axes[i, 1].imshow(pred_mask.squeeze(), cmap="gray")
        axes[i, 1].set_title("Predicted Mask")
        
        axes[i, 2].imshow(true_mask.squeeze(), cmap="gray")
        axes[i, 2].set_title("Ground Truth Mask")
        
        for ax in axes[i]:
            ax.axis("off")
    
    plt.tight_layout()

    # wandb에 이미지 로그
    if use_wandb:
        wandb.log({"Segmentation Results": wandb.Image(fig, caption=f"Epoch {step}")})
    plt.close(fig)  # 메모리 절약을 위해 plt 닫기
