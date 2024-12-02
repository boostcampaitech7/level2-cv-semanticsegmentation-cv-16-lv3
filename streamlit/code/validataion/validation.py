import torch
import wandb
import torch.nn.functional as F
from tqdm.auto import tqdm
from loss_function import dice_coef
from omegaconf import OmegaConf
import os

config_path = os.path.join('configs', 'config.yaml')
config = OmegaConf.load(config_path)


CLASSES = config.data.classes

def validation(epoch, model, data_loader, criterion, thr=0.5,use_wandb = False):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    val_loss = 0
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            #Dice 계수 계산
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)

    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    avg_val_loss = total_loss / cnt

    # wandb에 로깅
    if use_wandb:
        wandb.log({
            "val_loss": avg_val_loss,
            "val_dice": avg_dice
        }, step=epoch)
    
    return avg_dice