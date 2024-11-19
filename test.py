import os
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import albumentations as A
import torch.nn.functional as F
from omegaconf import OmegaConf  # OmegaConf 추가
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import XRayInferenceDataset

# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


def inference(conf, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(conf.test.model_path +"/"+conf.test.model_path2 + "/" +conf.test.model_file,
                       weights_only=True).to(device)
    model.eval()
    
    rles = []
    filename_and_class = []
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="[Inference...]", disable=False) as pbar:
            for images, image_names in data_loader:
                images = images.to(device)  
                if args.channel == 1:
                    images = images[:, 0, :, :]  # 첫 번째 채널 선택 (B, C, H, W)
                    images = images.unsqueeze(1)
                outputs = model(images)
                
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > conf.validation.threshold).detach().cpu().numpy()
                
                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{data_loader.dataset.ind2class[c]}_{image_name}")
                
                pbar.update(1)
                    
    return rles, filename_and_class


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model to use")
    parser.add_argument("--image_root", type=str, default="/data/ephemeral/home/data/test/DCM")
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="./output.csv")
    parser.add_argument("--resize", type=int, default=512, help="Size to resize images (both width and height)")
    parser.add_argument("--channel", type=int, default=3, help="set channel")
    args = parser.parse_args()
    conf = OmegaConf.load("configs/config.yaml")
    print(conf)


    fnames = {
        osp.relpath(osp.join(root, fname), start=conf.test.image_root)
        for root, _, files in os.walk(conf.test.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }
    resize_config = conf.transform.Resize
    tf = A.Resize(height=resize_config.height, width=resize_config.width)

    test_dataset = XRayInferenceDataset(fnames,
                                        conf.test.image_root,
                                        transforms=tf)
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    rles, filename_and_class = inference(
        conf = conf,
        data_loader=test_loader)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    df.to_csv(conf.test.output_csv, index=False)