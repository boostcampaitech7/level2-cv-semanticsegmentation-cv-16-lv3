# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import torch.nn.functional as F
import pandas as pd

import albumentations as A

from mmengine.runner import Runner, load_checkpoint
from mmseg.models.utils.wrappers import resize
from mmseg.registry import MODELS
from mmseg.datasets.XRayDataset import CLASSES,IND2CLASS
from mmseg.datasets.XRayInferenceDataset import XRayInferenceDataset, encode_mask_to_rle

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from code.loader.mmseg_config_loader import get_config
    
def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()
            outputs = model(images)        

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")

    return rles, filename_and_class    



def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg

def main():

    # # load config
    # cfg = Config.fromfile(args.config)
    cfg, args = get_config()
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.load_from = args.checkpoint

    if cfg.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    model = MODELS.build(cfg.model)
    checkpoint = load_checkpoint(model, cfg.resume, map_location='cpu')

    size = cfg.model.data_preprocessor.size[0]
    tf = A.Resize(size,size)
    # model.load_state_dict(state_dict)
    test_dataset = XRayInferenceDataset(transforms=tf)        
    test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=cfg.test_dataloader.batch_size,
    shuffle=False,
    num_workers=cfg.test_dataloader.num_workers,
    drop_last=False)

    rles, filename_and_class = test(model, test_loader)
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
    })
    df.to_csv("test_uu.csv", index=False)
if __name__ == '__main__':
    main()



