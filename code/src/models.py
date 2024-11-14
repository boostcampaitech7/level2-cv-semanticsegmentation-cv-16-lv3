import os
import torch
import torch.nn as nn
from torchvision import models
import yaml

# Load configuration
config_path = os.path.join('configs', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    
SAVED_DIR = config['output']['checkpoint_dir']

# def get_model(model_name, num_classes, pretrained=True):
#     model = models.segmentation.__dict__[model_name](pretrained=pretrained)
#     model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
#     return model

# def get_model(model_name, num_classes, pretrained=True):
#     # Load the segmentation model with a specified backbone
#     model = models.segmentation.__dict__[model_name](pretrained=pretrained)

#     # Get the output channels of the backbone
#     backbone_out_channels = model.backbone[-1].out_channels  # Last layer output channels

#     # Adjust the classifier to match the number of classes
#     model.classifier[4] = nn.Conv2d(backbone_out_channels, num_classes, kernel_size=1)
    
#     return model

def get_model(model_name, num_classes, pretrained=True):
    # Load the segmentation model
    model = models.segmentation.__dict__[model_name](weights=pretrained)

    # Adjust the classifier's output layer to match the number of classes
    if hasattr(model.classifier[-1], 'out_channels'):
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    else:
        raise AttributeError("The classifier structure doesn't match the expected layout.")

    return model