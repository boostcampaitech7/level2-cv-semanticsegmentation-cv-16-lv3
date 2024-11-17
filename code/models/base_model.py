import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UnetModel(nn.Module):
    """
    Base Model Unet
    """
    def __init__(self,
                 **kwargs):
        super(UnetModel, self).__init__()
        self.model = smp.Unet(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
class DeepLabV3PlusModel(nn.Module):
    """
    Base Model DeepLabV3Plus
    """
    def __init__(self, **kwargs):
        super(DeepLabV3PlusModel, self).__init__()
        self.model = smp.DeepLabV3Plus(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)

class UnetPlusPlus(nn.Module):
    """
    Base Model UnetPlusPlus
    """
    def __init__(self,
                 **kwargs):
        super(UnetPlusPlus, self).__init__()
        self.model = smp.UnetPlusPlus(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)