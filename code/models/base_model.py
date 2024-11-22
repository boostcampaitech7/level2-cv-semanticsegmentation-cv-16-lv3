import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import peft

class UnetModel(nn.Module):
    """
    Base Model Unet
    """
    def __init__(self, **model_parameters):
        super(UnetModel, self).__init__()
        self.model = smp.Unet(**model_parameters)  # default로 smp 라이브러리 사용

        if model_parameters.get("lora_use", False):  
            lora_config = model_parameters.get("lora_config")
            lora_config = peft.LoraConfig(**lora_config)
            self.model = peft.get_peft_model(self.model, lora_config)
            
    def forward(self, x: torch.Tensor):
        return self.model(x)
    
class DeepLabV3PlusModel(nn.Module):
    """
    Base Model DeepLabV3Plus
    """
    def __init__(self, **model_parameters):
        super(DeepLabV3PlusModel, self).__init__()
        self.model = smp.DeepLabV3Plus(**model_parameters)  # default로 smp 라이브러리 사용

        if model_parameters.get("lora_use", False):  
            lora_config = model_parameters.get("lora_config")
            lora_config = peft.LoraConfig(**lora_config)
            self.model = peft.get_peft_model(self.model, lora_config)
    
    def forward(self, x: torch.Tensor):
        return self.model(x)

    
class DeepLabV3PlusModel_channel0(nn.Module):
    """
    Base Model DeepLabV3Plus
    """
    def __init__(self, **model_parameters):
        super(DeepLabV3PlusModel_channel0, self).__init__()
        self.additional_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.model = smp.DeepLabV3Plus(**model_parameters)  # default로 smp 라이브러리 사용

        if model_parameters.get("lora_use", False):  
            lora_config = model_parameters.get("lora_config")
            lora_config = peft.LoraConfig(**lora_config)
            self.model = peft.get_peft_model(self.model, lora_config)
                
    def forward(self, x: torch.Tensor):
        x = self.additional_conv(x)        
        return self.model(x)
        

class UnetPlusPlus(nn.Module):
    """
    Base Model UnetPlusPlus
    """
    def __init__(self, **model_parameters):
        super(UnetPlusPlus, self).__init__()
        self.model = smp.UnetPlusPlus(**model_parameters)  # default로 smp 라이브러리 사용

        if model_parameters.get("lora_use", False):  
            lora_config = model_parameters.get("lora_config")
            lora_config = peft.LoraConfig(**lora_config)
            self.model = peft.get_peft_model(self.model, lora_config)

    def forward(self, x: torch.Tensor):

        return self.model(x)
    