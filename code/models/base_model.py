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
        self.model = smp.Unet(**model_parameters)
        
    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def load_pretrained_weights(self, checkpoint_path: str):
        self.model = torch.load(checkpoint_path)
        
    def apply_lora(self, lora_config: dict):
        lora_config = peft.LoraConfig(**lora_config)
        self.model = peft.get_peft_model(self.model, lora_config)
    
    
class DeepLabV3PlusModel(nn.Module):
    """
    Base Model DeepLabV3Plus with optional LoRA
    """
    def __init__(self, **model_parameters):
        super(DeepLabV3PlusModel, self).__init__()
        self.model = smp.DeepLabV3Plus(**model_parameters)
    
    def forward(self, x: torch.Tensor):
        return self.model(x)
 
    def load_pretrained_weights(self, checkpoint_path: str):
        self.model = torch.load(checkpoint_path)
        
    def apply_lora(self, lora_config: dict):
        lora_config = peft.LoraConfig(**lora_config)
        self.model = peft.get_peft_model(self.model, lora_config)
        
    
class DeepLabV3PlusModel_channel0(nn.Module):
    """
    Base Model DeepLabV3Plus
    """
    def __init__(self, **model_parameters):
        super(DeepLabV3PlusModel_channel0, self).__init__()
        self.additional_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.model = smp.DeepLabV3Plus(**model_parameters)
                
    def forward(self, x: torch.Tensor):
        x = self.additional_conv(x)        
        return self.model(x)
        
    def load_pretrained_weights(self, checkpoint_path: str):
        self.model = torch.load(checkpoint_path)
        
    def apply_lora(self, lora_config: dict):
        lora_config = peft.LoraConfig(**lora_config)
        self.model = peft.get_peft_model(self.model, lora_config)


class UnetPlusPlus(nn.Module):
    """
    Base Model UnetPlusPlus
    """
    def __init__(self, **model_parameters):
        super(UnetPlusPlus, self).__init__()
        self.model = smp.UnetPlusPlus(**model_parameters)
        
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def load_pretrained_weights(self, checkpoint_path: str):
        self.model = torch.load(checkpoint_path)

    def apply_lora(self, lora_config: dict):
        lora_config = peft.LoraConfig(**lora_config)
        self.model = peft.get_peft_model(self.model, lora_config)
    