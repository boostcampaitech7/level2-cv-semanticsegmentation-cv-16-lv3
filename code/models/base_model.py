import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchseg
import peft

class UnetModel(nn.Module):
    """
    Base Model Unet
    """
    def __init__(self, **model_parameters):
        super(UnetModel, self).__init__()
        
        if model_parameters.get("torchseg_use", False):  # torchseg_use가 True일 경우에만 torchseg 라이브러리 사용
            encoder_depth = len(torchseg.encoders.TIMM_ENCODERS[model_parameters["encoder_name"]]['channels'])  # encoder의 depth
            if model_parameters.get("transformer_use", False):  # encoder가 transformer일 경우에만 추가되는 파라미터
                model_parameters["encoder_params"] = {"img_size": model_parameters["img_size"]}

            if encoder_depth == 4:  # encoder의 depth가 4일 경우 파라미터를 수정해야함, 5일 경우에는 수정할 필요 없음
                model_parameters["encoder_depth"] = 4
                model_parameters["decoder_channels"] = [256, 128, 64, 32]
                model_parameters["head_upsampling"] = 2
            
            elif encoder_depth == 3 or encoder_depth == 6:  # encoder의 depth가 3이거나 6인 경우는 에러 나오게 설정했음
                raise(ValueError(f" encoder의 depth가 {encoder_depth}인 경우는 지원하지 않습니다."))
            
            # config의 model parameter가 전부 들어가 불필요한 인자는 제거하는 작업
            model_parameters = {key: value for key, value in model_parameters.items() 
                                   if key not in ["lora_use", "lora_config", "torchseg_use", "transformer_use", "img_size"]}
            self.model = torchseg.Unet(**model_parameters)
        
        else: 
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

        if model_parameters.get("torchseg_use", False):  # torchseg_use가 True일 경우에만 torchseg 라이브러리 사용
            encoder_depth = len(torchseg.encoders.TIMM_ENCODERS[model_parameters["encoder_name"]]['channels'])  # encoder의 depth
            if model_parameters.get("transformer_use", False):  # encoder가 transformer일 경우에만 추가되는 파라미터
                model_parameters["encoder_params"] = {"img_size": model_parameters["img_size"]}

            if encoder_depth == 4:  # encoder의 depth가 4일 경우 파라미터를 수정해야함
                model_parameters["encoder_depth"] = 4
            
            elif encoder_depth == 3 or encoder_depth == 6:  # encoder의 depth가 3이거나 6인 경우는 에러 나오게 설정했음
                raise(ValueError(f" encoder의 depth가 {encoder_depth}인 경우는 지원하지 않습니다."))
            
             # config의 model parameter가 전부 들어가 불필요한 인자는 제거하는 작업
            model_parameters = {key: value for key, value in model_parameters.items() 
                                   if key not in ["lora_use", "lora_config", "torchseg_use", "transformer_use", "img_size"]}
            self.model = torchseg.DeepLabV3Plus(**model_parameters)
        
        else:
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

        if model_parameters.get("torchseg_use", False):  # torchseg_use가 True일 경우에만 torchseg 라이브러리 사용
            encoder_depth = len(torchseg.encoders.TIMM_ENCODERS[model_parameters["encoder_name"]]['channels'])  # encoder의 depth
            if model_parameters.get("transformer_use", False):  # encoder가 transformer일 경우에만 추가되는 파라미터
                model_parameters["encoder_params"] = {"img_size": model_parameters["img_size"]}

            if encoder_depth == 4:  # encoder의 depth가 4일 경우 파라미터를 수정해야함
                model_parameters["encoder_depth"] = 4
            
            elif encoder_depth == 3 or encoder_depth == 6:  # encoder의 depth가 3이거나 6인 경우는 에러 나오게 설정했음
                raise(ValueError(f" encoder의 depth가 {encoder_depth}인 경우는 지원하지 않습니다."))
            
             # config의 model parameter가 전부 들어가 불필요한 인자는 제거하는 작업
            model_parameters = {key: value for key, value in model_parameters.items() 
                                   if key not in ["lora_use", "lora_config", "torchseg_use", "transformer_use", "img_size"]}
            self.model = torchseg.DeepLabV3Plus(**model_parameters)

        else:
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

        if model_parameters.get("torchseg_use", False):  # torchseg_use가 True일 경우에만 torchseg 라이브러리 사용
            encoder_depth = len(torchseg.encoders.TIMM_ENCODERS[model_parameters["encoder_name"]]['channels'])  # encoder의 depth
            if model_parameters.get("transformer_use", False):  # encoder가 transformer일 경우에만 추가되는 파라미터
                model_parameters["encoder_params"] = {"img_size": model_parameters["img_size"]}

            if encoder_depth == 4:  # encoder의 depth가 4일 경우 파라미터를 수정해야함
                model_parameters["encoder_depth"] = 4
                model_parameters["decoder_channels"] = [256, 128, 64, 32]
                model_parameters["head_upsampling"] = 2
            
            elif encoder_depth == 3 or encoder_depth == 6:  # encoder의 depth가 3이거나 6인 경우는 에러 나오게 설정했음
                raise(ValueError(f" encoder의 depth가 {encoder_depth}인 경우는 지원하지 않습니다."))
            
             # config의 model parameter가 전부 들어가 불필요한 인자는 제거하는 작업
            model_parameters = {key: value for key, value in model_parameters.items() 
                                   if key not in ["lora_use", "lora_config", "torchseg_use", "transformer_use", "img_size"]}
            self.model = torchseg.UnetPlusPlus(**model_parameters)

        else:
            self.model = smp.UnetPlusPlus(**model_parameters)  # default로 smp 라이브러리 사용

        if model_parameters.get("lora_use", False):  
            lora_config = model_parameters.get("lora_config")
            lora_config = peft.LoraConfig(**lora_config)
            self.model = peft.get_peft_model(self.model, lora_config)

    def forward(self, x: torch.Tensor):

        return self.model(x)
    