from .base_model import UnetModel, DeepLabV3PlusModel

class ModelSelector():
    """
    model을 새롭게 추가하기 위한 방법
        1. model 폴더 내부에 사용하고자하는 custom model 구현
        2. 구현한 Model Class를 model_selector.py 내부로 import
        3. self.model_classes에 아래와 같은 형식으로 추가
        4. yaml파일의 model_name을 설정한 key값으로 변경
    """
    def __init__(self) -> None:
        self.model_classes = {
            "Unet" : UnetModel,
            "DeepLabV3Plus": DeepLabV3PlusModel,
            "UnetPlusPlus": UnetPlusPlus
        }

    
    def get_model(self, model_name, **model_parameter):
        if model_name not in self.model_classes:
            raise ValueError(f"모델 '{model_name}'은 등록되지 않았습니다.")
        
        #get 매서드는 딕셔너리 매서드로, get를 통해 키가 존재하면 해당 키값 반환, 존재하지 않으면 None
        return self.model_classes.get(model_name, None)(**model_parameter)