from .base_loss import (
    CustomBCEWithLogitsLoss,
    FocalLoss,
    DiceLoss,
    IOULoss,
    CombinedLoss,
    FocalDiceLoss,
    FocalIOULoss,
)

class LossSelector:
    """
    Loss를 새롭게 추가하기 위한 방법:
    1. loss 폴더 내부에 사용하고자 하는 custom loss 구현
    2. 구현한 Loss Class를 loss_selector.py 내부로 import
    3. self.loss_classes에 아래와 같은 형식으로 추가
    4. yaml파일의 loss_name을 설정한 key값으로 변경
    """
    def __init__(self) -> None:
        self.loss_classes = {
            "BCEWithLogitsLoss": CustomBCEWithLogitsLoss,
            "FocalLoss": FocalLoss,
            "DiceLoss": DiceLoss,
            "IOULoss": IOULoss,
            "CombinedLoss": CombinedLoss,
            "FocalDiceLoss": FocalDiceLoss,
            "FocalIOULoss": FocalIOULoss,
        }

    def get_loss(self, loss_name, **loss_parameter):
        loss_class = self.loss_classes.get(loss_name, None)
        if loss_class is None:
            raise ValueError(f"Loss function '{loss_name}' is not defined.")
        return loss_class(**loss_parameter)