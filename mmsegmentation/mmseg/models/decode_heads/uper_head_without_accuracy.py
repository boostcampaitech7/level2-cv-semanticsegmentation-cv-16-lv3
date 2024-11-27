from mmseg.models.decode_heads import UPerHead
from mmseg.models.losses import LossByFeatMixIn
from mmseg.registry import MODELS

@MODELS.register_module()
class UPerHeadWithoutAccuracy(LossByFeatMixIn, UPerHead):
    pass

