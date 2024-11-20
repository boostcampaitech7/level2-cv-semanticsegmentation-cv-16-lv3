from mmseg.models.decode_heads.aspp_head import ASPPHead
from mmseg.models.losses import LossByFeatMixIn
from mmseg.registry import MODELS

@MODELS.register_module()
class ASPPHeadWithoutAccuracy(LossByFeatMixIn, ASPPHead):
    pass