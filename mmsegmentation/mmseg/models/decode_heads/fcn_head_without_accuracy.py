from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.losses import LossByFeatMixIn
from mmseg.registry import MODELS

@MODELS.register_module()
class FCNHeadWithoutAccuracy(LossByFeatMixIn, FCNHead):
    pass