from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.models.losses import LossByFeatMixIn
from mmseg.registry import MODELS

@MODELS.register_module()
class SegformerHeadWithoutAccuracy(LossByFeatMixIn, SegformerHead):
    pass