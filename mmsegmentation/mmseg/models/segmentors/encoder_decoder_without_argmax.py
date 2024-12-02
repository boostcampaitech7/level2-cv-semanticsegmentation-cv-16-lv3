from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.registry import MODELS

from mmseg.utils import PostProcessResultMixin

    
@MODELS.register_module()
class EncoderDecoderWithoutArgmax(PostProcessResultMixin, EncoderDecoder):
    pass