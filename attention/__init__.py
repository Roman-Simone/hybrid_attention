from .mha import CausalSelfAttention
from .mla import MLA, MLACache, MLALayerCache, RMSNorm
from .swa import SlidingWindowAttention
from .hybrid import HybridAttention


def build_attention(config):
    if config.attention_type == 'mha':
        return CausalSelfAttention(config)
    elif config.attention_type == 'mla':
        return MLA(config)
    elif config.attention_type == 'swa':
        return SlidingWindowAttention(config)
    elif config.attention_type == 'hybrid':
        return HybridAttention(config)
    else:
        raise ValueError('attention_type must be mha, mla, swa or hybrid')
