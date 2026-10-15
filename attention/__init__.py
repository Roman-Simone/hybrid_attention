from .mha import CausalSelfAttention
from .mla import MLA, MLACache, MLALayerCache, RMSNorm
from .swa import SlidingWindowAttention


def build_attention(config):
    if config.attention_type == 'mha':
        return CausalSelfAttention(config)
    elif config.attention_type == 'mla':
        return MLA(config)
    elif config.attention_type == 'swa':
        return SlidingWindowAttention(config)
    else:
        raise ValueError('attention_type must be mha, mla or swa')
