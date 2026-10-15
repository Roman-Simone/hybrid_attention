"""MLA with a causal sliding window and a compressed local KV cache."""
from .mla import MLA, MLALayerCache


class HybridAttention(MLA):

    def __init__(self, config):
        super().__init__(config)
        self.sliding_window = config.sliding_window

    def _attention_mask(self, query_pos, key_pos):
        # Each query sees itself and at most sliding_window - 1 previous tokens.
        distance = query_pos[:, None] - key_pos[None, :]
        return (distance >= 0) & (distance < self.sliding_window)

    def forward_cached(self, x, cache, start_pos):
        y, cache = super().forward_cached(x, cache, start_pos)
        # Clone slices so the cache releases storage for tokens outside the window.
        cache = MLALayerCache(cache.latent[:, -self.sliding_window:].clone(),
                              cache.rope_key[:, -self.sliding_window:].clone())
        return y, cache
