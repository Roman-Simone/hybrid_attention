"""
Sliding Window Attention (SWA), inspired by Mistral's causal local attention.
Keep the same projections and attention heads as the nanoGPT-style MHA.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class SlidingWindowAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.sliding_window = config.sliding_window
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads and move head forward.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Each query sees itself and at most sliding_window - 1 previous tokens.
        pos = torch.arange(T, device=x.device)
        distance = pos[:, None] - pos[None, :]
        mask = (distance >= 0) & (distance < self.sliding_window) # (T, T), True means visible

        if self.flash:
            # The local mask already includes causality.
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                               dropout_p=self.dropout if self.training else 0.0,
                                               is_causal=False)
        else:
            # Manual attention with the same causal sliding window.
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(~mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
