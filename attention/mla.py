"""Multi-head latent attention with decoupled RoPE, following DeepSeek."""
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * self.weight

@dataclass
class MLALayerCache:
    latent: torch.Tensor
    rope_key: torch.Tensor

@dataclass
class MLACache:
    layers: list[MLALayerCache]
    length: int

class MLA(nn.Module):
    """
    Multi-head Latent Attention (MLA) with joint KV compression and decoupled RoPE.

    Training: reconstructs per-head keys and values from compressed KV latents.
    Inference: uses compressed KV caching and matrix absorption to avoid
    explicitly reconstructing keys and values for previous tokens.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.rank = config.kv_lora_rank
        self.nope_dim = config.qk_nope_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.value_dim = config.v_head_dim
        self.qk_dim = self.nope_dim + self.rope_dim
        self.dropout = config.dropout

        # Project queries for all attention heads.
        self.wq = nn.Linear(self.n_embd, self.n_head * self.qk_dim, bias=False)

        # Compress KV into a shared latent and produce positional keys for RoPE.
        self.wkv_a = nn.Linear(self.n_embd, self.rank + self.rope_dim, bias=False)

        # Normalize compressed KV representations for training stability.
        self.kv_norm = RMSNorm(self.rank)

        # Reconstruct content keys and values for all attention heads.
        self.wkv_b = nn.Linear(self.rank, self.n_head * (self.nope_dim + self.value_dim), bias=False)

        # Combine attention head outputs and project back to embedding size.
        self.c_proj = nn.Linear(self.n_head * self.value_dim, self.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Precompute inverse frequencies for Rotary Positional Embeddings.
        self.register_buffer("inv_freq", 1.0 / (config.rope_theta ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim)), persistent=False)

    def _rope(self, x, start_pos):
        """Apply rotary positional embeddings to pairs of adjacent dimensions."""

        positions = torch.arange(start_pos, start_pos + x.size(1), device=x.device).float()
        angles = positions[:, None] * self.inv_freq.float()[None, :]
        cos = angles.cos()[None, :, None, :].to(x.dtype)
        sin = angles.sin()[None, :, None, :].to(x.dtype)

        # Rotate pairs of even and odd dimensions.
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def _project(self, x, start_pos):
        """
        Project queries and compress KV representations.
        Separate content and positional components for decoupled RoPE.
        """

        B, T, C = x.size()

        # Project queries for all heads and separate content and positional parts.
        q = self.wq(x).view(B, T, self.n_head, self.qk_dim)
        q_content, q_rope = q.split((self.nope_dim, self.rope_dim), dim=-1)

        # Compress KV into one shared latent per token and extract positional keys.
        kv_latent, k_rope = self.wkv_a(x).split((self.rank, self.rope_dim), dim=-1)

        # Apply RoPE only to positional components.
        q_rope = self._rope(q_rope, start_pos)
        k_rope = self._rope(k_rope.unsqueeze(2), start_pos).squeeze(2)
        kv_latent = self.kv_norm(kv_latent)

        return q_content, q_rope, kv_latent, k_rope

    def forward(self, x):
        """
        Training forward pass: reconstruct keys and values from KV latents,
        then compute causal scaled dot-product attention.
        """

        B, T, C = x.size()
        q_content, q_rope, kv_latent, k_rope = self._project(x, 0)

        # Reconstruct content keys and values for all heads.
        kv = self.wkv_b(kv_latent).view(B, T, self.n_head, self.nope_dim + self.value_dim)
        k_content, v = kv.split((self.nope_dim, self.value_dim), dim=-1)

        # Combine content and positional components of queries and keys.
        q = torch.cat((q_content, q_rope), dim=-1).transpose(1, 2)

        # Positional keys are shared across all attention heads.
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, self.n_head, -1)
        k = torch.cat((k_content, k_rope), dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)

        # Causal self-attention:
        # q, k: (B, n_head, T, qk_dim)
        # v:    (B, n_head, T, value_dim)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)

        # Reassemble attention heads and project to embedding dimension.
        y = y.transpose(1, 2).contiguous()
        return self.resid_dropout(self.c_proj(y.flatten(2)))

    def forward_cached(self, x, cache, start_pos):
        """
        MLA inference with compressed KV caching and matrix absorption.

        Compute attention directly on compressed latents without
        reconstructing full keys and values for previous tokens.
        """

        B, T, C = x.size()
        q_content, q_rope, kv_latent, k_rope = self._project(x, start_pos)

        # Append new compressed KV representations to the cache.
        if cache is not None:
            kv_latent = torch.cat((cache.latent, kv_latent), dim=1)
            k_rope = torch.cat((cache.rope_key, k_rope), dim=1)

        # Separate key and value projection weights for each head.
        kv_weights = self.wkv_b.weight.view(self.n_head, self.nope_dim + self.value_dim, self.rank)
        k_weights = kv_weights[:, :self.nope_dim]
        v_weights = kv_weights[:, self.nope_dim:]

        # Matrix absorption: project queries into latent space instead of expanding keys.
        q_latent = torch.einsum("bthd,hdr->bthr", q_content, k_weights)

        # Compute content and positional attention scores.
        content_scores = torch.einsum("bthr,bsr->bhts", q_latent, kv_latent)
        position_scores = torch.einsum("bthd,bsd->bhts", q_rope, k_rope)
        scores = (content_scores + position_scores) * (self.qk_dim ** -0.5)

        # Causal mask: prevent attention to future tokens.
        query_pos = start_pos + torch.arange(T, device=x.device)
        key_pos = torch.arange(kv_latent.size(1), device=x.device)
        scores = scores.masked_fill(key_pos[None, :] > query_pos[:, None], float("-inf"))

        # Convert attention scores into probabilities.
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)

        # Aggregate compressed KV latents using attention probabilities.
        aggregated = torch.einsum("bhts,bsr->bthr", probs, kv_latent)

        # Apply value projection only after latent aggregation.
        y = torch.einsum("bthr,hdr->bthd", aggregated, v_weights)

        # Combine attention heads and return the updated compressed KV cache.
        return self.resid_dropout(self.c_proj(y.flatten(2))), MLALayerCache(kv_latent, k_rope)

