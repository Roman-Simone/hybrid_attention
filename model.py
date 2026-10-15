"""
The code take inspiration from the code creaated from these three repo:
- https://github.com/deepseek-ai/DeepSeek-V3.git - nanoGPT
- https://github.com/mistralai/mistral-inference.git - for SLA
- https://github.com/karpathy/nanoGPT.git - for MLA
"""
import inspect
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

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
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

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


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config) if config.attention_type == 'mha' else MLA(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attention_type: str = 'mha'
    kv_lora_rank: int = None
    qk_nope_head_dim: int = None
    qk_rope_head_dim: int = None
    v_head_dim: int = None
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.attention_type not in ('mha', 'mla'):
            raise ValueError('attention_type must be mha or mla')
        for name in ('n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.n_embd % self.n_head:
            raise ValueError('n_embd must be divisible by n_head')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if self.attention_type == 'mla':
            head_dim = self.n_embd // self.n_head
            defaults = dict(kv_lora_rank=max(1, self.n_embd // 4), qk_nope_head_dim=head_dim,
                            qk_rope_head_dim=head_dim // 2, v_head_dim=head_dim)
            for name, default in defaults.items():
                if getattr(self, name) is None:
                    setattr(self, name, default)
                if getattr(self, name) <= 0:
                    raise ValueError(f'{name} must be positive')
            if self.qk_rope_head_dim % 2:
                raise ValueError('qk_rope_head_dim must be even')
            if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
                raise ValueError('rope_theta must be finite and positive')

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            **({'wpe': nn.Embedding(config.block_size, config.n_embd)} if config.attention_type == 'mha' else {}),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and 'wpe' in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) if 'wpe' in self.transformer else 0
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def forward_cached(self, idx, cache=None):
        """Return all new-token logits and an explicit MLA cache in eval mode.

        Cache belongs to this model and batch; discard it after changing weights,
        device, dtype, or token history. Overflow requires a fresh window prefill.
        """
        if self.training or self.config.attention_type != 'mla':
            raise ValueError('forward_cached requires an MLA model in eval mode')
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError('idx must be a nonempty (batch, time) tensor')
        start_pos = 0 if cache is None else cache.length
        if start_pos < 0 or start_pos + idx.size(1) > self.config.block_size:
            raise ValueError('Cached context exceeds block_size; prefill a fresh window')
        if cache is not None:
            if len(cache.layers) != self.config.n_layer:
                raise ValueError('Cache layer count does not match model')
            for layer in cache.layers:
                if (layer.latent.shape != (idx.size(0), start_pos, self.config.kv_lora_rank)
                        or layer.rope_key.shape != (idx.size(0), start_pos, self.config.qk_rope_head_dim)
                        or layer.latent.device != idx.device or layer.rope_key.device != idx.device):
                    raise ValueError('Cache shape or device does not match input/model')
        x = self.transformer.drop(self.transformer.wte(idx))
        layers = []
        for i, block in enumerate(self.transformer.h):
            y, layer_cache = block.attn.forward_cached(
                block.ln_1(x), None if cache is None else cache.layers[i], start_pos)
            x = x + y
            x = x + block.mlp(block.ln_2(x))
            layers.append(layer_cache)
        return self.lm_head(self.transformer.ln_f(x)), MLACache(layers, start_pos + idx.size(1))

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert 0 < block_size <= self.config.block_size
        self.config.block_size = block_size
        if 'wpe' in self.transformer:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        if self.config.attention_type == 'mla':
            raise NotImplementedError('The GPT-2 MFU estimate is not valid for MLA')
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        cache = None
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            if self.config.attention_type == 'mla' and not self.training:
                if cache is None or cache.length >= self.config.block_size:
                    logits, cache = self.forward_cached(idx_cond)
                else:
                    logits, cache = self.forward_cached(idx[:, -1:], cache)
            else:
                logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

