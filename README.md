# hybrid_attention

One GPT with selectable MHA, MLA, Sliding Window Attention (SWA) or hybrid attention.

Attention implementations live in `attention/mha.py`, `attention/mla.py` and
`attention/swa.py`. `model.py` contains GPT, its configuration and transformer
blocks. RoPE stays inside MLA.

Hybrid attention (`attention/hybrid.py`) combines MLA's compressed KV projections
and decoupled RoPE with SWA's causal local window. It reuses MLA and keeps only
the last `sliding_window` compressed tokens in the inference cache. Like MLA,
it uses no learned position embeddings. Training uses a dense local SDPA mask.

```sh
python train.py --dataset shakespeare_char --attention_type hybrid --sliding_window 64 --out_dir out/hybrid
```

SWA keeps the same projections, attention heads and learned position embeddings
as MHA. Each token attends only to itself and the previous `sliding_window - 1`
tokens, following the causal local window used by Mistral. The default window
is 256 tokens. A window covering the sequence is equivalent to full causal MHA
when using the same weights.

```sh
python train.py --dataset shakespeare_char --attention_type swa --sliding_window 64 --out_dir out/swa
python train.py --dataset shakespeare_char --init_from resume --out_dir out/swa
```

The window is saved in checkpoints and restored on resume. Generation keeps the
existing full-context forward and crops at `block_size`. SWA uses a dense local
mask with PyTorch SDPA (or manual attention); it does not introduce a KV cache
or an optimized sparse attention kernel.

Run attention tests with:

```sh
python -m unittest discover -s tests -v
```

