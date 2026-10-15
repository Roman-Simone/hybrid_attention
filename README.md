# hybrid_attention

One GPT with selectable MHA, MLA or Sliding Window Attention (SWA).

Attention implementations live in `attention/mha.py`, `attention/mla.py` and
`attention/swa.py`. `model.py` contains GPT, its configuration and transformer
blocks. RoPE stays inside MLA.

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

