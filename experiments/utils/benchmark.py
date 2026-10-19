"""Verify and benchmark cached inference for the four attention implementations.

The harness loads existing checkpoints, compares full-sequence and cached outputs,
checks cache size and boundary behavior, then records prefill and token-generation
latencies with warm-ups and repeated samples. It supports CPU, MPS, and CUDA; each run
writes machine-readable JSON with timing samples, model/checkpoint metadata, and
verification results. No training or checkpoint modification is performed.
"""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import GPT, GPTConfig

VARIANTS = {
    'mha_sdpa': ('mha', 'sdpa'), 'mla_native': ('mla', 'native'),
    'swa_sdpa': ('swa', 'sdpa'), 'hybrid_native': ('hybrid', 'native'),
    'mha_manual': ('mha', 'manual'), 'mla_explicit': ('mla', 'explicit'),
    'swa_manual': ('swa', 'manual'), 'hybrid_explicit': ('hybrid', 'explicit'),
}
LENGTHS = [16, 32, 48, 63, 64, 65, 96, 128, 192, 240]
ORIGINAL_EINSUM = torch.einsum


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def stats(samples):
    return dict(median=float(np.median(samples)), q25=float(np.quantile(samples, .25)),
                q75=float(np.quantile(samples, .75)), samples=samples)


def close(a, b):
    torch.testing.assert_close(a, b, atol=3e-4, rtol=3e-4)
    return float((a-b).abs().max())


def explicit_softmax(input, dim=None, _stacklevel=3, dtype=None):
    assert dim is not None
    x = input.to(dtype) if dtype is not None else input
    exp = (x-x.amax(dim=dim, keepdim=True)).exp()
    return exp/exp.sum(dim=dim, keepdim=True)


def scores_mm(equation, *operands, **kwargs):
    # Diagnostic only: equivalent score products for the benchmark's batch of one.
    if equation in ('bthr,bsr->bhts', 'bthd,bsd->bhts'):
        q, kv = operands
        b, t, h, d = q.shape
        if b != 1 or kv.size(0) != 1:
            raise ValueError('The explicit diagnostic path requires batch 1')
        return (q[0].transpose(0,1).reshape(h*t,d) @ kv[0].T).reshape(1,h,t,-1)
    if equation == 'bhts,bsr->bthr':
        probs, kv = operands
        b, h, t, s = probs.shape
        if b != 1 or kv.size(0) != 1:
            raise ValueError('The explicit diagnostic path requires batch 1')
        return (probs[0].reshape(h*t,s) @ kv[0]).reshape(h,t,-1).transpose(0,1).unsqueeze(0)
    return ORIGINAL_EINSUM(equation, *operands, **kwargs)


@contextmanager
def execution_path(variant):
    with ExitStack() as stack:
        if VARIANTS[variant][1] == 'explicit':
            stack.enter_context(patch.object(torch, 'einsum', scores_mm))
            stack.enter_context(patch.object(F, 'softmax', explicit_softmax))
        yield


@dataclass
class KVCache:
    layers: list
    length: int


def cached(model, idx, cache=None, *, variant):
    """Run one cached chunk, retaining only the keys needed by the attention mode."""
    kind, backend = VARIANTS[variant]
    assert not model.training and kind == model.config.attention_type
    if kind in ('mla', 'hybrid'):
        if backend == 'explicit' and idx.size(0) != 1:
            raise ValueError('The explicit diagnostic path requires batch 1')
        return model.forward_cached(idx, cache)
    start = 0 if cache is None else cache.length
    if idx.ndim != 2 or idx.size(1) == 0 or start+idx.size(1) > model.config.block_size:
        raise ValueError('Invalid cached input or context overflow')
    x = model.transformer.drop(model.transformer.wte(idx) + model.transformer.wpe(
        torch.arange(start, start+idx.size(1), device=idx.device)))
    layers = []
    for i, block in enumerate(model.transformer.h):
        a = block.attn
        b, t, c = x.shape
        q, k, v = [z.reshape(b,t,a.n_head,c//a.n_head).transpose(1,2)
                   for z in a.c_attn(block.ln_1(x)).split(c, dim=-1)]
        if cache is not None:
            k = torch.cat((cache.layers[i][0], k), dim=2)
            v = torch.cat((cache.layers[i][1], v), dim=2)
        qp = torch.arange(start,start+t,device=x.device)
        kp = torch.arange(start+t-k.shape[2],start+t,device=x.device)
        distance = qp[:,None]-kp[None,:]
        mask = distance >= 0
        window = getattr(a, 'sliding_window', None)
        if window is not None:
            mask &= distance < window
        if backend == 'sdpa':
            y = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
        else:
            scores = (q @ k.transpose(-2,-1)) * (k.size(-1)**-.5)
            probs = F.softmax(scores.masked_fill(~mask, float('-inf')), dim=-1)
            y = probs @ v
        x = x+a.resid_dropout(a.c_proj(y.transpose(1,2).contiguous().reshape(b,t,c)))
        x = x+block.mlp(block.ln_2(x))
        keep = k.shape[2] if window is None else window
        layers.append((k[:,:,-keep:].clone(), v[:,:,-keep:].clone()))
    return model.lm_head(model.transformer.ln_f(x)), KVCache(layers,start+idx.size(1))


def memory(model, cache):
    tensors = []
    for layer in cache.layers:
        tensors.extend(layer if isinstance(layer,tuple) else (layer.latent,layer.rope_key))
    storages = {(str(t.device),t.untyped_storage().data_ptr()):t.untyped_storage().nbytes() for t in tensors}
    logical, allocated = sum(t.numel()*t.element_size() for t in tensors), sum(storages.values())
    config = model.config
    local = config.attention_type in ('swa','hybrid')
    used = min(cache.length,config.sliding_window) if local else cache.length
    width = 2*config.n_embd if config.attention_type in ('mha','swa') else config.kv_lora_rank+config.qk_rope_head_dim
    expected = tensors[0].size(0)*used*width*config.n_layer*tensors[0].element_size()
    assert logical == allocated == expected, (logical,allocated,expected)
    return dict(length=cache.length,logical_bytes=logical,storage_bytes=allocated,expected_bytes=expected)


def load(kind, device):
    path = ROOT/'out'/kind/'best.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = GPT(GPTConfig(**ck['model_args'])).eval()
    model.load_state_dict(ck['model'])
    model.to(device)
    meta = dict(config=asdict(model.config), parameters=sum(p.numel() for p in model.parameters()),
                checkpoint=str(path.relative_to(ROOT)), sha256=sha(path), iteration=ck['iter_num'],
                best_val_loss=ck['best_val_loss'], training_config=ck['config'])
    del ck
    gc.collect()
    return model, meta


def verify(model, tokens, variant):
    """Check cached results against full forward across chunks, resets, and limits."""
    fn = lambda idx, cache=None: cached(model,idx,cache,variant=variant)
    full = model(tokens,tokens)[0]
    # Reference native prefill before enabling the explicit diagnostic variant.
    baseline_variant = model.config.attention_type+('_sdpa' if model.config.attention_type in ('mha','swa') else '_native')
    baseline = cached(model,tokens,variant=baseline_variant)[0]
    errors, records = {}, []
    with execution_path(variant):
        y, cache = fn(tokens)
        errors['prefill_full'] = close(full,y)
        errors['prefill_native'] = close(baseline,y)
        parts, cache = [], None
        for j in range(256):
            y, cache = fn(tokens[:,j:j+1],cache)
            parts.append(y)
            if j+1 in LENGTHS+[256]:
                records.append(dict(mode='incremental',**memory(model,cache)))
        errors['incremental'] = close(full,torch.cat(parts,1))
        for cuts in ([0,63,64,65,127,128,129,256],[0,80,160,256]):
            parts, cache = [], None
            for a,b in zip(cuts,cuts[1:]):
                y, cache = fn(tokens[:,a:b],cache)
                parts.append(y)
            errors['chunks_'+str(cuts)] = close(full,torch.cat(parts,1))
        for n in LENGTHS+[256]:
            y, cache = fn(tokens[:,:n])
            errors['prefix_'+str(n)] = close(full[:,:n],y)
            records.append(dict(mode='prefill',**memory(model,cache)))
        flipped = tokens.flip(1)
        errors['reset_other'] = close(model(flipped,flipped)[0],fn(flipped)[0])
        errors['reset_original'] = close(full,fn(tokens)[0])
        try:
            fn(tokens[:,:1],cache)
        except ValueError:
            pass
        else:
            raise AssertionError('Overflow was accepted')
        two = tokens[:,:80].expand(2,-1).contiguous()
        if VARIANTS[variant][1] == 'explicit':
            try:
                fn(two)
            except ValueError:
                batch2 = 'explicitly rejected: diagnostic supports batch 1 only'
            else:
                raise AssertionError('Explicit path unexpectedly accepted batch 2')
        else:
            errors['batch2'] = close(model(two,two)[0],fn(two)[0])
            batch2 = 'passed'
        if VARIANTS[variant][1] == 'manual':
            with patch.object(F,'scaled_dot_product_attention',side_effect=AssertionError('Unexpected SDPA')):
                errors['without_sdpa'] = close(full[:,:64],fn(tokens[:,:64])[0])
    return dict(passed=True,atol=3e-4,rtol=3e-4,max_abs_errors=errors,memory=records,
                overflow_rejected=True,batch2=batch2)


def measure(fn, args, divisor=1):
    """Return per-call timings after warm-up, synchronizing asynchronous MPS work."""
    for _ in range(args.warmups):
        fn()
    sync(args.device)
    samples = []
    for _ in range(args.repetitions):
        sync(args.device)
        start = time.perf_counter()
        result = fn()
        sync(args.device)
        samples.append((time.perf_counter()-start)*1000/divisor)
    return stats(samples)


def benchmark(model, tokens, args):
    fn = lambda idx, cache=None: cached(model,idx,cache,variant=args.variant)
    rows = []
    metrics = ['prefill_ms','fixed_decode_ms','decode16_ms']
    metrics = metrics[args.round%3:]+metrics[:args.round%3]
    with execution_path(args.variant):
        for metric in metrics:
            lengths = args.lengths.copy()
            random.Random(1337+args.round).shuffle(lengths)
            for n in lengths:
                if metric == 'prefill_ms':
                    timing = measure(lambda: fn(tokens[:,:n]),args)
                elif metric == 'fixed_decode_ms':
                    _, initial = fn(tokens[:,:n])
                    timing = measure(lambda: fn(tokens[:,n:n+1],initial),args)
                else:
                    def growing_trial():
                        # Rebuild cache outside timer. This measures decoding AFTER a real prefill.
                        _, cache = fn(tokens[:,:n])
                        sync(args.device)
                        start = time.perf_counter()
                        for j in range(n,n+16):
                            _, cache = fn(tokens[:,j:j+1],cache)
                        sync(args.device)
                        return (time.perf_counter()-start)*1000/16
                    for _ in range(args.warmups):
                        growing_trial()
                    timing = stats([growing_trial() for _ in range(args.repetitions)])
                rows.append(dict(variant=args.variant,round=args.round,metric=metric,length=n,**timing))
                print(args.variant,args.round,metric,n,round(timing['median'],4),flush=True)
    return dict(metric_order=metrics,rows=rows)


def protected_manifest():
    files = [ROOT/'model.py', *sorted((ROOT/'attention').glob('*.py')),
             *sorted((ROOT/'out').glob('*/*.pt')),
             *sorted((ROOT/'data/shakespeare_char').glob('*'))]
    return {str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()}


@torch.inference_mode()
def worker(args):
    torch.manual_seed(1337)
    torch.set_num_threads(4)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; GPU access is required, no automatic CPU fallback')
    data = np.fromfile(ROOT/'data/shakespeare_char/val.bin',dtype=np.uint16)
    tokens = torch.tensor(data[:256].astype('int64'),device=args.device)[None]
    model, metadata = load(VARIANTS[args.variant][0],args.device)
    result = verify(model,tokens,args.variant) if args.phase == 'verify' else benchmark(model,tokens,args)
    dump(args.output,dict(metadata=metadata,device=args.device,variant=args.variant,result=result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',choices=['mps','cuda','cpu'],default='mps')
    p.add_argument('--phase',choices=['verify','bench'])
    p.add_argument('--variant',choices=list(VARIANTS))
    p.add_argument('--round',type=int,default=0)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--warmups',type=int,default=5)
    p.add_argument('--repetitions',type=int,default=10)
    p.add_argument('--lengths',type=int,nargs='+',default=LENGTHS)
    args = p.parse_args()
    if any(n<1 or n>240 for n in args.lengths):
        p.error('Lengths must fit a 16-token decode inside block_size 256')
    if min(args.rounds,args.warmups,args.repetitions) <= 0:
        p.error('Round, warmup and repetition counts must be positive')
    if args.phase:
        if not args.variant:
            p.error('--variant required for a worker phase')
        if args.output.exists():
            raise FileExistsError(args.output)
        return worker(args)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; run with GPU access')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run with a CUDA-capable GPU')
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'workers').mkdir()
    before = protected_manifest()
    dump(out/'preservation_before.json',before)
    env = dict(start_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,python=sys.version,
               executable=sys.executable,torch=torch.__version__,torch_git=torch.version.git_version,
               platform=platform.platform(),device=args.device,batch=1,dtype='float32',threads=4,
               seed=1337,rounds=args.rounds,warmups=args.warmups,repetitions=args.repetitions,
               lengths=args.lengths,decode_steps=16,variants=VARIANTS,
               environment={k:v for k,v in os.environ.items() if k.startswith(('PYTORCH_','OMP_','MKL_'))})
    env['hardware'] = platform.platform()
    if sys.platform == 'darwin':
        try:
            env['hardware'] = subprocess.check_output(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string','hw.memsize','hw.model'],text=True).splitlines()
        except (OSError, subprocess.CalledProcessError):
            pass
    dump(out/'environment.json',env)
    dump(out/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'model.py',*sorted((ROOT/'attention').glob('*.py'))]})
    commands = []
    def launch(phase,variant,round_id=0,device=None):
        device = device or args.device
        name = f'{phase}_{device}_{variant}'+(f'_r{round_id}' if phase=='bench' else '')
        command = [sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--variant',variant,
                   '--device',device,'--round',str(round_id),'--warmups',str(args.warmups),
                   '--repetitions',str(args.repetitions),'--output',str(out/'workers'/(name+'.json')),
                   '--lengths',*map(str,args.lengths)]
        commands.append(command)
        dump(out/'commands.json',commands)
        print('START',name,flush=True)
        with (out/'workers'/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
        print('DONE',name,flush=True)
    # Correctness is separated from performance workers, including graph warmup state.
    for variant in VARIANTS:
        launch('verify',variant)
    # A CPU campaign has already checked these paths in the loop above.
    if args.device != 'cpu':
        for variant in ('mla_explicit','hybrid_explicit'):
            launch('verify',variant,device='cpu')
    variants = list(VARIANTS)
    for round_id in range(args.rounds):
        # Reversal in odd rounds, rotation by four in later rounds balances position across 4 rounds.
        order = variants.copy() if round_id%2 == 0 else list(reversed(variants))
        offset = 4*(round_id//2)%len(order)
        order = order[offset:]+order[:offset]
        for variant in order:
            launch('bench',variant,round_id)
    after = protected_manifest()
    dump(out/'preservation_after.json',after)
    assert before == after,'Protected source, checkpoint, data or archive changed'
    env['end_utc'] = datetime.now(timezone.utc).isoformat()
    dump(out/'environment.json',env)
    print('COMPLETE',out,flush=True)


if __name__ == '__main__':
    main()
