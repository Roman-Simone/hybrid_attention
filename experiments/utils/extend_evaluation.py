"""Benchmark cached text generation and context quality on CPU, MPS, or CUDA.

For each attention model, this runner verifies cached inference, measures
end-to-end generation (including prompt transfer and first-token latency), and
evaluates loss on shared validation targets at several context lengths. It saves
environment and source metadata plus per-round worker JSON in the run directory;
``plot_readme.py`` can aggregate those workers into the generation CSV and plot.
Existing checkpoints are loaded read-only, and this script does not create reports.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import random
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

import benchmark as bench
from evaluate import load_validation_data

KINDS = ['mha', 'mla', 'swa', 'hybrid']
VARIANTS = dict(mha='mha_sdpa', mla='mla_native', swa='swa_sdpa', hybrid='hybrid_native')
CONTEXTS = [16, 32, 64, 128, 256]
CASES = [(p, g) for p in [32, 128] for g in [16, 64, 128, 256]]


def read(path):
    return json.loads(Path(path).read_text())


def protected_manifest(run):
    files = set((bench.ROOT / 'experiments').rglob('*'))
    files.update(bench.ROOT / name for name in bench.protected_manifest())
    files.update(bench.ROOT.glob('*.md'))
    return {str(p.relative_to(bench.ROOT)): bench.sha(p) for p in sorted(files)
            if p.is_file() and not p.is_relative_to(run)
            and '__pycache__' not in p.parts and p.name != '.DS_Store'}


def memory_snapshot(device):
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    if sys.platform == 'linux':
        peak_rss *= 1024
    snapshot = dict(process_peak_rss_bytes=peak_rss)
    if device == 'mps':
        snapshot.update(tensor_bytes=torch.mps.current_allocated_memory(),
                        driver_bytes=torch.mps.driver_allocated_memory())
    elif device == 'cuda':
        snapshot.update(tensor_bytes=torch.cuda.memory_allocated(),
                        reserved_bytes=torch.cuda.memory_reserved())
    # Leave room for macOS and other applications on the 8 GiB machine.
    if device == 'mps' and snapshot['driver_bytes'] > 4 * 1024 ** 3:
        raise RuntimeError(f'MPS driver allocation exceeds the 4 GiB experiment budget: {snapshot}')
    return snapshot


def generation(model, text, stoi, itos, count, variant, device):
    """Time CPU prompt text to CPU generated text, handing off the first token."""
    bench.sync(device)
    start = time.perf_counter()
    prompt = [stoi[char] for char in text]
    history = torch.tensor([prompt], dtype=torch.long, device=device)
    cache, resets = None, 0
    for step in range(count):
        # Rebuild from the latest context once the cache has reached the model limit.
        if cache is None or cache.length >= model.config.block_size:
            resets += int(cache is not None)
            idx = history[:, -model.config.block_size:]
            logits, cache = bench.cached(model, idx, variant=variant)
        else:
            logits, cache = bench.cached(model, history[:, -1:], cache, variant=variant)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        history = torch.cat((history, next_token), dim=1)
        if step == 0:
            first_character = itos[int(next_token.cpu().item())]
            bench.sync(device)
            ttft_ms = (time.perf_counter() - start) * 1000
    output_ids = history[0, len(prompt):].cpu().tolist()
    output_text = ''.join(itos[i] for i in output_ids)
    bench.sync(device)
    total_ms = (time.perf_counter() - start) * 1000
    assert len(output_ids) == count and output_text[0] == first_character
    assert resets == max(0, len(prompt) + count - model.config.block_size - 1)
    assert cache.length == min(model.config.block_size, len(prompt) + count - 1)
    assert 0 < ttft_ms <= total_ms
    return dict(total_ms=total_ms, ttft_ms=ttft_ms, tokens_per_second=1000 * count / total_ms,
                cache_resets=resets, cached_single_steps=count - 1 - resets,
                output_ids=output_ids, output_text=output_text,
                output_sha256=hashlib.sha256(output_text.encode()).hexdigest(),
                cache=bench.memory(model, cache))


def context_inputs(data, positions, length, device):
    # Every row ends immediately BEFORE the shared target, with positions reset to zero.
    values = np.stack([data[p - length:p] for p in positions]).astype('int64')
    return torch.tensor(values, device=device)


def verify(model, data, targets, tokenizer, kind, device):
    variant = VARIANTS[kind]
    tokens = torch.tensor(data[:256].astype('int64'), device=device)[None]
    checks = bench.verify(model, tokens, variant)
    generation_checks = []
    for prompt_length, count in [(32, 40), (128, 16), (240, 20)]:
        history = tokens[:, :prompt_length].clone()
        cache, maximum_error, resets = None, 0.0, 0
        for _ in range(count):
            reference, _ = model(history[:, -model.config.block_size:])
            if cache is None or cache.length >= model.config.block_size:
                resets += int(cache is not None)
                actual, cache = bench.cached(model, history[:, -model.config.block_size:], variant=variant)
            else:
                actual, cache = bench.cached(model, history[:, -1:], cache, variant=variant)
            maximum_error = max(maximum_error, bench.close(reference[:, -1], actual[:, -1]))
            expected_token = reference[:, -1].argmax(-1, keepdim=True)
            actual_token = actual[:, -1].argmax(-1, keepdim=True)
            torch.testing.assert_close(expected_token, actual_token, rtol=0, atol=0)
            history = torch.cat((history, actual_token), dim=1)
            bench.memory(model, cache)
        prompt_text = ''.join(tokenizer['itos'][int(t)] for t in data[:prompt_length])
        result = generation(model, prompt_text, tokenizer['stoi'], tokenizer['itos'], count, variant, device)
        expected_ids = history[0, prompt_length:].cpu().tolist()
        assert result['output_ids'] == expected_ids
        assert result['cache_resets'] == resets
        # Stock generate with top_k=1 is a second deterministic greedy reference.
        original = model.generate(tokens[:, :prompt_length], count, top_k=1)
        assert original[0, prompt_length:].cpu().tolist() == expected_ids
        generation_checks.append(dict(prompt_length=prompt_length, generated_tokens=count,
                                      max_abs_error=maximum_error, cache_resets=resets,
                                      identical_greedy_tokens=True, stock_generate_matches=True))
    positions = np.array(targets['positions'])[np.linspace(0, len(targets['positions']) - 1, 8, dtype=int)]
    context_checks = []
    for length in CONTEXTS:
        inputs = context_inputs(data, positions, length, device)
        batched = model(inputs)[0][:, -1]
        single = torch.cat([model(inputs[i:i + 1])[0][:, -1] for i in range(len(inputs))])
        error = bench.close(batched, single)
        y = torch.tensor(data[positions].astype('int64'), device=device)
        loss_error = bench.close(F.cross_entropy(batched, y, reduction='none'),
                                 F.cross_entropy(single, y, reduction='none'))
        context_checks.append(dict(length=length, max_abs_logit_error=error,
                                   max_abs_loss_error=loss_error, batch=len(inputs)))
    return dict(passed=True, cached_checks=checks, generation_checks=generation_checks,
                context_batch_checks=context_checks, memory=memory_snapshot(device))


def benchmark_generation(model, prompts, tokenizer, args):
    cases = CASES.copy()
    random.Random(1337 + args.round).shuffle(cases)
    rows, outputs, snapshots = [], {}, []
    for prompt_length, count in cases:
        texts = prompts[str(prompt_length)]
        for warmup in range(args.warmups):
            generation(model, texts[warmup % len(texts)]['text'], tokenizer['stoi'],
                       tokenizer['itos'], count, VARIANTS[args.kind], args.device)
        prompt_ids = [i % len(texts) for i in range(args.repetitions)]
        random.Random(1337 + args.round + prompt_length + count).shuffle(prompt_ids)
        for repetition, prompt_id in enumerate(prompt_ids):
            result = generation(model, texts[prompt_id]['text'], tokenizer['stoi'],
                                tokenizer['itos'], count, VARIANTS[args.kind], args.device)
            key = f'{prompt_length}_{count}_{prompt_id}'
            if key in outputs:
                assert result['output_sha256'] == outputs[key]['sha256']
            outputs[key] = dict(sha256=result['output_sha256'], text=result['output_text'])
            rows.append(dict(model=args.kind, variant=VARIANTS[args.kind], round=args.round,
                             prompt_length=prompt_length, generated_tokens=count,
                             prompt_id=prompt_id, repetition=repetition,
                             **{k: v for k, v in result.items() if k not in ['output_ids', 'output_text', 'cache']},
                             final_cache_bytes=result['cache']['storage_bytes']))
            snapshots.append(memory_snapshot(args.device))
        print('GENERATION', args.kind, args.round, prompt_length, count, flush=True)
    return dict(rows=rows, outputs=outputs, memory_snapshots=snapshots, case_order=cases)


def evaluate_context(model, data, targets, args):
    positions = np.array(targets['positions'])
    rows, snapshots = [], []
    for length in CONTEXTS:
        losses = []
        for start in range(0, len(positions), args.context_batch):
            selected = positions[start:start + args.context_batch]
            x = context_inputs(data, selected, length, args.device)
            y = torch.tensor(data[selected].astype('int64'), device=args.device)
            logits, _ = model(x)
            losses.extend(F.cross_entropy(logits[:, -1], y, reduction='none').cpu().tolist())
            snapshots.append(memory_snapshot(args.device))
        assert len(losses) == len(positions) and np.isfinite(losses).all()
        rows.append(dict(model=args.kind, length=length, n=len(losses), loss=float(np.mean(losses)),
                         nll_sum=float(np.sum(losses)), samples=losses))
        print('CONTEXT', args.kind, length, len(losses), rows[-1]['loss'], flush=True)
    return dict(rows=rows, memory_snapshots=snapshots)


@torch.inference_mode()
def worker(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    data, _, _ = load_validation_data()
    with (bench.ROOT / 'data/shakespeare_char/meta.pkl').open('rb') as stream:
        tokenizer = pickle.load(stream)
    model, metadata = bench.load(args.kind, args.device)
    assert all(p.device.type == args.device and p.dtype == torch.float32 for p in model.parameters())
    initial_state = memory_snapshot(args.device)
    targets = read(args.run / 'targets.json')
    assert np.array_equal(data[targets['positions']], targets['tokens'])
    started = datetime.now(timezone.utc).isoformat()
    if args.phase == 'verify':
        result = verify(model, data, targets, tokenizer, args.kind, args.device)
    elif args.phase == 'generation':
        result = benchmark_generation(model, read(args.run / 'prompts.json'), tokenizer, args)
    else:
        result = evaluate_context(model, data, targets, args)
    bench.dump(output, dict(metadata=metadata, device=args.device, phase=args.phase,
                            start_utc=started, end_utc=datetime.now(timezone.utc).isoformat(),
                            initial_memory=initial_state, result=result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    parser.add_argument('--phase', choices=['verify', 'generation', 'context'])
    parser.add_argument('--kind', choices=KINDS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=4)
    parser.add_argument('--context-targets', type=int, default=2048)
    parser.add_argument('--context-batch', type=int, default=8)
    args = parser.parse_args()
    if min(args.rounds, args.warmups, args.repetitions, args.context_batch) < 1 or args.context_targets < 8:
        parser.error('Counts must be positive, with at least 8 context targets')
    if args.context_batch > 8:
        parser.error('Context batch is capped at 8 for the 8 GiB machine')
    if args.repetitions % 2:
        parser.error('Use an even number of repetitions to balance the two prompts')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; GPU access required, no CPU fallback')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; a CUDA-capable GPU is required')
    if args.device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Disable MPS CPU fallback for this experiment')
    torch.manual_seed(1337)
    torch.set_num_threads(4)
    args.run = args.run.resolve()
    if args.phase:
        if not args.kind or not args.output:
            parser.error('Worker phases require --kind and --output')
        return worker(args)
    args.run.mkdir(parents=True, exist_ok=False)
    (args.run / 'workers').mkdir()
    before = protected_manifest(args.run)
    bench.dump(args.run / 'preservation_before.json', before)
    data, train_characters, vocab = load_validation_data()
    positions = np.linspace(256, len(data) - 2, args.context_targets, dtype=int)
    assert len(np.unique(positions)) == args.context_targets
    bench.dump(args.run / 'targets.json', dict(positions=positions.tolist(), tokens=data[positions].tolist(),
               selection='deterministic evenly spaced positions from 256 through N-2, inclusive',
               validation_sha256=bench.sha(bench.ROOT / 'data/shakespeare_char/val.bin')))
    with (bench.ROOT / 'data/shakespeare_char/meta.pkl').open('rb') as stream:
        tokenizer = pickle.load(stream)
    prompts = {str(n): [dict(start=p, text=''.join(tokenizer['itos'][int(t)] for t in data[p:p + n]))
                        for p in [0, len(data) // 2]] for n in [32, 128]}
    bench.dump(args.run / 'prompts.json', prompts)
    hardware = platform.platform()
    if sys.platform == 'darwin':
        hardware = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string',
                                            'hw.memsize', 'hw.model'], text=True).splitlines()
    environment = dict(start_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                       python=sys.version, executable=sys.executable, torch=torch.__version__,
                       torch_git=torch.version.git_version, platform=platform.platform(), hardware=hardware,
                       device=args.device, batch=1, dtype='float32', threads=4, seed=1337,
                       rounds=args.rounds, warmups=args.warmups, repetitions=args.repetitions,
                       prompt_lengths=[32, 128], generation_lengths=[16, 64, 128, 256], prompts_per_length=2,
                       generation_policy='greedy argmax, fixed output length, no early stopping',
                       context_targets=args.context_targets, context_batch=args.context_batch,
                       context_lengths=CONTEXTS, train_characters=train_characters,
                       validation_characters=len(data), vocab_size=vocab, split_verified=True,
                       environment={k: v for k, v in os.environ.items()
                                    if k.startswith(('PYTORCH_', 'OMP_', 'MKL_'))})
    bench.dump(args.run / 'environment.json', environment)
    source_files = [Path(__file__).resolve(), Path(bench.__file__).resolve(),
                    bench.ROOT / 'experiments/utils/evaluate.py', bench.ROOT / 'model.py',
                    *sorted((bench.ROOT / 'attention').glob('*.py'))]
    bench.dump(args.run / 'source_hashes.json',
               {str(p.relative_to(bench.ROOT)): bench.sha(p) for p in source_files})
    commands = []

    def launch(phase, kind, round_id=0):
        name = f'{phase}_{kind}' + (f'_r{round_id}' if phase == 'generation' else '')
        command = [sys.executable, str(Path(__file__).resolve()), '--run', str(args.run),
                   '--phase', phase, '--kind', kind, '--round', str(round_id), '--device', args.device,
                   '--output', str(args.run / 'workers' / f'{name}.json'),
                   '--warmups', str(args.warmups), '--repetitions', str(args.repetitions),
                   '--context-batch', str(args.context_batch)]
        commands.append(command)
        bench.dump(args.run / 'commands.json', commands)
        print('START', name, flush=True)
        with (args.run / 'workers' / f'{name}.log').open('w') as stream:
            subprocess.run(command, cwd=bench.ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print('DONE', name, flush=True)

    for kind in KINDS:
        launch('verify', kind)
    for round_id in range(args.rounds):
        order = KINDS.copy() if round_id % 2 == 0 else list(reversed(KINDS))
        offset = 2 * (round_id // 2) % len(order)
        for kind in order[offset:] + order[:offset]:
            launch('generation', kind, round_id)
    for kind in KINDS:
        launch('context', kind)
    after = protected_manifest(args.run)
    bench.dump(args.run / 'preservation_after.json', after)
    assert before == after, 'Existing experimental results, README, model, weights or data changed'
    environment.update(end_utc=datetime.now(timezone.utc).isoformat(), preservation_verified=True)
    bench.dump(args.run / 'environment.json', environment)
    print('COMPLETE', args.run, flush=True)


if __name__ == '__main__':
    main()
