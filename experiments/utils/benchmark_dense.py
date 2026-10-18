"""Measure inference latency on a uniform prompt-length grid.

Runs correctness checks first, then benchmarks the reference execution paths for
MHA, MLA, SWA, and Hybrid across prompt lengths 16–240 in four-token steps.
Select CPU, MPS, or CUDA with ``--device``. Workers save raw JSON samples; the parent process pools repetitions into
``benchmark.csv`` (medians and quartiles) and ``samples.csv`` (individual times)
for plotting and later inspection. Checkpoints are read-only.
"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

import benchmark as bench


VARIANTS = ['mha_sdpa', 'mla_native', 'swa_sdpa', 'hybrid_native']
LENGTHS = list(range(16, 241, 4))


@torch.inference_mode()
def worker(args):
    torch.manual_seed(1337)
    torch.set_num_threads(4)
    data = np.fromfile(bench.ROOT / 'data/shakespeare_char/val.bin', dtype=np.uint16)
    tokens = torch.tensor(data[:256].astype('int64'), device=args.device)[None]
    kind = bench.VARIANTS[args.variant][0]
    model, metadata = bench.load(kind, args.device)
    metadata['checkpoint_config'] = metadata['config'].copy()
    if kind in ('swa', 'hybrid'):
        model.config.sliding_window = args.window
        for block in model.transformer.h:
            block.attn.sliding_window = args.window
    metadata['config'] = asdict(model.config)
    if args.phase == 'verify':
        # Also exercise the exact window boundary, outside the plotted grid.
        bench.LENGTHS = sorted(set(LENGTHS + [args.window - 1, args.window + 1]))
        result = bench.verify(model, tokens, args.variant)
        cuts = [0, args.window - 1, args.window, args.window + 1, 128, 256]
        full = model(tokens, tokens)[0]
        parts, cache = [], None
        for start, stop in zip(cuts, cuts[1:]):
            output, cache = bench.cached(model, tokens[:, start:stop], cache, variant=args.variant)
            parts.append(output)
        result['max_abs_errors']['configured_window_chunks'] = bench.close(full, torch.cat(parts, 1))
    else:
        result = bench.benchmark(model, tokens, args)
    bench.dump(args.output, dict(metadata=metadata, device=args.device, variant=args.variant, result=result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], required=True)
    parser.add_argument('--window', type=int, choices=[60, 64], default=64)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--phase', choices=['verify', 'bench'])
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--round', type=int, default=0)
    args = parser.parse_args()
    args.lengths = LENGTHS
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; GPU access is required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; a CUDA-capable GPU is required')
    if min(args.rounds, args.warmups, args.repetitions) <= 0:
        parser.error('Round, warmup and repetition counts must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.phase:
        if not args.variant:
            parser.error('--variant is required with --phase')
        return worker(args)

    run = args.output.resolve()
    (run / 'workers').mkdir(parents=True)
    before = bench.protected_manifest()
    bench.dump(run / 'preservation_before.json', before)
    environment = dict(start_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                       executable=sys.executable, python=sys.version, torch=torch.__version__,
                       platform=platform.platform(), device=args.device, threads=4, batch=1,
                       dtype='float32', seed=1337, window=args.window, lengths=LENGTHS,
                       decode_steps=16, rounds=args.rounds, warmups=args.warmups,
                       repetitions=args.repetitions,
                       variants={v: bench.VARIANTS[v] for v in VARIANTS})
    bench.dump(run / 'environment.json', environment)
    sources = [Path(__file__), Path(bench.__file__), bench.ROOT / 'model.py',
               *sorted((bench.ROOT / 'attention').glob('*.py'))]
    bench.dump(run / 'source_hashes.json',
               {str(p.resolve().relative_to(bench.ROOT)): bench.sha(p) for p in sources})
    commands = []

    def launch(phase, variant, round_id=0):
        name = f'{phase}_{args.device}_{variant}' + (f'_r{round_id}' if phase == 'bench' else '')
        command = [sys.executable, str(Path(__file__).resolve()), '--output',
                   str(run / 'workers' / f'{name}.json'), '--phase', phase,
                   '--variant', variant, '--round', str(round_id), '--window', str(args.window),
                   '--device', args.device,
                   '--warmups', str(args.warmups), '--repetitions', str(args.repetitions)]
        commands.append(command)
        bench.dump(run / 'commands.json', commands)
        print('START', name, flush=True)
        with (run / 'workers' / f'{name}.log').open('w') as log:
            subprocess.run(command, cwd=bench.ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
        print('DONE', name, flush=True)

    for variant in VARIANTS:
        launch('verify', variant)
    for round_id in range(args.rounds):
        order = VARIANTS.copy() if round_id % 2 == 0 else list(reversed(VARIANTS))
        offset = 2 * (round_id // 2) % len(order)
        for variant in order[offset:] + order[:offset]:
            launch('bench', variant, round_id)
    after = bench.protected_manifest()
    bench.dump(run / 'preservation_after.json', after)
    assert before == after, 'Protected files changed'
    environment['end_utc'] = datetime.now(timezone.utc).isoformat()
    bench.dump(run / 'environment.json', environment)
    aggregate_rows, sample_rows = [], []
    metrics = ['prefill_ms', 'fixed_decode_ms', 'decode16_ms']
    for variant in VARIANTS:
        # Pool rounds before calculating quantiles so the CSV matches the README summary.
        rows = []
        for round_id in range(args.rounds):
            result = json.loads((run / 'workers' / f'bench_{args.device}_{variant}_r{round_id}.json').read_text())
            rows.extend(result['result']['rows'])
        for metric in metrics:
            for length in LENGTHS:
                selected = [row for row in rows
                            if row['metric'] == metric and row['length'] == length]
                values = [value for row in selected for value in row['samples']]
                aggregate_rows.append(dict(variant=variant, metric=metric, length=length,
                                           median=float(np.median(values)),
                                           q25=float(np.quantile(values, .25)),
                                           q75=float(np.quantile(values, .75)),
                                           n_samples=len(values)))
                for row in selected:
                    sample_rows.extend(dict(variant=variant, metric=metric, length=length,
                                            round=row['round'], repetition=index, ms=value)
                                       for index, value in enumerate(row['samples']))
    for name, rows in [('benchmark.csv', aggregate_rows), ('samples.csv', sample_rows)]:
        with (run / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('COMPLETE', run, flush=True)


if __name__ == '__main__':
    main()
