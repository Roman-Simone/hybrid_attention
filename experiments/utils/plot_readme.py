"""Generate benchmark figures from completed inference and evaluation runs.

The cache command reads verification and evaluation JSON, the inference command
reads benchmark CSV data, and the generation command aggregates per-worker JSON
into ``generation.csv``. Figures are saved to ``figure/`` by default. This
script does not run model inference or modify checkpoints.
"""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hybrid_attention_matplotlib')
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

KINDS = ['mha', 'mla', 'swa', 'hybrid']
VARIANTS = {'mha': 'mha_sdpa', 'mla': 'mla_native',
            'swa': 'swa_sdpa', 'hybrid': 'hybrid_native'}
LABELS = {'mha': 'MHA · SDPA', 'mla': 'MLA · native',
          'swa': 'SWA · SDPA', 'hybrid': 'Hybrid · native'}
COLORS = {'mha': '#2563eb', 'mla': '#dc2626',
          'swa': '#059669', 'hybrid': '#7c3aed'}
ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def output_path(run, requested, name):
    destination = requested.resolve() if requested else ROOT / 'figure'
    destination.mkdir(parents=True, exist_ok=True)
    return destination / name


def configure_plots():
    """Keep typography and axes consistent with the original report figures."""
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})


def plot_inference(args):
    run = args.run.resolve()
    env = read(run / 'environment.json')
    rows = list(csv.DictReader((run / 'benchmark.csv').open()))
    lengths = sorted({int(row['length']) for row in rows})
    metrics = [('prefill_ms', 'Isolated prefill', 'ms/prompt'),
               ('fixed_decode_ms', 'Single step with fixed cache', 'ms/token'),
               ('decode16_ms', '16 consecutive steps', 'ms/token, mean over 16 steps')]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for ax, (metric, title, unit) in zip(axes, metrics):
        for kind in KINDS:
            variant = VARIANTS[kind]
            selected = {int(row['length']): row for row in rows
                        if row['variant'] == variant and row['metric'] == metric}
            xs = [length for length in lengths if length in selected]
            medians = [float(selected[length]['median']) for length in xs]
            q25 = [float(selected[length]['q25']) for length in xs]
            q75 = [float(selected[length]['q75']) for length in xs]
            ax.plot(xs, medians, '.-', color=COLORS[kind], markersize=3,
                    linewidth=1.5, label=LABELS[kind])
            ax.fill_between(xs, q25, q75, color=COLORS[kind], alpha=.12)

        ax.set(title=title, xlabel='Initial prompt length (tokens)', ylabel=unit,
               xticks=[16, env['window'], 96, 144, 192, 240],
               xlim=(12, 244), ylim=(0, None))
        ax.xaxis.set_minor_locator(MultipleLocator(4))
        ax.axvline(env['window'], color='#9ca3af', linestyle=':', linewidth=1)
        ax.grid(axis='y', alpha=.2)

    axes[0].legend(fontsize=8)
    device = env['device'].upper()
    fig.suptitle(f'{device} · {env["threads"]} threads · batch {env["batch"]} · '
                 f'{env["dtype"]} · median and IQR\n'
                 f'Measured every 4 tokens · SWA/Hybrid window: {env["window"]} tokens')
    path = output_path(run, args.output, f'tempi_{env["device"]}.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def plot_cache(args):
    run = args.run.resolve()
    evaluation = read(args.evaluation)
    memory_series, context_series = [], []
    for kind in KINDS:
        variant = VARIANTS[kind]
        verify = read(run / 'workers' / f'verify_{evaluation["device"]}_{variant}.json')
        assert verify['result']['passed']
        memory = [row for row in verify['result']['memory']
                  if row['mode'] == 'prefill' and row['length'] <= 240]
        memory_series.append((kind, memory))
        context = evaluation['models'][kind]['best.pt']['context']
        context_series.append((kind, context))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for kind, rows in memory_series:
        axes[0].plot([row['length'] for row in rows],
                     [row['storage_bytes'] / 1024 for row in rows], '.-',
                     color=COLORS[kind], label=kind.upper())
    axes[0].set(title='Actual cache storage, six layers',
                xlabel='Prompt length (tokens)', ylabel='Persistent cache (KiB)',
                xticks=[16, 64, 128, 192, 240], ylim=(0, None))
    axes[0].axvline(evaluation.get('window', 64), color='#9ca3af',
                    linestyle=':', linewidth=1)
    axes[0].grid(axis='y', alpha=.2)
    axes[0].legend()

    for kind, rows in context_series:
        axes[1].plot([row['length'] for row in rows], [row['loss'] for row in rows],
                     '.-', color=COLORS[kind], label=kind.upper())
    axes[1].set(title='Quality on 128 fixed targets',
                xlabel='Available context (tokens)',
                ylabel='Cross-entropy (nats/character)',
                xticks=[16, 32, 64, 128, 256])
    axes[1].grid(axis='y', alpha=.2)
    axes[1].legend()

    path = output_path(run, args.output,
                       f'memoria_contesto_{evaluation["device"]}.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def plot_generation(args):
    run = args.run.resolve()
    env = read(run / 'environment.json')
    rows = []
    for kind in KINDS:
        # Pool timed samples before calculating the median and quartiles.
        samples = []
        for round_id in range(env['rounds']):
            worker = read(run / 'workers' / f'generation_{kind}_r{round_id}.json')
            assert worker['device'] == env['device']
            samples.extend(worker['result']['rows'])
        for prompt in env['prompt_lengths']:
            for count in env['generation_lengths']:
                selected = [row for row in samples
                            if row['prompt_length'] == prompt
                            and row['generated_tokens'] == count]
                assert len(selected) == env['rounds'] * env['repetitions']
                row = dict(model=kind, prompt_length=prompt, generated_tokens=count,
                           n=len(selected),
                           regime='context_reset' if prompt + count - 1 > 256 else 'within_context',
                           cache_resets=selected[0]['cache_resets'],
                           final_cache_bytes=selected[0]['final_cache_bytes'])
                for metric in ('total_ms', 'ttft_ms', 'tokens_per_second'):
                    values = [sample[metric] for sample in selected]
                    row[f'{metric}_median'] = float(np.median(values))
                    row[f'{metric}_q25'] = float(np.quantile(values, .25))
                    row[f'{metric}_q75'] = float(np.quantile(values, .75))
                rows.append(row)

    fields = list(rows[0])
    output_csv = args.output_csv or run / 'generation.csv'
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = [('total_ms', 'Total generation time (ms)'),
               ('ttft_ms', 'Time to first CPU character (ms)'),
               ('tokens_per_second', 'End-to-end throughput (tokens/s)')]
    for i, prompt in enumerate(env['prompt_lengths']):
        for ax, (metric, label) in zip(axes[i], metrics):
            ax.axvspan(181, 290, color='#94a3b8', alpha=.13)
            ax.text(256, .12, 'Context\nresets', transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=8, color='#475569')
            for kind in KINDS:
                selected = sorted((row for row in rows
                                   if row['model'] == kind
                                   and row['prompt_length'] == prompt),
                                  key=lambda row: row['generated_tokens'])
                medians = np.array([row[f'{metric}_median'] for row in selected])
                q25 = np.array([row[f'{metric}_q25'] for row in selected])
                q75 = np.array([row[f'{metric}_q75'] for row in selected])
                ax.errorbar([row['generated_tokens'] for row in selected], medians,
                            yerr=[medians - q25, q75 - medians], fmt='o-',
                            color=COLORS[kind], label=kind.upper(), markersize=4,
                            capsize=3)
            ax.set_xscale('log', base=2)
            ax.set(xlim=(13, 290), ylim=(0, None), xlabel='Generated tokens',
                   ylabel=label, title=f'Prompt: {prompt} tokens',
                   xticks=env['generation_lengths'])
            ax.set_xticklabels(env['generation_lengths'])
            ax.grid(axis='y', alpha=.2)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{env["device"].upper()} cached greedy generation · batch 1 · '
                 'float32 · median and IQR\n'
                 'Text input → first character on CPU → complete generated text')
    path = output_path(run, args.output, f'generation_{env["device"]}.png')
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(path)


def main():
    configure_plots()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='kind', required=True)
    inference = subparsers.add_parser('inference', help='Plot the dense inference benchmark.')
    inference.add_argument('run', type=Path)
    inference.add_argument('--output', type=Path)
    inference.set_defaults(func=plot_inference)
    cache = subparsers.add_parser('cache', help='Plot validation quality and KV cache use.')
    cache.add_argument('run', type=Path)
    cache.add_argument('--evaluation', type=Path, required=True)
    cache.add_argument('--output', type=Path)
    cache.set_defaults(func=plot_cache)
    generation = subparsers.add_parser('generation', help='Aggregate and plot device generation.')
    generation.add_argument('run', type=Path)
    generation.add_argument('--output', type=Path)
    generation.add_argument('--output-csv', type=Path)
    generation.set_defaults(func=plot_generation)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
