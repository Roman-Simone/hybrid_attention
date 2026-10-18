"""Evaluate language-model quality and generate the KV-cache memory figure.

This script evaluates the best and final checkpoints, checks cached-inference
correctness and memory use, and generates a device-specific cache figure. It
saves evaluation metrics and benchmark measurements in the selected run folder.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='New run directory for evaluation results.')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], required=True)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repetitions', type=int, default=10)
    args = parser.parse_args()

    run_dir = args.run.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation = run_dir / 'evaluation.json'
    benchmark = run_dir / f'benchmark_{args.device}'
    if evaluation.exists() or benchmark.exists():
        parser.error(f'{run_dir} already contains evaluation or benchmark results')

    run([sys.executable, str(ROOT / 'experiments/utils/evaluate.py'), str(run_dir), '--device', args.device])
    run([sys.executable, str(ROOT / 'experiments/utils/benchmark.py'), '--output', str(benchmark),
         '--device', args.device, '--rounds', str(args.rounds), '--warmups', str(args.warmups),
         '--repetitions', str(args.repetitions)])
    run([sys.executable, str(ROOT / 'experiments/utils/plot_readme.py'), 'cache', str(benchmark),
         '--evaluation', str(evaluation), '--output', str(ROOT / 'figure')])


if __name__ == '__main__':
    main()
