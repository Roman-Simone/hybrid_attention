"""Measure autoregressive generation on the selected device and generate its figure.

This script runs cached greedy generation, correctness checks, and shared-target
context evaluation. It aggregates worker timings into ``generation.csv`` and
generates a device-specific performance figure.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='New directory for this benchmark campaign.')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], required=True)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=4)
    parser.add_argument('--context-batch', type=int, default=8)
    args = parser.parse_args()

    run_dir = args.run.resolve()
    command = [sys.executable, str(ROOT / 'experiments/utils/extend_evaluation.py'), '--run', str(run_dir),
               '--device', args.device,
               '--rounds', str(args.rounds), '--warmups', str(args.warmups),
               '--repetitions', str(args.repetitions), '--context-batch', str(args.context_batch)]
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'experiments/utils/plot_readme.py'), 'generation', str(run_dir),
                    '--output', str(ROOT / 'figure')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
