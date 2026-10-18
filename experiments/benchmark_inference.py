"""Measure inference latency on the selected device and generate its figure.

This script measures prefill and cached decoding for four attention paths over
a dense prompt-length grid. It saves aggregate and raw CSV data, environment
details, and worker results, then generates a device-specific figure.
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
    parser.add_argument('--window', type=int, choices=[60, 64], default=64)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repetitions', type=int, default=10)
    args = parser.parse_args()

    run_dir = args.run.resolve()
    command = [sys.executable, str(ROOT / 'experiments/utils/benchmark_dense.py'),
               '--output', str(run_dir), '--device', args.device,
               '--window', str(args.window), '--rounds', str(args.rounds),
               '--warmups', str(args.warmups), '--repetitions', str(args.repetitions)]
    subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'experiments/utils/plot_readme.py'), 'inference', str(run_dir),
                    '--output', str(ROOT / 'figure')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
