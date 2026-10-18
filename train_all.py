"""Train MHA, MLA, SWA and hybrid sequentially with the same training options."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='Additional options are passed to train.py, e.g. --max_iters 1000 --tensorboard.')
    parser.add_argument('--dataset', default='shakespeare_char')
    parser.add_argument('--out_dir', type=Path, default=Path('out'))
    args, training_args = parser.parse_known_args()
    train_script = Path(__file__).resolve().parent / 'train.py'

    for attention_type in ('mha', 'mla', 'swa', 'hybrid'):
        out_dir = args.out_dir / attention_type
        print(f'\nTraining {attention_type.upper()} -> {out_dir}', flush=True)
        command = [sys.executable, str(train_script), *training_args,
                   '--dataset', args.dataset,
                   '--attention_type', attention_type,
                   '--out_dir', str(out_dir)]
        subprocess.run(command, check=True)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
