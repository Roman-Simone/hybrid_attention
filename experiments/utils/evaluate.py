"""
Evaluate the best and final checkpoints for MHA, MLA, SWA and hybrid.

Measure loss on the full validation split, then compare context lengths on the
same target tokens. Save the results to evaluation.json in an existing run folder.

Example:
    python experiments/utils/evaluate.py experiments/run_language_modeling --device mps
"""

import argparse
import gc
import math
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from benchmark import ROOT, GPT, GPTConfig, sha, dump, protected_manifest


def parse_args():
    """Keep command-line options in one place, as in train.py."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path, help='Existing folder where evaluation.json will be saved.')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    return parser.parse_args()


def load_validation_data():
    """Check that the token files match the tokenizer and the original 90/10 split."""
    data_dir = ROOT / 'data' / 'shakespeare_char'
    validation_data = np.fromfile(data_dir / 'val.bin', dtype=np.uint16)
    with (data_dir / 'meta.pkl').open('rb') as f:
        metadata = pickle.load(f)

    text = (data_dir / 'input.txt').read_text()
    encoded = np.array([metadata['stoi'][char] for char in text], dtype=np.uint16)
    split_index = int(0.9 * len(encoded))
    train_data = np.fromfile(data_dir / 'train.bin', dtype=np.uint16)
    assert np.array_equal(encoded[:split_index], train_data)
    assert np.array_equal(encoded[split_index:], validation_data)
    return validation_data, split_index, metadata['vocab_size']


def tensor(data, device):
    """Convert token IDs to a tensor with shape (1, sequence_length)."""
    return torch.tensor(np.asarray(data, dtype='int64'), device=device)[None]


def validation(model, data, device):
    """Evaluate every next-token target once, resetting context between blocks."""
    total_nll = 0.0
    token_count = 0
    block_size = model.config.block_size

    for start in range(0, len(data) - 1, block_size):
        length = min(block_size, len(data) - 1 - start)
        x = tensor(data[start:start + length], device)
        y = tensor(data[start + 1:start + length + 1], device)
        # Passing targets makes GPT return logits for every position in the block.
        logits, _ = model(x, y)
        block_nll = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='sum')
        total_nll += float(block_nll)
        token_count += length

    # Sum token losses before averaging: the final block may be shorter.
    assert token_count == len(data) - 1
    loss = total_nll / token_count
    return dict(n=token_count, nll_sum=total_nll, loss=loss,
                perplexity=math.exp(loss), bpc=loss / math.log(2))


def context(model, data, device):
    """Compare context lengths using the same 128 target positions each time."""
    positions = np.linspace(256, len(data) - 2, 128, dtype=int)
    rows = []
    for length in [16, 32, 64, 128, 256]:
        samples = []
        for position in positions:
            # The target is excluded from the input, which contains only its past.
            x = tensor(data[position - length:position], device)
            logits, _ = model(x)
            prediction = logits[:, -1]
            target = torch.tensor([int(data[position])], device=device)
            samples.append(float(F.cross_entropy(prediction, target)))
        rows.append(dict(length=length, n=len(positions), loss=float(np.mean(samples)),
                         positions=positions.tolist(), samples=samples))
    return rows


@torch.inference_mode()
def main():
    args = parse_args()
    output = args.run / 'evaluation.json'
    if output.exists():
        raise FileExistsError(output)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; GPU access required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; a CUDA-capable GPU is required')

    torch.manual_seed(1337)
    torch.set_num_threads(4)
    before = protected_manifest()
    data, train_characters, vocab_size = load_validation_data()
    results = dict(start_utc=datetime.now(timezone.utc).isoformat(), device=args.device,
                   split_verified=True, train_characters=train_characters,
                   validation_characters=len(data), vocab_size=vocab_size, models={})

    for attention_type in ['mha', 'mla', 'swa', 'hybrid']:
        results['models'][attention_type] = {}
        for checkpoint_name in ['best.pt', 'ckpt.pt']:
            checkpoint_path = ROOT / 'out' / attention_type / checkpoint_name
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model = GPT(GPTConfig(**checkpoint['model_args'])).eval()
            model.load_state_dict(checkpoint['model'])
            model.to(args.device)

            result = dict(checkpoint=str(checkpoint_path.relative_to(ROOT)),
                          sha256=sha(checkpoint_path), iteration=checkpoint['iter_num'],
                          config=asdict(model.config),
                          parameters=sum(p.numel() for p in model.parameters()),
                          quality=validation(model, data, args.device))
            # The context comparison uses the best checkpoint, as do the benchmarks.
            if checkpoint_name == 'best.pt':
                result['context'] = context(model, data, args.device)
            results['models'][attention_type][checkpoint_name] = result

            # Save progress after each checkpoint and release memory for the next one.
            dump(output, results)
            print(attention_type, checkpoint_name, 'step', result['iteration'],
                  'loss', result['quality']['loss'], flush=True)
            del model, checkpoint
            gc.collect()
            if args.device == 'mps':
                torch.mps.empty_cache()

    after = protected_manifest()
    assert before == after, 'Protected artifacts changed during evaluation'
    results.update(end_utc=datetime.now(timezone.utc).isoformat(), preservation_verified=True)
    dump(output, results)
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
