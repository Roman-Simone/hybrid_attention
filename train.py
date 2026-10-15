"""
Training code inspired by Andrej Karpathy's nanoGPT implementation:
https://github.com/karpathy/nanoGPT/blob/master/train.py (MIT license).

Prepare the character-level dataset:
    python data/shakespeare_char/prepare.py

Train on Apple Silicon, CUDA, or CPU:
    python train.py --device mps
    python train.py --device cuda --dtype bfloat16 --compile
    python train.py --device cpu --n_layer 2 --n_head 2 --n_embd 128

Override hyperparameters and resume a checkpoint:
    python train.py --device mps --batch_size 16 --learning_rate 0.001
    python train.py --device mps --init_from resume --out_dir out

List all options:
    python train.py --help
"""

import argparse
import math
import pickle
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig


def parse_args():
    """Keep the training defaults and CLI overrides in one place."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data and checkpoints.
    parser.add_argument('--dataset', required=True) # ex: 'shakespeare_char'
    parser.add_argument('--out_dir', type=Path, default=Path('out'))
    parser.add_argument('--init_from', choices=['scratch', 'resume'], default='scratch')
    parser.add_argument('--eval_interval', type=int, default=250)
    parser.add_argument('--eval_iters', type=int, default=20)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_only', action='store_true')
    # Model and batch sizes.
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--attention_type', choices=['mha', 'mla'], default='mha')
    for name in ('kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim'):
        parser.add_argument(f'--{name}', type=int, default=None)
    parser.add_argument('--rope_theta', type=float, default=10000.0)
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='Override meta.pkl; defaults to 50304 when metadata is absent.')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--bias', action=argparse.BooleanOptionalAction, default=False)
    # AdamW and cosine learning rate decay.
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--decay_lr', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--lr_decay_iters', type=int, default=None,
                        help='Defaults to max_iters.')
    parser.add_argument('--min_lr', type=float, default=1e-4)
    # Use full precision on CPU/MPS; CUDA supports mixed precision.
    default_device = 'cuda' if torch.cuda.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--device', default=default_device)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='float32')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.max_iters
    for name in ('batch_size', 'block_size', 'gradient_accumulation_steps', 'n_layer',
                 'n_head', 'n_embd', 'eval_interval', 'eval_iters', 'log_interval', 'max_iters'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.n_embd % args.n_head:
        parser.error('--n_embd must be divisible by --n_head')
    if args.warmup_iters < 0 or (args.decay_lr and args.lr_decay_iters <= args.warmup_iters):
        parser.error('--lr_decay_iters must exceed --warmup_iters (which must be nonnegative)')
    if not 0 <= args.dropout < 1 or args.grad_clip < 0 or args.weight_decay < 0:
        parser.error('Invalid dropout, grad_clip, or weight_decay')
    if args.learning_rate <= 0 or not 0 <= args.min_lr <= args.learning_rate:
        parser.error('Require learning_rate > 0 and 0 <= min_lr <= learning_rate')
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        parser.error('AdamW betas must be in [0, 1)')
    if args.vocab_size is not None and args.vocab_size <= 0:
        parser.error('--vocab_size must be positive')
    if torch.device(args.device).type not in ('cpu', 'mps', 'cuda'):
        parser.error('--device must be cpu, mps, or cuda (optionally cuda:N)')
    if torch.device(args.device).type != 'cuda' and args.dtype != 'float32':
        parser.error('Use --dtype float32 on CPU/MPS; mixed precision is enabled only on CUDA')
    return args


def get_batch(data_dir, split, args):
    """Sample contiguous token sequences and their next-token targets."""
    # Reopen the memory map each batch, as in nanoGPT, to limit resident memory.
    data = np.memmap(data_dir / f'{split}.bin', dtype=np.uint16, mode='r')
    indices = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + args.block_size].astype(np.int64))
                     for i in indices.tolist()])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + args.block_size].astype(np.int64))
                     for i in indices.tolist()])
    if torch.device(args.device).type == 'cuda':
        return (x.pin_memory().to(args.device, non_blocking=True),
                y.pin_memory().to(args.device, non_blocking=True))
    return x.to(args.device), y.to(args.device)


@torch.no_grad()
def estimate_loss(model, data_dir, args, context):
    """Average several batches for a less noisy train/validation estimate."""
    losses = {}
    model.eval()
    for split in ('train', 'val'):
        total = 0.0
        for _ in range(args.eval_iters):
            x, y = get_batch(data_dir, split, args)
            with context:
                _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / args.eval_iters
    model.train()
    return losses


def get_lr(step, args):
    """Linear warmup followed by cosine decay to min_lr."""
    if not args.decay_lr:
        return args.learning_rate
    if step < args.warmup_iters:
        return args.learning_rate * (step + 1) / (args.warmup_iters + 1)
    if step >= args.lr_decay_iters:
        return args.min_lr
    ratio = (step - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (
        args.learning_rate - args.min_lr)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device_type = torch.device(args.device).type
    if device_type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    context = (torch.autocast('cuda', dtype=getattr(torch, args.dtype))
               if device_type == 'cuda' and args.dtype != 'float32' else nullcontext())
    scaler = torch.amp.GradScaler('cuda', enabled=device_type == 'cuda' and args.dtype == 'float16')

    # Initialize the model, or restore its architecture and training state.
    data_dir = Path(__file__).resolve().parent / 'data' / args.dataset
    checkpoint = None
    step, best_val_loss = 0, float('inf')
    if args.init_from == 'resume':
        checkpoint = torch.load(args.out_dir / 'ckpt.pt', map_location='cpu', weights_only=False)
        model_args = checkpoint['model_args']
        args.block_size = model_args['block_size']
        step, best_val_loss = checkpoint['iter_num'], checkpoint['best_val_loss']
    else:
        vocab_size = args.vocab_size
        if vocab_size is None:
            meta_path = data_dir / 'meta.pkl'
            if meta_path.exists():
                with meta_path.open('rb') as f:
                    vocab_size = pickle.load(f)['vocab_size']
            else:
                vocab_size = 50304
        model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                          block_size=args.block_size, vocab_size=vocab_size,
                          dropout=args.dropout, bias=args.bias,
                          attention_type=args.attention_type, kv_lora_rank=args.kv_lora_rank,
                          qk_nope_head_dim=args.qk_nope_head_dim,
                          qk_rope_head_dim=args.qk_rope_head_dim,
                          v_head_dim=args.v_head_dim, rope_theta=args.rope_theta)
    for split in ('train', 'val'):
        path = data_dir / f'{split}.bin'
        if not path.exists():
            raise FileNotFoundError(f'{path} is missing. Run data/{args.dataset}/prepare.py first.')
        if path.stat().st_size % 2 or path.stat().st_size // 2 <= args.block_size:
            raise ValueError(f'{path} must contain more than block_size uint16 tokens.')
    model = GPT(GPTConfig(**model_args)).to(args.device)
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate,
                                           (args.beta1, args.beta2), device_type)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        # Restore sampling/dropout RNG state for the same device backend.
        torch.set_rng_state(checkpoint['rng_state'])
        if checkpoint['device_type'] == device_type:
            if device_type == 'cuda':
                torch.cuda.set_rng_state_all(checkpoint['device_rng_state'])
            elif device_type == 'mps':
                torch.mps.set_rng_state(checkpoint['device_rng_state'])
    raw_model = model
    if args.compile:
        model = torch.compile(model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device: {args.device}; tokens per step: '
          f'{args.batch_size * args.block_size * args.gradient_accumulation_steps:,}')

    # Evaluate at intervals and at the end; ckpt.pt always contains the latest state.
    while True:
        if step % args.eval_interval == 0 or step >= args.max_iters or args.eval_only:
            losses = estimate_loss(model, data_dir, args, context)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if args.eval_only:
                break
            improved = losses['val'] < best_val_loss
            best_val_loss = min(best_val_loss, losses['val'])
            device_rng_state = None
            if device_type == 'cuda':
                device_rng_state = torch.cuda.get_rng_state_all()
            elif device_type == 'mps':
                device_rng_state = torch.mps.get_rng_state()
            state = dict(model=raw_model.state_dict(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), model_args=asdict(raw_model.config),
                         iter_num=step, best_val_loss=best_val_loss, config=vars(args),
                         rng_state=torch.get_rng_state(), device_type=device_type,
                         device_rng_state=device_rng_state)
            torch.save(state, args.out_dir / 'ckpt.pt')
            if improved:
                torch.save(state, args.out_dir / 'best.pt')
            if step >= args.max_iters:
                break

        start = time.perf_counter()
        lr = get_lr(step, args)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(args.gradient_accumulation_steps):
            x, y = get_batch(data_dir, 'train', args)
            with context:
                _, loss = model(x, y)
                loss = loss / args.gradient_accumulation_steps
            train_loss += loss.item()
            scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step % args.log_interval == 0:
            elapsed = (time.perf_counter() - start) * 1000
            print(f'step {step}: loss {train_loss:.4f}, lr {lr:.6g}, time {elapsed:.0f}ms')


if __name__ == '__main__':
    main()
