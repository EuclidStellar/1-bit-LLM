"""
Standalone CLI trainer.

NOTE: this is NOT the code that produced the published results. Those came from
notebooks/02_train_kaggle.py, run cell by cell on a Kaggle T4. This file is a
more elaborate variant (argparse, checkpoint/resume, AMP) written first and never
actually used -- it should work, but it is untested against the published numbers.

Training loop for BitLM.

    python -m bitllm.train --data data/tinystories --out runs/fp32    --weight-mode none
    python -m bitllm.train --data data/tinystories --out runs/ternary --weight-mode ternary

Both arms use identical everything except --weight-mode, which is the point:
the only free variable is how the weights are quantized.

Checkpointing is the default path, not an afterthought. Kaggle and Colab both
cap sessions at 12 hours and can kill you earlier, so every run saves full
state (model, optimizer, scaler, step, RNG) and resumes automatically if it
finds a checkpoint in --out.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from .model import Config, build_model


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_split(data_dir: Path, split: str, limit_tokens: int | None):
    """Memory-map a token stream. Nothing is read until a batch asks for it."""
    arr = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    if limit_tokens is not None and split == "train" and limit_tokens < len(arr):
        arr = arr[:limit_tokens]
    return arr


def get_batch(arr, batch_size, block_size, device):
    """Random windows out of the stream.

    Sampling random offsets rather than iterating in order means we never need
    an epoch boundary, and every step sees a fresh mix of stories.
    """
    ix = np.random.randint(0, len(arr) - block_size - 1, size=batch_size)
    x = np.stack([arr[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([arr[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    x = torch.from_numpy(x).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return x, y


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------

def lr_at(step, max_steps, base_lr, warmup, min_ratio=0.1):
    """Linear warmup then cosine decay to min_ratio * base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, max_steps - warmup)
    prog = min(1.0, max(0.0, prog))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, arr, batch_size, block_size, device, iters=40):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(arr, batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------

def save_ckpt(path: Path, model, opt, scaler, step, cfg, best_val, args):
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "config": cfg.__dict__,
        "best_val": best_val,
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, model, opt, scaler, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    torch.set_rng_state(ck["torch_rng"].cpu())
    np.random.set_state(ck["numpy_rng"])
    return ck["step"], ck.get("best_val", float("inf"))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tinystories")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")

    # model -- defaults are the locked ~11.1M config
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=320)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--weight-mode", default="ternary",
                   help="none | ternary | binary | int8 | int4 | int2")
    p.add_argument("--act-bits", type=int, default=8)
    p.add_argument("--keep-edge-blocks-fp", action="store_true",
                   help="pilot scaffolding: leave first/last block unquantized")

    # budget
    p.add_argument("--max-tokens", type=float, default=100e6,
                   help="total tokens to train on; sets max_steps")
    p.add_argument("--limit-tokens", type=float, default=None,
                   help="use only the first N tokens of train.bin")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)

    # optimizer
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # logging / checkpointing
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "off"])
    args = p.parse_args()

    device = pick_device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)
    meta = json.loads((data_dir / "meta.json").read_text())

    cfg = Config(
        vocab_size=meta["vocab_size"],
        n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
        block_size=args.block_size, weight_mode=args.weight_mode,
        act_bits=args.act_bits, keep_edge_blocks_fp=args.keep_edge_blocks_fp,
    )

    train_arr = load_split(data_dir, "train",
                           int(args.limit_tokens) if args.limit_tokens else None)
    val_arr = load_split(data_dir, "val", None)

    tokens_per_step = args.batch_size * cfg.block_size * args.grad_accum
    max_steps = int(args.max_tokens // tokens_per_step)

    model = build_model(cfg).to(device)
    rep = model.param_report()

    # AdamW, with weight decay only on matrices -- decaying norms and
    # embeddings hurts and is standard practice to skip.
    decay, no_decay = [], []
    seen = set()
    for name, prm in model.named_parameters():
        if id(prm) in seen:
            continue
        seen.add(id(prm))
        (decay if prm.dim() >= 2 else no_decay).append(prm)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # T4 is Turing: fp16 tensor cores, no bf16. Pick accordingly.
    amp = args.amp
    if amp == "auto":
        if device == "cuda":
            # NOTE: is_bf16_supported() defaults to including_emulation=True,
            # which returns True on a T4 (SM 7.5) even though Turing has no
            # bf16 tensor-core path and emulates it slowly. Ask for native
            # support only -- on a T4 this correctly picks fp16.
            amp = "bf16" if torch.cuda.is_bf16_supported(
                including_emulation=False) else "fp16"
        else:
            amp = "off"      # MPS autocast is not reliably faster here
    scaler = torch.amp.GradScaler(device) if amp == "fp16" else None
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)

    step, best_val = 0, float("inf")
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists() and not args.no_resume:
        step, best_val = load_ckpt(ckpt_path, model, opt, scaler, device)
        print(f"resumed from {ckpt_path} at step {step}")

    print(json.dumps({
        "device": device, "amp": amp, "weight_mode": cfg.weight_mode,
        "params_total": rep["total"], "params_ternary": rep["quantizable"],
        "params_full_precision": rep["full_precision"],
        "tokens_per_step": tokens_per_step, "max_steps": max_steps,
        "train_tokens_available": int(len(train_arr)),
        "tokens_per_param": round(args.max_tokens / rep["total"], 2),
    }, indent=2), flush=True)

    log_path = out / "log.jsonl"
    logf = open(log_path, "a")
    t_start = time.time()
    t_win, tok_win = time.time(), 0

    model.train()
    while step < max_steps:
        lr = lr_at(step, max_steps, args.lr, args.warmup, args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x, y = get_batch(train_arr, args.batch_size, cfg.block_size, device)
            if amp_dtype is not None:
                with torch.autocast(device_type=device, dtype=amp_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler is not None:
            scaler.step(opt); scaler.update()
        else:
            opt.step()

        step += 1
        tok_win += tokens_per_step

        if step % args.log_every == 0:
            dt = time.time() - t_win
            tps = tok_win / dt
            done = step / max_steps
            eta = (time.time() - t_start) / max(done, 1e-9) * (1 - done)
            rec = {"step": step, "loss": round(loss.item() * args.grad_accum, 4),
                   "lr": round(lr, 6), "tok_per_s": round(tps),
                   "eta_min": round(eta / 60, 1)}
            print(f"step {step:>6}/{max_steps}  loss {rec['loss']:.4f}  "
                  f"lr {lr:.2e}  {tps:>7,.0f} tok/s  eta {rec['eta_min']:.0f}m",
                  flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            t_win, tok_win = time.time(), 0

        if step % args.eval_every == 0 or step == max_steps:
            vl = estimate_loss(model, val_arr, args.batch_size, cfg.block_size, device)
            print(f"  ** val loss {vl:.4f}  (ppl {math.exp(min(vl, 20)):.1f})", flush=True)
            logf.write(json.dumps({"step": step, "val_loss": round(vl, 4)}) + "\n")
            logf.flush()
            if vl < best_val:
                best_val = vl
                save_ckpt(out / "best.pt", model, opt, scaler, step, cfg, best_val, args)

        if step % args.ckpt_every == 0 or step == max_steps:
            save_ckpt(ckpt_path, model, opt, scaler, step, cfg, best_val, args)

    save_ckpt(ckpt_path, model, opt, scaler, step, cfg, best_val, args)
    logf.close()
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min, best val {best_val:.4f}")


if __name__ == "__main__":
    main()
