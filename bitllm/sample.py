"""
Generate text from a checkpoint, and look at the weights.

    python -m bitllm.sample --ckpt runs/ternary/best.pt --data data/tinystories
    python -m bitllm.sample --ckpt runs/ternary/best.pt --data data/tinystories --inspect
"""

import argparse
from pathlib import Path

import torch

from .model import Config, build_model
from .viz import print_ternary, sparsity_report, memory_report


def load(ckpt_path: Path, device: str):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ck["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/tinystories")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--inspect", action="store_true",
                   help="also print weight grids, sparsity and memory")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 ("mps" if torch.backends.mps.is_available() else "cpu")

    model, cfg, ck = load(Path(a.ckpt), device)
    print(f"step {ck['step']}  best_val {ck.get('best_val', float('nan')):.4f}  "
          f"weights={cfg.weight_mode}")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.data) / "tokenizer.json"))

    ids = tok.encode(a.prompt).ids
    for i in range(a.n):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, a.tokens, a.temperature, a.top_k)
        print(f"\n--- sample {i + 1} " + "-" * 50)
        print(tok.decode(out[0].tolist()))

    if a.inspect:
        memory_report(model)
        sparsity_report(model)
        for name, m in model.named_modules():
            if type(m).__name__ == "BitLinear" and m.weight_mode != "none":
                print_ternary(m, rows=18, cols=76, title=name)
                break


if __name__ == "__main__":
    main()
