"""
Gradio Space for the TinyStories 1-bit LLM.

Two things you cannot do with most language models:
  1. load the whole thing in 2.31 MB
  2. read its weights with your eyes -- ternary weights have three states, so a
     weight matrix prints as a grid of - . +
"""

import math
import os

import numpy as np
import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from bitllm import load_packed_model, LM
from bitllm.model import BitLinear

MODEL_REPO = "euclidstellar/tinystories-1bit-llm"
DATA_REPO = "euclidstellar/tinystories-bpe4096"

TOK = Tokenizer.from_file(
    hf_hub_download(DATA_REPO, "tokenizer.json", repo_type="dataset"))

# the 2.31 MB model loads at startup; the 44.7 MB comparison arms are lazy
PACKED, HEADER = load_packed_model(
    hf_hub_download(MODEL_REPO, "model_packed.bin"))
_CACHE = {"1-bit (packed, 2.31 MB)": PACKED}

ARMS = {
    "1-bit (packed, 2.31 MB)":   None,                    # already loaded
    "fp32 control (22.3 MB)":    ("rung7_fp32.pt", "none", 0),
    "PTQ-ternary (broken)":      ("rung7_fp32.pt", "ternary", 8),
}


def get_arm(name):
    """Lazily fetch a comparison arm. PTQ is the fp32 checkpoint loaded into a
    model that quantizes in the forward pass -- no separate file needed."""
    if name in _CACHE:
        return _CACHE[name]
    fname, wmode, abits = ARMS[name]
    ck = torch.load(hf_hub_download(MODEL_REPO, fname),
                    map_location="cpu", weights_only=False)
    m = LM(vocab=4096, d=320, n_layer=8, n_head=8, mult=4,
           weight_mode=wmode, act_bits=abits, embed_mode="none")
    m.load_state_dict(ck["model"])
    m.eval()
    _CACHE[name] = m
    return m


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def generate(prompt, arm, max_new, temperature, top_k, seed):
    if not prompt.strip():
        return "Type a prompt. This model only knows simple children's stories."
    torch.manual_seed(int(seed))
    m = get_arm(arm)
    ids = torch.tensor([TOK.encode(prompt).ids])
    out = m.generate(ids, max_new=int(max_new), temperature=float(temperature),
                     top_k=int(top_k) if top_k > 0 else None)
    return TOK.decode(out[0].tolist())


def compare(prompt, max_new, temperature, seed):
    """Same prompt, same seed, three arms. This is the whole experiment."""
    blocks = []
    for name in ARMS:
        txt = generate(prompt, name, max_new, temperature, 100, seed)
        blocks.append(f"### {name}\n\n{txt}")
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------
# weight inspection
# --------------------------------------------------------------------------

GLYPH = {-1: "-", 0: "·", 1: "+"}

QUANT_LAYERS = [n for n, mod in PACKED.named_modules()
                if isinstance(mod, BitLinear)] or \
               [f"blocks.{i}.{p}" for i in range(8)
                for p in ("q", "k", "v", "o", "up", "down")]


def _states(name):
    mod = dict(PACKED.named_modules())[name]
    w = mod.weight.detach()
    scale = w.abs().max().clamp(min=1e-12)
    return (w / scale).round().clamp(-1, 1).to(torch.int8)


def inspect(name, rows, cols):
    s = _states(name)
    n = s.numel()
    neg = (s == -1).sum().item() / n
    zero = (s == 0).sum().item() / n
    pos = (s == 1).sum().item() / n
    r, c = min(int(rows), s.shape[0]), min(int(cols), s.shape[1])
    grid = "\n".join("".join(GLYPH[int(v)] for v in s[i, :c]) for i in range(r))
    stats = (f"**{name}** — {s.shape[0]} x {s.shape[1]} = {n:,} weights\n\n"
             f"| state | fraction |\n|---|---|\n"
             f"| `-1` | {neg:.2%} |\n| `0` (no connection) | {zero:.2%} |\n"
             f"| `+1` | {pos:.2%} |\n\n"
             f"Showing the top-left {r} x {c} corner. For Gaussian weights the "
             f"zero fraction is analytically "
             f"2·Φ(0.5·√(2/π)) − 1 = **31.01%**; training barely moves it.")
    return stats, grid


def sparsity_table():
    lines = ["| layer | weights | `-1` | `0` | `+1` |", "|---|---|---|---|---|"]
    for name in QUANT_LAYERS:
        s = _states(name)
        n = s.numel()
        lines.append(f"| `{name}` | {n:,} | {(s==-1).sum().item()/n:.1%} "
                     f"| {(s==0).sum().item()/n:.1%} "
                     f"| {(s==1).sum().item()/n:.1%} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

SIZE_MB = os.path.getsize(hf_hub_download(MODEL_REPO, "model_packed.bin")) / 1e6

HEAD = f"""
# TinyStories 1-bit LLM

An **11,159,360-parameter** BitNet b1.58 model built from scratch. The entire
model is **{SIZE_MB:.2f} MB** — 11,141,120 weights stored at **1.6 bits each**,
plus 18,240 fp32 normalization parameters.

| | |
|---|---|
| val loss / perplexity | 2.3107 / 10.1 |
| packed size | **{SIZE_MB:.2f} MB** — 9.65× smaller than fp16 |
| of the file | **99.2% ternary**, 0.79% full precision |
| speed | **1.7× slower** than fp32 in PyTorch — see the model card |

It writes simple children's stories. It knows no facts, cannot answer questions,
and **loses track of who is who after a few sentences** — it is 11× under-trained
by Chinchilla's rule, deliberately, because the point was measuring quantization
rather than maximizing quality.
"""

with gr.Blocks(title="TinyStories 1-bit LLM") as demo:
    gr.Markdown(HEAD)

    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", value="Once upon a time",
                                    lines=2)
                arm = gr.Radio(list(ARMS), value=list(ARMS)[0], label="Model")
                with gr.Row():
                    max_new = gr.Slider(20, 300, 150, step=10, label="New tokens")
                    temp = gr.Slider(0.1, 1.5, 0.8, step=0.05, label="Temperature")
                with gr.Row():
                    topk = gr.Slider(0, 500, 100, step=10,
                                     label="Top-k (0 = off)")
                    seed = gr.Number(value=0, label="Seed", precision=0)
                go = gr.Button("Generate", variant="primary")
            with gr.Column(scale=3):
                out = gr.Textbox(label="Output", lines=16, show_copy_button=True)
        go.click(generate, [prompt, arm, max_new, temp, topk, seed], out)
        gr.Markdown(
            "Comparison arms are 44.7 MB and download on first use. "
            "**PTQ-ternary** is the same fp32 weights quantized *after* training "
            "instead of during — it should be visibly broken, which is the point.")
        gr.Examples([["Once upon a time"], ["Lily saw a big"],
                     ["The dog was very"], ["One day, Tom and his mom"]], prompt)

    with gr.Tab("Compare all three"):
        gr.Markdown("Same prompt, same seed, three arms. Identical architecture "
                    "and training — the only variable is quantization.")
        cprompt = gr.Textbox(label="Prompt", value="Once upon a time")
        with gr.Row():
            cmax = gr.Slider(20, 200, 120, step=10, label="New tokens")
            ctemp = gr.Slider(0.1, 1.5, 0.8, step=0.05, label="Temperature")
            cseed = gr.Number(value=0, label="Seed", precision=0)
        cgo = gr.Button("Compare", variant="primary")
        cout = gr.Markdown()
        cgo.click(compare, [cprompt, cmax, ctemp, cseed], cout)

    with gr.Tab("Look at the weights"):
        gr.Markdown(
            "Ternary weights have three states, so a weight matrix prints as a "
            "grid. `-` is −1, `·` is zero (no connection), `+` is +1. Very few "
            "models let you read their parameters directly.")
        with gr.Row():
            layer = gr.Dropdown(QUANT_LAYERS, value=QUANT_LAYERS[0],
                                label="Layer")
            nrows = gr.Slider(8, 64, 28, step=4, label="Rows")
            ncols = gr.Slider(20, 160, 96, step=4, label="Columns")
        ilook = gr.Button("Show", variant="primary")
        istats = gr.Markdown()
        igrid = gr.Code(label="weights")
        ilook.click(inspect, [layer, nrows, ncols], [istats, igrid])

        gr.Markdown("### State distribution, every quantized layer")
        stab = gr.Button("Compute (48 layers)")
        stout = gr.Markdown()
        stab.click(lambda: sparsity_table(), None, stout)

    gr.Markdown(
        "---\n"
        "[Model card](https://huggingface.co/euclidstellar/tinystories-1bit-llm) · "
        "[Code and build notes](https://github.com/EuclidStellar/1-bit-LLM) · "
        "[BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285)")

if __name__ == "__main__":
    demo.launch()
