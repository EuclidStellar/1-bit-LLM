# ---------------------------------------------------------------------------
# The actual training run, as executed on one free Kaggle T4.
#
# This is the code that produced every number in the model card. Paste each
# block below into a Kaggle notebook cell, in order (they are marked `# %%`),
# or run the file top to bottom in a GPU environment.
#
# Prerequisites
#   Kaggle notebook, Accelerator = GPU T4 x2, Internet = ON.
#   Data comes from HF, so notebooks/01_data_kaggle.md only needs running once
#   (it built the 4,096-token BPE and the 477,236,558-token stream).
#
# Total cost: about 2 hours of GPU time for all four arms.
# ---------------------------------------------------------------------------

# %% ------------------------------------------------------------------ cell 1
# Setup: code from GitHub, data from HF, batching.

import os, sys, math, time, json, subprocess, importlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO_DIR = "/kaggle/working/repo"
subprocess.run(["rm", "-rf", REPO_DIR], check=False)
subprocess.run(["git", "clone", "-q",
                "https://github.com/EuclidStellar/1-bit-LLM.git", REPO_DIR], check=True)
for k in [k for k in list(sys.modules) if k.startswith("bitllm")]:
    del sys.modules[k]
sys.path.insert(0, REPO_DIR)
importlib.invalidate_caches()          # REQUIRED: a failed clone poisons the
                                       # import cache and the files stay invisible
from bitllm import (LM, BitLinear, fp32_model, ternary_model,
                    ternary_embed_model, weight_ternary, weight_intk)

V, D = 4096, 320
DS = "euclidstellar/tinystories-bpe4096"
paths = {f: hf_hub_download(DS, f, repo_type="dataset")
         for f in ["tokenizer.json", "train-20m.bin", "val.bin"]}
train = np.memmap(paths["train-20m.bin"], dtype=np.uint16, mode="r")
val   = np.memmap(paths["val.bin"],       dtype=np.uint16, mode="r")
tk    = Tokenizer.from_file(paths["tokenizer.json"])
dev   = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(arr, bs, T):
    """Random windows out of a flat uint16 token stream.

    targets[t] is the token that FOLLOWS inputs[t]. Sampling random offsets
    rather than iterating means no epoch boundary and no shuffle buffer.
    """
    ix = np.random.randint(0, len(arr) - T - 1, size=bs)
    x = np.stack([arr[i:i + T]         for i in ix]).astype(np.int64)
    y = np.stack([arr[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


print(dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
# NOTE: torch.cuda.is_bf16_supported() defaults to including_emulation=True and
# returns True on a T4, which has NO bf16 tensor-core path. Ask for native only.
print("bf16 native:", torch.cuda.is_bf16_supported(including_emulation=False))


# %% ------------------------------------------------------------------ cell 2
# The shared harness. Every arm uses this UNCHANGED -- identical data, batch,
# steps, optimizer and seed. The model is the only free variable, which is what
# makes the comparison mean anything.

TOKENS, BS, T = 20_000_000, 32, 256
STEPS, BASE_LR, WARMUP = TOKENS // (BS * T), 1e-3, 100     # STEPS = 2441


@torch.no_grad()
def evaluate(m, iters=60):
    m.eval()
    ls = [m(*get_batch(val, BS, T))[1].item() for _ in range(iters)]
    m.train()
    return float(np.mean(ls))


def run(build, label, lr=BASE_LR):
    torch.manual_seed(1337); np.random.seed(1337)
    m = build().to(dev)
    n = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        # linear warmup then cosine decay to 10%
        if step <= WARMUP:
            cur = lr * step / WARMUP
        else:
            prog = (step - WARMUP) / (STEPS - WARMUP)
            cur = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur

        x, y = get_batch(train, BS, T)
        opt.zero_grad(set_to_none=True)
        _, l = m(x, y)
        l.backward()
        # NOT optional. Without clipping, the ReLU-squared FFN diverges: on a
        # probe task the same run gave 8.0065 unclipped and 0.0144 clipped.
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

        if step % 500 == 0 or step == STEPS:
            print(f"  {step:>5}/{STEPS}  train {l.item():.4f}  lr {cur:.2e}")

    vl = evaluate(m)
    print(f"\n[{label}]  params {n:,}   VAL {vl:.4f}   ppl {math.exp(vl):.1f}"
          f"   {time.time() - t0:.0f}s")
    return m, vl


@torch.no_grad()
def causality_test(m, label=""):
    """Change a LATER token; the logits at position 5 must not move. This bug
    raises no exception -- you get a beautiful loss curve and a model that
    cannot generate, because at inference time the future does not exist."""
    m.eval()
    x = torch.randint(0, V, (1, 16)).to(dev)
    a = m(x)[0][0, 5].clone()
    x[0, 6] = (x[0, 6] + 1) % V
    d = (a - m(x)[0][0, 5]).abs().max().item()
    print(f"causality {label}: {'PASS' if d < 1e-4 else 'FAIL, MASK LEAKING'}"
          f"  (delta {d:.2e})")
    m.train()


print(f"{STEPS:,} steps x {BS * T:,} tokens = {STEPS * BS * T:,} tokens per arm")


# %% ------------------------------------------------------------------ cell 3
# Sanity checks before spending GPU time.

for label, build in [("fp32", fp32_model), ("ternary", ternary_model)]:
    s = build().param_split()
    print(f"[{label:<7}] total {s['total']:,}  quantized {s['quantized']:,} "
          f"({s['quantized'] / s['total']:.1%})  fp {s['full_precision']:,}")
causality_test(ternary_model().to(dev), "ternary")
# expect: total 11,159,360   quantized 9,830,400 (88.1%)   fp 1,328,960


# %% ------------------------------------------------------------------ cell 4
# Arm 1: fp32 control.                                    measured VAL 2.0553
m_fp32, v_fp32 = run(fp32_model, "fp32 control")

# %% ------------------------------------------------------------------ cell 5
# Arm 2: ternary body, fp16 embedding, quantized DURING training.
#                                                         measured VAL 2.1760
m_qat, v_qat = run(ternary_model, "QAT-ternary")
causality_test(m_qat, "QAT")

# %% ------------------------------------------------------------------ cell 6
# Arm 3: PTQ. Costs ZERO GPU time -- state_dict keys are identical across arms,
# so loading fp32 weights into a model that ternarizes in its forward pass IS
# post-training quantization.                             measured VAL 5.0229
m_ptq = ternary_model().to(dev)
m_ptq.load_state_dict(m_fp32.state_dict())
v_ptq = evaluate(m_ptq)
print(f"[PTQ-ternary] VAL {v_ptq:.4f}  ppl {math.exp(v_ptq):.1f}   (no training)")
print(f"QAT recovered {(v_ptq - v_qat) / (v_ptq - v_fp32):.1%} of what PTQ destroyed")

# %% ------------------------------------------------------------------ cell 7
# Arm 4: ternary body AND ternary tied embedding, both QAT. Beyond BitNet, which
# holds embeddings at bf16. This is the model that ships. measured VAL 2.3107
m_te, v_te = run(ternary_embed_model, "QAT-ternary + ternary embed")
causality_test(m_te, "QAT-both")

# %% ------------------------------------------------------------------ cell 8
# Arm 5: the equal-MEMORY comparison. Same ~4.5 MB budget spent on fp16
# precision instead of parameter count.                    measured VAL 2.3607
m_small, v_small = run(lambda: fp32_model(d=128, n_head=8), "fp16 d=128")

# %% ------------------------------------------------------------------ cell 9
# Results.
print(f"\n{'arm':<34}{'val':>9}{'ppl':>9}")
print("-" * 52)
for lbl, v in [("fp32 control", v_fp32), ("QAT-ternary", v_qat),
               ("QAT-ternary + ternary embed", v_te),
               ("PTQ-ternary (no training)", v_ptq),
               ("fp16 d=128 (equal memory)", v_small)]:
    print(f"{lbl:<34}{v:>9.4f}{math.exp(v):>9.1f}")

# Both of these are true, and reporting only one misleads:
#   equal PARAMETER count -> ternary loses 0.1207 nats
#   equal MEMORY budget   -> ternary wins ~0.17 nats (2.31 MB buys 11.1M ternary
#                            parameters or 1.1M fp16 ones)


# %% ----------------------------------------------------------------- cell 10
# Pack the winner into real bits and save everything.
from bitllm.pack import save_packed, load_packed_model

TRAIN_CFG = dict(vocab=4096, d=320, n_layer=8, n_head=8, mult=4,
                 weight_mode="ternary", act_bits=8, embed_mode="ternary")
info = save_packed(m_te, "/kaggle/working/model_packed.bin", config=TRAIN_CFG,
                   full_dtype="fp32", meta={"val_loss": v_te})
print(info)     # 2,313,205 bytes, 1.6 bits/weight, 49 packed tensors

# The packed file IS the model, not an approximation of it: load_packed_model
# builds with weight_mode="none" because the stored weights are already
# quantized. Re-quantizing them would recompute the absmean scale as
# g*(1 - zero_fraction) ~ 0.686g and shrink every weight by 31%.
m_packed, header = load_packed_model("/kaggle/working/model_packed.bin", dev)
print(f"packed {evaluate(m_packed):.6f}  vs  source {evaluate(m_te):.6f}")

torch.save({"model": m_te.state_dict(), "config": TRAIN_CFG,
            "val_loss": v_te}, "/kaggle/working/qat_ternary_embed.pt")

from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi
api = HfApi(token=UserSecretsClient().get_secret("hf_token"))
for f in ["model_packed.bin", "qat_ternary_embed.pt"]:
    api.upload_file(path_or_fileobj=f"/kaggle/working/{f}", path_in_repo=f,
                    repo_id=f"{api.whoami()['name']}/tinystories-1bit-llm")
    print("uploaded", f)
