# Phase 1 — build the token stream (run once on Kaggle)

Notebook settings: **GPU T4 x2**, **Internet ON**. GPU isn't needed for this
phase, but keeping one notebook avoids re-attaching later.

## Cell 1 — verify the machine

```python
import torch
print(torch.__version__, torch.cuda.get_device_name(0))
print("bf16 native   :", torch.cuda.is_bf16_supported(including_emulation=False))
print("bf16 emulated :", torch.cuda.is_bf16_supported())
```

Expect `False` then `True`. That gap is why we use fp16.

## Cell 2 — get the code

```python
!git clone -q https://github.com/EuclidStellar/1-bit-LLM.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q tokenizers
```

## Cell 3 — build the data (~20-30 min)

Downloads TinyStories (1.9GB), trains a 4,096-token BPE, encodes to `uint16`.

```python
!python -m bitllm.data --out /kaggle/working/data --vocab-size 4096
```

## Cell 4 — verify it round-trips

```python
import numpy as np, json
from tokenizers import Tokenizer
tok  = Tokenizer.from_file("/kaggle/working/data/tokenizer.json")
meta = json.load(open("/kaggle/working/data/meta.json"))
arr  = np.memmap("/kaggle/working/data/train.bin", dtype=np.uint16, mode="r")

print(meta)
assert arr.max() < meta["vocab_size"], "token id out of range"
print(repr(tok.decode(arr[:80].tolist())))
```

You should read recognizable story text. If not, stop — nothing downstream will
work.

## Cell 5 — drop the 1.9GB of raw text

Keep the notebook output small enough to save as a Dataset.

```python
!rm -rf /kaggle/working/data/raw /kaggle/working/repo
!du -sh /kaggle/working/data/*
```

## Then: Save Version

**Save Version → Save & Run All (Commit).** When it finishes, the output becomes
a dataset you attach to every future training notebook — so you never download
or tokenize again.
