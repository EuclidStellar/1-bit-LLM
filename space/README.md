---
title: TinyStories 1-bit LLM
emoji: 🔢
colorFrom: yellow
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# TinyStories 1-bit LLM

An 11,159,360-parameter BitNet b1.58 language model built from scratch, packed
into **2.31 MB** at 1.6 bits per weight.

Three tabs:

- **Generate** — sample from the 1-bit model, or from the fp32 control, or from
  the post-training-quantized arm that is supposed to be broken
- **Compare all three** — same prompt, same seed, three arms. This is the
  experiment
- **Look at the weights** — ternary weights print as a `-` `·` `+` grid. Almost
  no other model lets you read its parameters with your eyes

Full results, limitations, and an honest speed benchmark (it is **1.7× slower**
than fp32 in PyTorch) are in the
[model card](https://huggingface.co/euclidstellar/tinystories-1bit-llm).
