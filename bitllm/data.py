"""
Data preparation: TinyStories -> a 4096-token BPE -> flat uint16 token stream.

Run once before training:

    python -m bitllm.data --out data/tinystories --vocab-size 4096

Produces, in ``--out``:

    tokenizer.json   the trained BPE
    train.bin        uint16 token ids, one flat stream
    val.bin          same, held out
    meta.json        vocab size and token counts

Why a flat binary stream rather than a Dataset object: training reads random
windows out of a memory-mapped array, which costs nothing and never becomes
the bottleneck. At 15M parameters the GPU is fast enough that a slow data
path would dominate the run.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"

# TinyStories separates stories with this marker.
STORY_SEP = "<|endoftext|>"


def download(url: str, dest: Path):
    """Stream a file to disk, skipping if it is already there."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    import urllib.request
    print(f"  fetching {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    print(f"  wrote {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def train_tokenizer(text_path: Path, vocab_size: int, out_path: Path,
                    sample_bytes: int = 200_000_000):
    """Train a byte-level BPE on (a prefix of) the corpus.

    ``sample_bytes`` caps how much text the trainer reads. 200MB is far more
    than a 4k-vocab BPE needs to converge, and reading all 1.9GB just makes
    this step slow for no benefit.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    if out_path.exists():
        print(f"  have {out_path.name}")
        return Tokenizer.from_file(str(out_path))

    print(f"  training BPE, vocab_size={vocab_size} ...")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[STORY_SEP],           # id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def chunks():
        read = 0
        with open(text_path, "r", encoding="utf-8") as f:
            buf = []
            for line in f:
                buf.append(line)
                read += len(line)
                if len(buf) >= 10_000:
                    yield "".join(buf)
                    buf = []
                if read >= sample_bytes:
                    break
            if buf:
                yield "".join(buf)

    tok.train_from_iterator(chunks(), trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    print(f"  wrote {out_path.name}, real vocab = {tok.get_vocab_size()}")
    return tok


def encode_to_bin(text_path: Path, tok, out_path: Path, sep_id: int) -> int:
    """Encode the corpus to a flat uint16 array on disk.

    uint16 holds ids up to 65535, so any vocab under that fits in 2 bytes per
    token. At 4096 it is comfortable.
    """
    if out_path.exists():
        n = out_path.stat().st_size // 2
        print(f"  have {out_path.name} ({n:,} tokens)")
        return n

    print(f"  encoding {text_path.name} ...")
    total = 0
    tmp = out_path.with_suffix(".part")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(text_path, "r", encoding="utf-8") as f, open(tmp, "wb") as out:
        batch, nbytes = [], 0
        def flush(batch):
            nonlocal total
            if not batch:
                return
            for enc in tok.encode_batch(batch):
                ids = enc.ids + [sep_id]      # mark the story boundary
                arr = np.asarray(ids, dtype=np.uint16)
                out.write(arr.tobytes())
                total += arr.size

        story = []
        for line in f:
            if line.strip() == STORY_SEP:
                batch.append("".join(story).strip())
                story = []
                nbytes += 1
                if len(batch) >= 2000:
                    flush(batch)
                    batch, nbytes = [], 0
                    print(f"    {total:,} tokens", end="\r")
            else:
                story.append(line)
        if story:
            batch.append("".join(story).strip())
        flush(batch)

    tmp.rename(out_path)
    print(f"\n  wrote {out_path.name} ({total:,} tokens)")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tinystories")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--raw", default=None,
                    help="directory holding TinyStories-train.txt / -valid.txt "
                         "(skips download; use the Kaggle mount path here)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.raw:
        raw = Path(args.raw)
        train_txt, valid_txt = raw / "TinyStories-train.txt", raw / "TinyStories-valid.txt"
        assert train_txt.exists(), f"not found: {train_txt}"
    else:
        raw = out / "raw"
        train_txt, valid_txt = raw / "TinyStories-train.txt", raw / "TinyStories-valid.txt"
        print("[1/3] download")
        download(TRAIN_URL, train_txt)
        download(VALID_URL, valid_txt)

    print("[2/3] tokenizer")
    tok = train_tokenizer(train_txt, args.vocab_size, out / "tokenizer.json")
    sep_id = tok.token_to_id(STORY_SEP)
    assert sep_id is not None, "separator token missing from vocab"

    print("[3/3] encode")
    n_val = encode_to_bin(valid_txt, tok, out / "val.bin", sep_id)
    n_train = encode_to_bin(train_txt, tok, out / "train.bin", sep_id)

    meta = {
        "vocab_size": tok.get_vocab_size(),
        "sep_id": sep_id,
        "train_tokens": n_train,
        "val_tokens": n_val,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n" + json.dumps(meta, indent=2))
    print(f"\ntrain tokens: {n_train / 1e6:.1f}M "
          f"-- token floor for a clean single-epoch run is ~50M")


if __name__ == "__main__":
    main()
