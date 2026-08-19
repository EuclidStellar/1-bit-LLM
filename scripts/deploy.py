#!/usr/bin/env python
"""
Publish the model card and the Gradio Space.

    python scripts/deploy.py            # show what would happen
    python scripts/deploy.py --publish  # actually do it

Requires `hf auth login` with a write-role token. The token is read from the
huggingface_hub cache; it is never printed, logged, or passed as an argument.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
MODEL_CARD = ROOT / "model_card.md"
SPACE_DIR = ROOT / "space"
SPACE_FILES = ["app.py", "requirements.txt", "README.md"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true",
                    help="perform the uploads (default is a dry run)")
    ap.add_argument("--space-name", default="tinystories-1bit-llm")
    ap.add_argument("--private-space", action="store_true")
    a = ap.parse_args()

    api = HfApi()
    try:
        me = api.whoami()
    except Exception as e:
        sys.exit(f"not authenticated ({type(e).__name__}). run:  hf auth login")

    user = me["name"]
    role = me.get("auth", {}).get("accessToken", {}).get("role")
    model_repo = f"{user}/tinystories-1bit-llm"
    space_repo = f"{user}/{a.space_name}"

    print(f"account    {user}   (token role: {role})")
    print(f"model card {MODEL_CARD.name} -> {model_repo}/README.md "
          f"({MODEL_CARD.stat().st_size:,} bytes)")
    print(f"space      {space_repo}   sdk=gradio "
          f"private={a.private_space}")
    for f in SPACE_FILES:
        print(f"             {f} ({(SPACE_DIR / f).stat().st_size:,} bytes)")

    if role != "write":
        sys.exit("\ntoken is not write-role; uploads would fail. make a write "
                 "token at huggingface.co/settings/tokens and re-run hf auth login")

    if not a.publish:
        print("\nDRY RUN. nothing uploaded. re-run with --publish to go live.")
        return

    print("\nuploading model card ...")
    api.upload_file(path_or_fileobj=str(MODEL_CARD), path_in_repo="README.md",
                    repo_id=model_repo, repo_type="model")
    print(f"  https://huggingface.co/{model_repo}")

    print("creating space ...")
    api.create_repo(space_repo, repo_type="space", space_sdk="gradio",
                    private=a.private_space, exist_ok=True)
    for f in SPACE_FILES:
        api.upload_file(path_or_fileobj=str(SPACE_DIR / f), path_in_repo=f,
                        repo_id=space_repo, repo_type="space")
        print(f"  uploaded {f}")
    print(f"  https://huggingface.co/spaces/{space_repo}")
    print("\nthe space will build for a few minutes. watch its Logs tab.")


if __name__ == "__main__":
    main()
