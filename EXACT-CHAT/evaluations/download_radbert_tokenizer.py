#!/usr/bin/env python3
"""
Download the roberta-base tokenizer (and base weights) used by RadBertClassifier.
Saves into ./radbert_local/ next to this script.
"""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

tok_dir = Path(__file__).parent / "radbert_local"
tok_dir.mkdir(parents=True, exist_ok=True)

repo_id   = "roberta-base"
repo_type = "model"
revision  = "main"

to_get = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]

saved = []
for fname in to_get:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=fname,
            revision=revision,
            local_dir=str(tok_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        saved.append(path)
    except Exception as e:
        print(f"[skip] {fname}: {e}")

print("Saved files:")
for p in saved:
    print(" -", p)
print("\nOK. Local model/tokenizer dir:", tok_dir)
