#!/usr/bin/env python3
"""
Download the roberta-base model + tokenizer used by classifier.py / dataset.py.
Saves into ./roberta_local/ next to this script.
"""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

model_dir = Path(__file__).parent / "roberta_local"
model_dir.mkdir(parents=True, exist_ok=True)

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
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        saved.append(path)
    except Exception as e:
        print(f"[skip] {fname}: {e}")

print("Saved files:")
for p in saved:
    print(" -", p)
print("\nLocal RoBERTa model/tokenizer dir:", model_dir)
print("Next: also run `python download_model.py` to fetch RadBertClassifier.pth,")
print("then move it into the same roberta_local/ directory.")
