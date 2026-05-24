#!/usr/bin/env python3
"""
Download RadBertClassifier.pth (CT-RATE) into the current evaluation folder.

The CT-RATE dataset is public on HuggingFace; no auth token is required.
If you are behind a firewall you may set HF_ENDPOINT to a mirror, e.g.:
    export HF_ENDPOINT=https://hf-mirror.com
"""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

target_dir = Path(__file__).parent
target_dir.mkdir(parents=True, exist_ok=True)

local_path = hf_hub_download(
    repo_id="ibrahimhamamci/CT-RATE",
    repo_type="dataset",
    filename="models/RadBertClassifier.pth",
    revision="main",
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,
    resume_download=True,
    token=os.environ.get("HF_TOKEN"),
)

print("Saved to:", local_path)
