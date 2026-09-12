# EXACT Pre-trained Checkpoints

Model weights for **EXACT: EXplainable Anomaly-aware ChesT CT Foundation Model**.

## Directory Structure

```
01_pretrain/
  ymamba_pretrain_best.pth        — Y-Mamba foundation model (zero-shot diagnosis & segmentation)

02_classification_finetune/
  classfine_best.pt               — Supervised multi-disease classifier head

03_segmentation_finetune/
  seg_covid_best.pth              — Supervised segmentation, fine-tuned on COVID-19
  seg_mosmed_best.pth             — Supervised segmentation, fine-tuned on MosMed
  seg_rex_best.pth                — Supervised segmentation, fine-tuned on ReX

04_exactchat_lora/
  checkpoint-38000/               — EXACT-CHAT LoRA adapter (inference-only)
    adapter_config.json
    adapter_model.safetensors     — LoRA weights (641 MB)
    non_lora_trainables.bin       — Projector + non-LoRA trainable params (1.1 GB)
    config.json
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
```

## Usage

1. Download this folder and place it at `EXACT/checkpoints/` in the project root.
2. For EXACT-CHAT, you also need the base LLM: `meta-llama/Llama-3.1-8B-Instruct` from HuggingFace.
3. Refer to the main `README.md` in the repository for detailed inference commands.

> **Do not rename `04_exactchat_lora/` or `checkpoint-38000/`.** LLaVA derives the
> model name from the last two path components and selects its LoRA loading branch
> by the substring `lora` in that name. Renaming either level makes the loader fall
> through to a non-LoRA branch, and the adapter is silently never applied.

## Total Size

~6.3 GB (DeepSpeed optimizer states excluded; only inference-essential files included).
