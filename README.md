# EXACT: EXplainable Anomaly-aware ChesT CT Foundation Model

---

## Table of Contents

- [About](#about)
- [Method Overview](#method-overview)
- [Code Structure](#code-structure)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Training](#training)
- [Inference](#inference)
- [Experimental Results](#experimental-results)
- [Visualization](#visualization)
- [Citation](#citation)

---

## About

**EXACT** (**EX**plainable **A**nomaly-aware **C**hes**T** CT Foundation Model) is an explainable anomaly-aware vision foundation model for 3D chest CT that unifies global disease understanding with voxel-level visual grounding in a single framework.

Unlike prior CLIP-style 3D foundation models that compress volumetric features into global embeddings — and consequently lose the spatial information required for lesion localization and clinical interpretability — EXACT learns directly from the voxel level. It is pre-trained on **25,692** paired CT scans and radiology reports (21,304 patients, CT-RATE) under **anatomy-aware weak supervision**, jointly learning organ segmentation and multi-instance anomaly detection **without any manual voxel-level annotation**. The resulting **Anomaly-aware Maps (AAmaps)** simultaneously encode lesion extent and organ-specific pathological context.

Key advantages:
- **Intrinsic spatial interpretability.** A Y-shaped Mamba backbone (**Y-Mamba**) couples a shared encoder with an organ-segmentation decoder and a multi-instance anomaly-detection decoder, producing 18-channel voxel-level AAmaps directly — no post-hoc CAM or saliency tricks required.
- **Anatomy-aware weak supervision.** Organ masks come automatically from Segment-Anything-by-Text; disease pseudo-labels come from RadBERT-parsed radiology reports. No radiologist-drawn lesion masks are needed at any stage.
- **Five downstream capabilities from one pre-trained backbone.** Zero-shot multi-disease diagnosis, fine-tuned diagnosis, zero-shot anomaly localization, fine-tuned segmentation (EXACT-Seg), and visually grounded radiology report generation (EXACT-CHAT) — all driven by the same AAmap representation.
- **Visually grounded radiology reports.** **EXACT-CHAT** is a multimodal AI assistant adapted from the LLaVA framework that integrates the frozen EXACT image encoder, a multimodal projector, and LLaMA-3.1-8B-Instruct. Structured diagnostic priors from the frozen AAmap classifier are additionally provided to the LLM as text tokens, encoding the predicted disease states across all target abnormalities. An optional GPT-4.1 refinement step yields **EXACT-CHAT (Refined)**, which calibrates the initial reports against upstream disease predictions to suppress hallucinations.
- **Consistent SOTA across multinational, multi-center cohorts.** EXACT outperforms state-of-the-art 3D medical foundation models on internal (CT-RATE) and external (RAD-ChestCT, MianYang, ReX, COVID-19, MosMed) cohorts in all five tasks.

EXACT builds on our prior work **Chest-OMDL** (MIDL 2025), extending it from organ-specific multi-disease detection into a general-purpose, voxel-level foundation model paradigm for 3D chest CT.

---

## Method Overview

### Pre-training Pipeline

![EXACT Overview](assets/fig1_overview.png)

CT volumes and free-text radiology reports are jointly used to pre-train the Y-Mamba backbone via MIL supervision, producing 18-channel voxel-level Anomaly-aware Maps (AAmap) that support all downstream tasks.

### Multi-disease Diagnosis

![Multi-disease Diagnosis Performance](assets/fig2_multidisease_diagnosis.png)

EXACT achieves state-of-the-art multi-disease classification under both zero-shot and supervised fine-tuning settings across internal (CT-RATE) and external (RAD-ChestCT, MianYang) datasets.

### Anomaly Localization

![Anomaly Localization Results](assets/fig4_anomaly_localization.jpg)

EXACT-Seg generates voxel-level anomaly segmentation masks directly from AAmaps, supporting both zero-shot thresholding and supervised fine-tuning on datasets including ReX, COVID-19, and MosMed.

### CT Report Generation (EXACT-CHAT)

![EXACT-CHAT Overview](assets/fig5_exactchat_overview.png)

EXACT-CHAT is a CT-specific vision-language model that feeds CT volume embeddings produced by the frozen Y-Mamba backbone (pre-trained in Stage 1) into LLaMA-3.1-8B-Instruct via an attentional pooling projector. AAmap-derived per-disease classification results are additionally injected as text in the prompt, and the model generates structured radiology reports conditioned on this combined visual + diagnostic context.

---

## Code Structure

```
EXACT/
├── EXACT_Pretrain/              # Stage 1 – Weakly supervised foundation model pre-training
│   ├── train.py                 # Main training entry
│   ├── engine.py                # Training / evaluation loops
│   ├── utils.py                 # Shared utilities
│   ├── configs/
│   │   └── config_setting.py    # All hyperparameters and paths
│   ├── datasets/
│   │   └── dataset.py           # Dataset & dataloader
│   └── models/
│       └── ymamba/
│           └── ymamba.py        # Y-Mamba model definition
│
├── EXACT_ClassFinetune/         # Stage 2 – Supervised disease classification fine-tuning
│   ├── train_heatmap.py         # Training entry (AAmap → lightweight classifier)
│   ├── engine.py / engine18.py  # Training loops (16- and 18-class variants)
│   ├── configs/
│   ├── datasets/
│   └── models/
│
├── EXACT-Seg/                   # Stage 3 – Anomaly localization & segmentation
│   ├── zero_shot_seg/           # Zero-shot segmentation from AAmap thresholding
│   │   ├── overlay_heatmap.py   # Aggregate disease-specific heatmaps
│   │   └── threshold_overlay.py # Threshold to binary segmentation masks
│   └── supervised_seg/          # Supervised segmentation fine-tuning
│       └── train_supervised.py
│
├── EXACT-CHAT/                  # Stage 4 – CT report generation (vision-language model)
│   ├── llava/                   # Core LLaVA package (CT-adapted)
│   │   ├── model/
│   │   │   ├── multimodal_encoder/
│   │   │   │   └── ct_clip.py           # LLaVA vision-tower placeholder (visual features are pre-computed offline by the frozen Y-Mamba backbone and loaded as .npz)
│   │   │   ├── multimodal_projector/
│   │   │   │   ├── builder.py           # attn_pool + MLP projector
│   │   │   │   └── coca_attentional_pooler.py
│   │   │   └── language_model/
│   │   │       └── llava_llama.py       # LLaMA-3.1 backbone
│   │   ├── train/
│   │   │   ├── train.py                 # Training logic (CT data loading)
│   │   │   ├── train_mem.py             # DeepSpeed entry point
│   │   │   └── llava_trainer.py
│   │   └── serve/                       # Inference & Gradio demo server
│   ├── scripts/
│   │   ├── pretrain.sh                  # Stage 4a – Projector pre-training
│   │   └── finetune_lora.sh             # Stage 4b – LoRA instruction fine-tuning
│   ├── evaluations/
│   │   ├── evaluate_llm.py              # LLM-based clinical accuracy scoring
│   │   ├── multi_metrics.py             # BLEU / METEOR / ROUGE / CIDEr by question type
│   │   └── new_llm_metrics.py           # Multiple-choice accuracy
│   ├── utils/                           # Data preparation & format-conversion scripts
│   ├── zero2.json                       # DeepSpeed ZeRO-2 config
│   └── zero3.json                       # DeepSpeed ZeRO-3 config
│
└── README.md
```

---

## Dataset

### Overview

| Split | Source | # Volumes |
|-------|--------|-----------|
| Pre-training | CT-RATE (training partition) | 24,128 |
| Internal validation / test | CT-RATE (held-out) | 1,564 |
| External test | RAD-ChestCT | 3,630 |
| External test | MianYang | 500 |

### Data Sources

- **CT-RATE** (primary): [https://huggingface.co/datasets/ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
  - 3D chest CT volumes with paired free-text radiology reports.
  - Disease labels are automatically extracted from reports using an LLM.

### Dataset Splits

- Training / Validation / Test split follows the official CT-RATE partitioning.
- Out-of-distribution test sets (RadChest, Mianyang) use no re-training; evaluation is zero-shot.

### Data Preprocessing

**For in-distribution data:**
```bash
python EXACT_Pretrain/data_preprocessed/data_preprocessed.py
```

**For out-of-distribution / external data (two steps):**
```bash
# Step 1: basic preprocessing
python EXACT_Pretrain/data_preprocessed/new_data_preprocessed.py

# Step 2: orientation alignment with the training set
python EXACT_Pretrain/data_preprocessed/flip_data.py
```

---

## Pre-trained Checkpoints

We provide all pre-trained model weights required to reproduce the results in this paper. Download from Google Drive:

**[Download All Checkpoints (~6.3 GB)](https://drive.google.com/drive/folders/1i2J6XUqTm2G8m3-OlbH7Wt00aBxIpClf?usp=sharing)**

After downloading, place the checkpoints under `EXACT/checkpoints/`:

```
EXACT/checkpoints/
├── 01_pretrain/
│   └── ymamba_pretrain_best.pth          # Y-Mamba foundation model (999 MB)
├── 02_classification_finetune/
│   └── classfine_best.pt                 # Supervised classifier head (1.2 GB)
├── 03_segmentation_finetune/
│   ├── seg_covid_best.pth                # COVID-19 segmentation (843 MB)
│   ├── seg_mosmed_best.pth               # MosMed segmentation (843 MB)
│   └── seg_rex_best.pth                  # ReX segmentation (843 MB)
└── 04_exactchat_lora/
    └── checkpoint-38000/                  # EXACT-CHAT LoRA weights (1.7 GB)
        ├── adapter_config.json
        ├── adapter_model.safetensors      # LoRA adapter weights
        ├── non_lora_trainables.bin        # Projector + non-LoRA trainable params
        ├── config.json
        ├── special_tokens_map.json
        ├── tokenizer.json
        └── tokenizer_config.json
```

### Checkpoint–Task Mapping

| Checkpoint | Used For |
|-----------|----------|
| `01_pretrain/ymamba_pretrain_best.pth` | Zero-shot multi-disease diagnosis; Zero-shot segmentation (AAmap thresholding) |
| `02_classification_finetune/classfine_best.pt` | Supervised multi-disease classification |
| `03_segmentation_finetune/seg_covid_best.pth` | Supervised segmentation on COVID-19 dataset |
| `03_segmentation_finetune/seg_mosmed_best.pth` | Supervised segmentation on MosMed dataset |
| `03_segmentation_finetune/seg_rex_best.pth` | Supervised segmentation on ReX dataset |
| `04_exactchat_lora/checkpoint-38000/` | EXACT-CHAT report generation (merge with LLaMA-3.1-8B-Instruct) |

> **Note:** The EXACT-CHAT checkpoint contains only inference-essential files. The base LLM (`meta-llama/Meta-Llama-3.1-8B-Instruct`) must be downloaded separately from [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

---

## Environment Setup

EXACT has **two separate environments** due to conflicting dependency requirements between the foundation model (Mamba-SSM) and the language model (DeepSpeed + PEFT).

---

### Environment 1 – Foundation Model (Pre-training, Classification, Segmentation)

Used for `EXACT_Pretrain`, `EXACT_ClassFinetune`, and `EXACT-Seg`.

**Base image**: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`

```bash
conda create -n exact python=3.10
conda activate exact

# PyTorch with CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install \
    numpy==2.1.2 \
    scipy==1.14.1 \
    nibabel==5.3.2 \
    SimpleITK==2.4.0 \
    medmnist==3.0.2 \
    MedPy==0.5.2 \
    opencv-python==4.10.0.84 \
    scikit-learn==1.5.2 \
    scikit-image==0.24.0 \
    pandas==2.2.3 \
    einops==0.8.0 \
    tqdm==4.66.5 \
    pillow==11.0.0 \
    torchio==0.20.1 \
    joblib==1.4.2 \
    monai==1.3.0 \
    tensorboardX==2.6.2.2 \
    itk==5.4.4.post1

# mamba-ssm requires compilation; must be installed AFTER PyTorch
pip install --no-build-isolation mamba-ssm==2.2.4
```

> **Note:** If you encounter `ModuleNotFoundError` at runtime, install the missing package manually with `pip install <package_name>`.

---

### Environment 2 – Report Generation (EXACT-CHAT)

Used for `EXACT-CHAT` only.

**Base image**: `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`

```bash
conda create -n exact-chat python=3.10
conda activate exact-chat

# PyTorch with CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# LLM / training dependencies
pip install \
    transformers==4.47.0 \
    accelerate==1.6.0 \
    tokenizers==0.21.0 \
    sentencepiece==0.2.0 \
    safetensors==0.4.5 \
    peft==0.15.2 \
    bitsandbytes==0.43.0 \
    deepspeed==0.16.7 \
    huggingface-hub==0.25.2 \
    einops==0.8.0 \
    einops-exts

# Medical imaging dependencies (shared with Env 1)
pip install \
    nibabel==5.3.2 \
    SimpleITK==2.4.0 \
    monai \
    torchio \
    opencv-python-headless \
    joblib

# Install the llava package from the repo root
cd EXACT-CHAT
pip install -e .
```

---

## Training

### Stage 1 – Foundation Model Pre-training (`EXACT_Pretrain`)

Edit `EXACT_Pretrain/configs/config_setting.py` to set your data paths, GPU IDs, batch size, etc.

```bash
conda activate exact
cd EXACT_Pretrain

python train.py
```

Key hyperparameters (in `configs/config_setting.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_data_path` | — | Path to preprocessed training volumes |
| `gpu_id` | `[0,1,2,3]` | GPU indices |
| `batch_size` | 2 | Per-GPU batch size (effective batch 8 across 4 GPUs) |
| `num_workers` | 8 | DataLoader workers |
| `epochs` | 150 | Total pre-training epochs |
| `lr` | 1e-4 | Initial learning rate |
| `work_dir` | — | Output directory for checkpoints & logs |

Checkpoints are saved under `<work_dir>/checkpoints/`. The best checkpoint is `best.pth`.

---

### Stage 2 – Disease Classification Fine-tuning (`EXACT_ClassFinetune`)

The EXACT backbone is **frozen**; only a lightweight classifier head is trained on top of the generated AAmaps.

```bash
conda activate exact
cd EXACT_ClassFinetune

python train_heatmap.py
```

---

### Stage 3a – Zero-shot Segmentation (`EXACT-Seg/zero_shot_seg`)

No training required. Directly apply the pre-trained AAmaps.

```bash
conda activate exact
cd EXACT-Seg/zero_shot_seg

# Step 1: aggregate disease-specific heatmaps
python overlay_heatmap.py \
  --input-root /path/to/work_dir/test_results/prediction_heatmaps/epoch_x \
  --output-dir /path/to/overlaid_heatmaps \
  --res high-res \
  --diseases "Atelectasis,Lung opacity,Consolidation" \
  --aggregate mean

# Step 2: threshold to binary segmentation masks
python threshold_overlay.py \
  --in-overlay /path/to/overlaid_heatmaps \
  --out-seg /path/to/segmentation_results \
  --thresh-mode both \
  --binary-threshold 0.004 \
  --ratio 0.1 \
  --overwrite
```

---

### Stage 3b – Supervised Segmentation Fine-tuning (`EXACT-Seg/supervised_seg`)

```bash
conda activate exact
cd EXACT-Seg/supervised_seg

python train_supervised.py --task train
```

Recommended checkpoint: `<work_dir>/checkpoints/best.pth`

---

### Stage 4a – EXACT-CHAT Projector Pre-training

Pre-trains the cross-modal projector (attentional pooler + MLP) while keeping both the visual encoder and LLM frozen.

```bash
conda activate exact-chat
cd EXACT-CHAT

bash scripts/pretrain.sh
```

Key arguments in `scripts/pretrain.sh`:

```bash
deepspeed --master_port 12438 llava/train/train_mem.py \
    --deepspeed ./zero3.json \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --version plain \
    --data_path /path/to/pretrain_data.json \
    --image_folder /path/to/pretrain_image_folder/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type attn_pool+mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --bf16 True \
    --output_dir ./checkpoints/llava-llama3.1_8B-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 12 \
    --learning_rate 1e-3 \
    --model_max_length 4096
```

> **Note:** `--vision_tower openai/clip-vit-large-patch14-336` is a LLaVA-inherited placeholder argument. The actual visual features fed to the projector are pre-computed `.npz` embeddings produced offline by the frozen Y-Mamba backbone (Stage 1) — no CLIP forward pass is performed. The same applies to Stage 4b.

---

### Stage 4b – EXACT-CHAT LoRA Instruction Fine-tuning

Fine-tunes the LLM with LoRA while unfreezing the projector.

```bash
conda activate exact-chat
cd EXACT-CHAT

bash scripts/finetune_lora.sh
```

Key arguments in `scripts/finetune_lora.sh`:

```bash
deepspeed --master_port 12600 llava/train/train_mem.py \
    --deepspeed ./zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --version llama3_1 \
    --data_path /path/to/train_vqa.json \
    --image_folder /path/to/ct_embeddings/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type "attn_pool+mlp2x_gelu" \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-llama3.1_8B-pretrain/mm_projector.bin \
    --bf16 True \
    --output_dir ./checkpoints/llava-llama3.1_8B-finetune-lora \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-5 \
    --model_max_length 128000 \
    --mm_projector_lr 2e-5
```

| LoRA Parameter | Value |
|---------------|-------|
| `lora_r` | 128 |
| `lora_alpha` | 256 |
| `lora_dropout` | 0.05 |
| Target modules | q/k/v/o/gate/up/down proj |

---

## Inference

### Multi-disease Diagnosis (Zero-shot)

```bash
conda activate exact
cd EXACT_Pretrain

python test.py \
    --resume_model ../checkpoints/01_pretrain/ymamba_pretrain_best.pth \
    --test_data_path /path/to/test_data
```

### Multi-disease Diagnosis (Supervised)

```bash
conda activate exact
cd EXACT_ClassFinetune

python test.py \
    --resume_model ../checkpoints/02_classification_finetune/classfine_best.pt \
    --test_data_path /path/to/test_data
```

### Zero-shot Segmentation

(See Stage 3a training section above – the same two-step pipeline is used for inference.)

### Supervised Segmentation

```bash
conda activate exact
cd EXACT-Seg/supervised_seg

python train_supervised.py \
    --task test \
    --resume_model ../checkpoints/03_segmentation_finetune/seg_covid_best.pth
    # Or: seg_mosmed_best.pth / seg_rex_best.pth depending on the target dataset
```

### Report Generation (EXACT-CHAT)

**Step 1 – Merge LoRA weights into the base model:**

```bash
conda activate exact-chat
cd EXACT-CHAT

python llava/serve/save_merged_model.py \
    --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --lora_model ../checkpoints/04_exactchat_lora/checkpoint-38000 \
    --output_dir ./checkpoints/merged_model
```

**Step 2 – Pre-compute CT volume embeddings with the Y-Mamba backbone:**

CT volumes must first be encoded into `.npz` embeddings using the Stage-1 Y-Mamba backbone. Each `.npz` stores the embedding under key `"arr"`. These embeddings are the visual tokens consumed by EXACT-CHAT.

**Step 3 – Single-sample inference example:**

The snippet below mirrors `llava/serve/ctchat_validation_llama.py` (the script used to produce the reported results) for a single sample.

```python
import torch
import numpy as np
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

disable_torch_init()
model_path = "./checkpoints/merged_model"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base=None, model_name=model_name, device="cuda"
)

# Load pre-computed Y-Mamba CT embedding (.npz with key "arr")
image_path = "/path/to/ct_embeddings/sample.npz"
image = np.load(image_path)["arr"]
image_size = image.size
image_tensor = torch.tensor(image).to(model.device, dtype=torch.float16)

# Build prompt: question + AAmap-derived per-disease classification results (text)
# Replace the 0/1 placeholders below with the actual per-disease predictions
# obtained from EXACT_ClassFinetune (see utils/generate_report_json.py).
disease_str = (
    "Medical material=0; Arterial wall calcification=0; Cardiomegaly=0; "
    "Pericardial effusion=0; Coronary artery wall calcification=0; Hiatal hernia=0; "
    "Lymphadenopathy=0; Emphysema=0; Atelectasis=0; Lung nodule=0; Lung opacity=0; "
    "Pulmonary fibrotic sequela=0; Pleural effusion=0; Mosaic attenuation pattern=0; "
    "Peribronchial thickening=0; Consolidation=0; Bronchiectasis=0; "
    "Interlobular septal thickening=0"
)
question = (
    "<image>\nWrite a radiology report for the following CT scan. "
    f"Known frontend model predictions (disease-wise): {disease_str}.<report_generation>"
)

conv = conv_templates["llama3"].copy()
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        do_sample=False,
        temperature=0.0,
        max_new_tokens=512,
        use_cache=True,
    )
report = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(report)
```

**Step 4 – Multi-GPU batch validation:**

```bash
python llava/serve/ctchat_validation_llama_multigpu.py \
    --model_path ./checkpoints/merged_model \
    --data_path /path/to/valid_vqa.json \
    --image_folder /path/to/ct_embeddings/ \
    --output_file ./output_validation.json
```

> **Note:** Each entry in `valid_vqa.json` should already contain the AAmap-derived per-disease prediction string injected into the human prompt (see [utils/generate_report_json.py](EXACT-CHAT/utils/generate_report_json.py) and [utils/filter_report_with_predictions.py](EXACT-CHAT/utils/filter_report_with_predictions.py) for the data-prep pipeline).

### Evaluation

The full report-generation evaluation pipeline lives in [`EXACT-CHAT/evaluations/`](EXACT-CHAT/evaluations/). It computes the metrics reported in the paper:

- **NLG metrics**: BLEU-1, METEOR, ROUGE-L, CIDEr (with optional 95% CI via bootstrap)
- **Clinical accuracy** (RadBERT-F1 / Precision / Recall): generated reports are passed through RadBertClassifier to predict 18 pathologies, then compared against ground-truth labels
- **CRG (Clinical Report Generation) score**

#### One-time setup — download the RadBERT classifier and tokenizer

```bash
conda activate exact-chat
cd EXACT-CHAT/evaluations

# RadBertClassifier.pth (multi-label disease classifier, from the CT-RATE release)
python download_model.py

# Backbone weights + tokenizer (roberta-base)
python download_RoBERTa_tokenizer.py

# Move the classifier weight into the same folder as the tokenizer
mv RadBertClassifier.pth roberta_local/
```

This populates `./roberta_local/` with the model + tokenizer used by [classifier.py](EXACT-CHAT/evaluations/classifier.py) and [dataset.py](EXACT-CHAT/evaluations/dataset.py).

#### Data layout

The evaluation scripts expect a `BASE_DIR` containing one sub-folder per inference run, each holding:

```
<BASE_DIR>/<run_name>/
├── result_transformat.json   # EXACT-CHAT outputs in evaluation format
├── ground_truth.json         # paired references for NLG metrics
└── ground_truth.csv          # 18-label binary ground truth for classification
```

`<run_name>` is the stem of the prediction JSON (e.g. `predictions_checkpoint38000_…`).

#### Run evaluation

Internal validation — **CT-RATE** (full pipeline: NLG + classification + CRG):

```bash
python evaluation.py \
    --prediction_jsons /path/to/predictions_<run>.json \
    --base_dir         /path/to/ctrate_eval_workdir

# With 95% bootstrap confidence intervals:
python evaluation_95ci.py \
    --prediction_jsons /path/to/predictions_<run>.json \
    --base_dir         /path/to/ctrate_eval_workdir
```

External validation — **RAD-ChestCT** (classification-only, two placeholder classes excluded):

```bash
python evaluation_radchest.py \
    --prediction_jsons /path/to/predictions_<run>.json \
    --base_dir         /path/to/radchest_eval_workdir

python evaluation_radchest_95ci.py \
    --prediction_jsons /path/to/predictions_<run>.json \
    --base_dir         /path/to/radchest_eval_workdir
```

External validation — **MianYang** (full pipeline; `Mosaic attenuation pattern` excluded):

```bash
python evaluation_mianyang.py \
    --prediction_jsons /path/to/predictions_<run>.json \
    --base_dir         /path/to/mianyang_eval_workdir

python evaluation_mianyang_95ci.py \
    --prediction_jsons /path/to/predictions_<run>.json \
    --base_dir         /path/to/mianyang_eval_workdir
```

Each script writes per-run metrics to `<base_dir>/<run_name>/metrics.json` and a cross-run summary to `<base_dir>/evaluation_summary.json`. You can pass multiple JSONs to `--prediction_jsons` to compare several checkpoints in one invocation.

#### Stand-alone metric helpers

If you already have CSV predictions/ground-truth pairs and just want one metric:

```bash
# Multi-label classification (precision / recall / F1 / accuracy)
python calc_scores.py        --pred_csv preds.csv --gt_csv gt.csv --out_json cls.json
python calc_scores_withci.py --pred_csv preds.csv --gt_csv gt.csv --out_json cls_ci.json

# Clinical report generation score
python crg_score.py --pred_csv preds.csv --gt_csv gt.csv --out_json crg.json

# NLG metrics
python nlg_metrics.py        --pred_json preds.json --gt_json gt.json --out_json nlg.json
python nlg_metrics_withci.py --pred_json preds.json --gt_json gt.json --out_json nlg_ci.json
```

---

## Experimental Results

### Multi-disease Classification

**Supplementary Table 4** | Comparison of multi-disease diagnosis performance. Metrics include AUROC, F1 Score, and Accuracy (mean [95% CI] where available). **Red** = best, *blue* = second best, **bold** = third best per column.

#### Internal Validation — CT-RATE Dataset (n = 1,564)

| Models | AUROC | F1 Score | Accuracy |
|--------|-------|----------|----------|
| CT-Net (Fine-tuning) | 0.629 | 0.657 | 0.617 |
| CT-CLIP (Zero-shot) | 0.731 | 0.707 | 0.668 |
| CT-CLIP (VocabFine) | 0.756 | 0.738 | 0.705 |
| CT-CLIP (ClassFine) | 0.756 | 0.724 | 0.689 |
| fVLM (Zero-shot) | 0.778 | 0.751 | 0.718 |
| MedVista3D (Zero-shot) | 0.782 | 0.770 | 0.745 |
| Merlin (Zero-shot) | 0.595 [0.584, 0.606] | 0.687 [0.674, 0.702] | 0.585 [0.573, 0.601] |
| T3D (Zero-shot) | 0.737 | 0.725 | 0.690 |
| T3D (Fine-tuning) | 0.802 | 0.778 | 0.763 |
| RadZero3D (Zero-shot) | 0.762 | 0.742 | 0.701 |
| BIUD (Zero-shot) | 0.713 | 0.716 | 0.681 |
| **EXACT (Zero-shot)** | **0.830 [0.823, 0.837]** | **0.836 [0.826, 0.844]** | **0.768 [0.758, 0.778]** |
| **EXACT (Fine-tuning)** | **0.833 [0.826, 0.840]** | **0.836 [0.829, 0.844]** | **0.769 [0.761, 0.780]** |

> **Note (vs. CT-Net):** AUROC p = 0.001; F1 n.s.; Accuracy n.s.

#### External Validation — RAD-ChestCT Dataset (n = 3,630)

| Models | AUROC | F1 Score | Accuracy |
|--------|-------|----------|----------|
| CT-Net (Fine-tuning) | 0.544 | 0.564 | 0.517 |
| CT-CLIP (Zero-shot) | 0.629 | 0.637 | 0.592 |
| CT-CLIP (VocabFine) | 0.650 | 0.677 | 0.636 |
| CT-CLIP (ClassFine) | 0.643 | 0.644 | 0.599 |
| fVLM (Zero-shot) | 0.644 | 0.663 | 0.619 |
| MedVista3D (Zero-shot) | 0.710 | 0.681 | 0.668 |
| Merlin (Zero-shot) | 0.603 [0.595, 0.610] | 0.657 [0.646, 0.670] | 0.598 [0.584, 0.608] |
| BIUD (Zero-shot) | 0.629 | 0.652 | 0.606 |
| **EXACT (Zero-shot)** | **0.728 [0.722, 0.734]** | **0.731 [0.728, 0.746]** | **0.677 [0.668, 0.704]** |
| **EXACT (Fine-tuning)** | **0.734 [0.728, 0.740]** | **0.737 [0.729, 0.744]** | **0.682 [0.670, 0.686]** |

> **Note (vs. CT-Net):** AUROC p = 0.022; F1 p < 0.001; Accuracy p < 0.001.

#### External Validation — MianYang Dataset (n = 500)

| Models | AUROC | F1 Score | Accuracy |
|--------|-------|----------|----------|
| CT-Net (Fine-tuning) | 0.612 [0.580, 0.645] | 0.618 [0.599, 0.653] | 0.603 [0.572, 0.674] |
| CT-CLIP (Zero-shot) | 0.689 [0.666, 0.709] | 0.746 [0.729, 0.771] | 0.679 [0.657, 0.708] |
| CT-CLIP (VocabFine) | 0.712 [0.687, 0.736] | 0.766 [0.746, 0.788] | 0.695 [0.675, 0.730] |
| CT-CLIP (ClassFine) | 0.704 [0.679, 0.729] | 0.757 [0.737, 0.787] | 0.694 [0.667, 0.731] |
| fVLM (Zero-shot) | 0.716 [0.696, 0.736] | 0.748 [0.725, 0.772] | 0.699 [0.672, 0.730] |
| Merlin (Zero-shot) | 0.602 [0.575, 0.629] | 0.695 [0.657, 0.731] | 0.610 [0.573, 0.658] |
| **EXACT (Zero-shot)** | **0.758 [0.737, 0.779]** | **0.773 [0.738, 0.807]** | **0.734 [0.699, 0.776]** |
| **EXACT (Fine-tuning)** | **0.769 [0.749, 0.788]** | **0.805 [0.780, 0.824]** | **0.761 [0.728, 0.788]** |

> **Note (vs. CT-Net):** AUROC p = 0.005; F1 p < 0.001; Accuracy p < 0.001.

---

### Anomaly Localization & Segmentation

**Supplementary Table 5** | Comparison of anomaly localization performance under zero-shot and supervised fine-tuning settings. DSC = Dice similarity coefficient; HIT = Hit Rate at threshold; AUPR = Area Under the Precision-Recall curve.

#### Task: Zero-shot Anomaly Localization

| Dataset | Models | DSC | HIT@5% | HIT@10% | AUPR | HIT@5% | HIT@10% |
|---------|--------|-----|--------|---------|------|--------|---------|
| ReX-Train (n=1102) | BiomedParse-v2 | 0.012 [0.001, 0.014] | 0.152 | 0.090 | 0.026 [0.024, 0.029] | 0.132 | 0.065 |
| | fVLM | 0.006 [0.005, 0.007] | 0.030 | 0.010 | 0.004 [0.003, 0.004] | 0.011 | 0.002 |
| | CT-CLIP | 0.004 [0.004, 0.005] | 0.000 | 0.000 | 0.002 [0.002, 0.002] | 0.004 | 0.000 |
| | **EXACT** | **0.050 [0.045, 0.055]** | **0.290** | **0.193** | **0.044 [0.039, 0.049]** | **0.231** | **0.153** |
| ReX-Val (n=157) | BiomedParse-v2 | 0.065 [0.051, 0.079] | 0.357 | 0.247 | 0.028 [0.022, 0.033] | 0.377 | 0.223 |
| | fVLM | 0.025 [0.019, 0.031] | 0.141 | 0.054 | 0.024 [0.019, 0.031] | 0.150 | 0.060 |
| | CT-CLIP | 0.005 [0.004, 0.006] | 0.003 | 0.000 | 0.002 [0.002, 0.002] | 0.000 | 0.000 |
| | **EXACT** | **0.071 [0.056, 0.086]** | **0.389** | **0.268** | **0.065 [0.051, 0.079]** | **0.395** | **0.242** |
| COVID-19 (n=20) | BiomedParse-v2 | 0.340 [0.185, 0.490] | 0.550 | 0.500 | 0.459 [0.303, 0.632] | 0.900 | 0.750 |
| | fVLM | 0.081 [0.041, 0.121] | 0.500 | 0.300 | 0.059 [0.035, 0.087] | 0.450 | 0.200 |
| | CT-CLIP | 0.023 [0.010, 0.035] | 0.000 | 0.000 | 0.010 [0.005, 0.016] | 0.000 | 0.000 |
| | **EXACT** | **0.435 [0.348, 0.526]** | **0.950** | **0.850** | **0.530 [0.440, 0.609]** | **0.950** | **0.900** |
| MosMed (n=50) | BiomedParse-v2 | 0.254 [0.196, 0.315] | 0.840 | 0.660 | 0.258 [0.201, 0.321] | 0.820 | 0.600 |
| | fVLM | 0.016 [0.012, 0.020] | 0.060 | 0.000 | 0.007 [0.006, 0.009] | 0.039 | 0.000 |
| | CT-CLIP | 0.004 [0.003, 0.005] | 0.000 | 0.000 | 0.002 [0.001, 0.003] | 0.000 | 0.000 |
| | **EXACT** | **0.363 [0.318, 0.404]** | **0.960** | **0.900** | **0.330 [0.283, 0.376]** | **0.960** | **0.920** |

> **Note (EXACT vs. BiomedParse-v2):** ReX-Train — Dice p < 0.001, AUPR p < 0.001; ReX-Val — Dice p = 0.016, AUPR p < 0.001; COVID-19 — Dice n.s., AUPR n.s.; MosMed — Dice p < 0.001, AUPR p = 0.002.

#### Task: Anomaly Localization with Supervised Fine-tuning

| Dataset | Models | DSC | HIT@5% | HIT@10% | AUPR | HIT@5% | HIT@10% |
|---------|--------|-----|--------|---------|------|--------|---------|
| ReX-Val (n=157) | RWKV-Unet | 0.112 [0.089, 0.135] | 0.312 | 0.242 | 0.180 [0.145, 0.219] | 0.580 | 0.465 |
| | YMamba | 0.198 [0.165, 0.230] | 0.556 | 0.494 | 0.187 [0.154, 0.223] | 0.556 | 0.494 |
| | **EXACT-Seg** | **0.215 [0.182, 0.249]** | **0.643** | **0.580** | **0.200 [0.165, 0.238]** | **0.592** | **0.478** |
| COVID-19 (n=16) | RWKV-Unet | 0.305 [0.205, 0.412] | 0.812 | 0.812 | 0.404 [0.292, 0.513] | 0.875 | 0.812 |
| | YMamba | 0.332 [0.221, 0.450] | 0.812 | 0.750 | 0.358 [0.235, 0.493] | 0.750 | 0.750 |
| | **EXACT-Seg** | **0.476 [0.332, 0.621]** | **0.875** | **0.875** | **0.529 [0.374, 0.679]** | **0.875** | **0.875** |
| MosMed (n=40) | RWKV-Unet | 0.348 [0.290, 0.405] | 0.950 | 0.875 | 0.373 [0.311, 0.438] | 0.950 | 0.850 |
| | YMamba | 0.352 [0.252, 0.378] | 0.850 | 0.850 | 0.324 [0.255, 0.393] | 0.825 | 0.800 |
| | **EXACT-Seg** | **0.454 [0.387, 0.520]** | **0.950** | **0.875** | **0.463 [0.393, 0.536]** | **0.925** | **0.900** |

> **Note (EXACT-Seg vs. RWKV-Unet):** ReX-Val — Dice p = 0.028, AUPR p = 0.007; COVID-19 — Dice p < 0.001, AUPR p = 0.006; MosMed — Dice p < 0.001, AUPR p < 0.001.

---

### Report Generation

**Supplementary Table 6** | Comparison of report generation performance. Metrics include BLEU-1, METEOR, CIDEr, ROUGE-L, and clinical efficacy scores (RadBERT-F1, RadBERT-Precision, RadBERT-Recall).

#### Internal Validation — CT-RATE Dataset (n = 1,564)

| Models | BLEU-1 | METEOR | CIDEr | ROUGE-L | RadBERT-F1 | RadBERT-Prec | RadBERT-Rec |
|--------|--------|--------|-------|---------|-----------|--------------|-------------|
| RadFM | 0.442 | 0.399 | N/A | 0.315 | 0.059 | 0.170 | 0.038 |
| CT2Rep | 0.444 | 0.402 | N/A | 0.310 | 0.160 | 0.435 | 0.128 |
| M3D | 0.436 | 0.400 | N/A | 0.326 | 0.148 | 0.407 | 0.090 |
| CT-CHAT (LLaMA 3.1 70B) | 0.395 | 0.219 | 0.221 | 0.321 | 0.184 | 0.450 | 0.158 |
| CT-CHAT (w/ nodule attrs) | N/A | N/A | N/A | N/A | 0.305 | 0.382 | 0.268 |
| MedVista3D | 0.474 | 0.252 | 0.349 | 0.386 | N/A | N/A | N/A |
| Reg2RG | 0.473 | 0.441 | N/A | 0.367 | 0.253 | 0.423 | 0.181 |
| T3D | 0.501 | N/A | N/A | 0.378 | 0.274 | 0.355 | 0.207 |
| CT-GRAPH | 0.485 | 0.421 | N/A | 0.313 | 0.296 | 0.386 | 0.248 |
| BTB3D | 0.439 | 0.223 | N/A | N/A | 0.258 | 0.260 | 0.260 |
| **EXACT-CHAT** | **0.444 [0.435, 0.453]** | **0.228 [0.223, 0.232]** | **0.139 [0.109, 0.177]** | **0.296 [0.288, 0.304]** | **0.310 [0.274, 0.347]** | **0.410 [0.292, 0.541]** | **0.371 [0.336, 0.408]** |
| **EXACT-CHAT (Refined)** | **0.465 [0.459, 0.471]** | **0.237 [0.234, 0.241]** | **0.077 [0.059, 0.097]** | **0.288 [0.281, 0.296]** | **0.501 [0.457, 0.543]** | **0.414 [0.368, 0.460]** | **0.730 [0.677, 0.780]** |

> **Note (EXACT-CHAT vs. RadFM):** p < 0.001.

#### External Validation — RAD-ChestCT Dataset (n = 3,630)

| Models | BLEU-1 | METEOR | CIDEr | ROUGE-L | RadBERT-F1 | RadBERT-Prec | RadBERT-Rec |
|--------|--------|--------|-------|---------|-----------|--------------|-------------|
| RadFM | N/A | N/A | N/A | N/A | 0.069 | 0.283 | 0.044 |
| CT2Rep | N/A | N/A | N/A | N/A | 0.133 | 0.299 | 0.139 |
| M3D | N/A | N/A | N/A | N/A | 0.113 [0.091, 0.137] | 0.269 [0.213, 0.329] | 0.080 [0.064, 0.097] |
| CT-CHAT (LLaMA 3.1 70B) | N/A | N/A | N/A | N/A | 0.182 | 0.382 | 0.171 |
| Merlin | N/A | N/A | N/A | N/A | 0.182 | 0.271 | 0.149 |
| BTB3D | N/A | N/A | N/A | N/A | 0.266 | 0.272 | 0.329 |
| Reg2RG | N/A | N/A | N/A | N/A | 0.113 [0.093, 0.134] | 0.277 [0.205, 0.354] | 0.082 [0.068, 0.098] |
| Hulu-Med | N/A | N/A | N/A | N/A | 0.279 [0.249, 0.309] | 0.398 [0.355, 0.441] | 0.254 [0.226, 0.283] |
| **EXACT-CHAT** | N/A | N/A | N/A | N/A | **0.289 [0.265, 0.313]** | **0.469 [0.328, 0.546]** | **0.298 [0.275, 0.321]** |
| **EXACT-CHAT (Refined)** | N/A | N/A | N/A | N/A | **0.441 [0.416, 0.467]** | **0.406 [0.380, 0.433]** | **0.610 [0.576, 0.642]** |

> **Note (EXACT-CHAT vs. RadFM):** p < 0.001.

#### External Validation — MianYang Dataset (n = 500)

| Models | BLEU-1 | METEOR | CIDEr | ROUGE-L | RadBERT-F1 | RadBERT-Prec | RadBERT-Rec |
|--------|--------|--------|-------|---------|-----------|--------------|-------------|
| RadFM | 0.000 [0.000, 0.000] | 0.008 [0.008, 0.009] | 0.000 [0.000, 0.000] | 0.019 [0.018, 0.020] | 0.023 [0.009, 0.043] | 0.143 [0.052, 0.188] | 0.046 [0.020, 0.076] |
| M3D | 0.000 [0.000, 0.000] | 0.024 [0.023, 0.026] | 0.000 [0.000, 0.000] | 0.051 [0.049, 0.053] | 0.068 [0.027, 0.118] | 0.162 [0.063, 0.293] | 0.055 [0.019, 0.102] |
| CT-CHAT (LLaMA 3.1 70B) | 0.259 [0.249, 0.268] | 0.184 [0.181, 0.188] | 0.003 [0.001, 0.005] | 0.259 [0.255, 0.264] | 0.073 [0.045, 0.103] | 0.119 [0.086, 0.151] | 0.088 [0.053, 0.128] |
| Merlin | 0.000 [0.000, 0.000] | 0.023 [0.022, 0.023] | 0.000 [0.000, 0.000] | 0.052 [0.051, 0.053] | 0.024 [0.021, 0.028] | 0.074 [0.013, 0.076] | 0.059 [0.057, 0.060] |
| Reg2RG | 0.249 [0.239, 0.259] | 0.161 [0.158, 0.164] | 0.006 [0.003, 0.008] | 0.184 [0.182, 0.187] | 0.086 [0.041, 0.145] | 0.169 [0.077, 0.294] | 0.066 [0.029, 0.118] |
| Hulu-Med | 0.134 [0.119, 0.150] | 0.111 [0.106, 0.117] | 0.003 [0.001, 0.005] | 0.155 [0.148, 0.162] | 0.175 [0.095, 0.265] | 0.265 [0.135, 0.425] | 0.176 [0.080, 0.289] |
| **EXACT-CHAT** | **0.402 [0.393, 0.411]** | **0.214 [0.210, 0.217]** | **0.012 [0.009, 0.017]** | **0.266 [0.262, 0.269]** | **0.290 [0.221, 0.358]** | **0.326 [0.221, 0.436]** | **0.367 [0.307, 0.430]** |
| **EXACT-CHAT (Refined)** | **0.446 [0.438, 0.453]** | **0.227 [0.224, 0.231]** | **0.022 [0.016, 0.028]** | **0.275 [0.272, 0.278]** | **0.410 [0.320, 0.498]** | **0.338 [0.259, 0.422]** | **0.667 [0.546, 0.784]** |

> **Note (EXACT-CHAT vs. RadFM):** p < 0.001.

---

## Visualization

### AAmap Heatmap & Report Grounding

![Report Visualization](assets/ext_fig2_visual_grounding.png)

Each example shows the generated report alongside the AAmap anomaly score bar chart and voxel-level heatmap overlays, demonstrating how EXACT-CHAT grounds pathology findings to specific anatomical regions.

---

## Citation

EXACT extends the following published work. If you find this project useful, please cite:

**Base Paper (MIDL 2025):**

```bibtex
@inproceedings{bai2025chestomdl,
  title     = {Chest-{OMDL}: Organ-specific Multidisease Detection and Localization
               in Chest Computed Tomography using Weakly Supervised Deep Learning
               from Free-text Radiology Report},
  author    = {Xuguang Bai and Mingxuan Liu and Yifei Chen and
               Hongjia Yang and Qiyuan Tian},
  booktitle = {Medical Imaging with Deep Learning},
  year      = {2025},
  url       = {https://openreview.net/forum?id=ns6nq592HX}
}
```

**EXACT (citation will be updated upon publication):**

```bibtex
@article{bai2025exact,
  title   = {EXACT: EXplainable Anomaly-aware ChesT CT Foundation Model},
  author  = {Xuguang Bai and Mingxuan Liu and ...},
  journal = {--},
  year    = {2025}
}
```
