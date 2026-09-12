# EXACT: EXplainable Anomaly-aware ChesT CT Foundation Model

EXACT is an explainable, anomaly-aware vision foundation model for 3D chest CT that
unifies global disease understanding with voxel-level visual grounding in a single
framework.

Unlike CLIP-style 3D foundation models that compress volumetric features into a
global embedding — and lose the spatial information needed for lesion localization —
EXACT learns directly at the voxel level. It is pre-trained on 25,692 paired CT
scans and radiology reports (21,304 patients, CT-RATE) under anatomy-aware weak
supervision, jointly learning organ segmentation and multi-instance anomaly
detection **without any manual voxel-level annotation**. The resulting
**Anomaly-aware Maps (AAmaps)** encode both lesion extent and organ-specific
pathological context, and drive five downstream tasks from one backbone:
zero-shot and fine-tuned multi-disease diagnosis, zero-shot anomaly localization,
supervised segmentation (EXACT-Seg), and grounded report generation (EXACT-CHAT).

📄 **Paper:** *link to be added upon publication*

EXACT extends our earlier [Chest-OMDL](https://openreview.net/forum?id=ns6nq592HX)
(MIDL 2025) from organ-specific multi-disease detection into a general-purpose
voxel-level foundation model.

![EXACT Overview](assets/fig1_overview.png)

---

## Repository layout

```
EXACT/
├── EXACT_Pretrain/                 # Stage 1 – weakly supervised pre-training
│   ├── data_preprocessed/          # CT → HDF5 preprocessing (run this first)
│   │   ├── data_preprocessed.py        # internal cohort: RadGenome CT + organ masks
│   │   ├── preprocess_ctrate.py        # raw CT-RATE → RadGenome grid (optional)
│   │   ├── new_data_preprocessed.py    # external cohorts: CT only
│   │   ├── flip_data.py                # orientation alignment
│   │   └── save_label_18.py            # attach disease labels to the store
│   ├── train.py                    # pre-training entry
│   ├── test.py                     # multi-disease diagnosis evaluation
│   ├── engine.py                   # train / eval loops, Abnormal_loss (MIL pooling)
│   ├── configs/config_setting.py   # hyperparameters and paths
│   ├── datasets/dataset.py
│   └── models/ymamba/ymamba.py     # Y-Mamba backbone
│
├── EXACT_ClassFinetune/            # Stage 2 – supervised diagnosis fine-tuning
│   ├── train_heatmap.py            # train and test entry (--task train|test)
│   ├── models/ymamba/heatmap.py    # lightweight classifier over frozen AAmaps
│   └── utils/disease_predictions_medics.py   # threshold + per-disease metrics
│
├── EXACT-Seg/                      # Stage 3 – anomaly localization
│   ├── zero_shot_seg/
│   │   ├── train_mamba.py          # export AAmaps
│   │   ├── datasets/resize.py      # build lesion masks on the model grid
│   │   └── evaluation/
│   │       ├── overlay_heatmap.py      # aggregate disease-specific heatmaps
│   │       ├── threshold_overlay.py    # threshold to binary masks
│   │       └── calc_dice.py / calc_aupr.py
│   └── supervised_seg/
│       ├── train_supervised.py     # --task train|test
│       └── evaluation/
│
├── EXACT-CHAT/                     # Stage 4 – grounded report generation
│   ├── llava/                      # CT-adapted LLaVA package
│   │   ├── model/multimodal_projector/   # attn_pool + MLP
│   │   └── serve/ctchat_validation_llama.py   # inference entry
│   ├── scripts/{pretrain.sh, finetune_lora.sh}
│   ├── evaluations/                # NLG, RadBERT clinical accuracy, CRG
│   └── utils/                      # data preparation
│
└── checkpoints/                    # released weights (download separately)
```

---

## Environment

Two environments are needed: `mamba-ssm` and the DeepSpeed/PEFT stack have
conflicting requirements. Both build on `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`.

**Environment 1 — foundation model** (`EXACT_Pretrain`, `EXACT_ClassFinetune`, `EXACT-Seg`)

```bash
conda create -n exact python=3.10 && conda activate exact
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-exact.txt
# mamba-ssm compiles against the installed torch, so it must come last
pip install --no-build-isolation mamba-ssm==2.2.4
```

**Environment 2 — report generation** (`EXACT-CHAT`)

```bash
conda create -n exact-chat python=3.10 && conda activate exact-chat
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-exactchat.txt
cd EXACT-CHAT && pip install -e .
```

---

## Checkpoints

**[Download all checkpoints (~6.3 GB)](https://drive.google.com/drive/folders/1i2J6XUqTm2G8m3-OlbH7Wt00aBxIpClf?usp=sharing)**

Unpack under `checkpoints/`:

| Path | Used for |
|---|---|
| `01_pretrain/ymamba_pretrain_best.pth` | Zero-shot diagnosis; AAmap export; zero-shot localization |
| `02_classification_finetune/classfine_best.pt` | Fine-tuned multi-disease diagnosis |
| `03_segmentation_finetune/seg_{rex,covid,mosmed}_best.pth` | Supervised segmentation |
| `04_exactchat_lora/checkpoint-38000/` | EXACT-CHAT report generation |

> **Do not rename the EXACT-CHAT checkpoint directories.** LLaVA derives the model
> name from the last two path components, and the LoRA loading branch is selected by
> the substring `lora` in that name. `04_exactchat_lora/checkpoint-38000` resolves to
> `04_exactchat_lora_checkpoint-38000` and loads correctly; a renamed directory
> silently falls through to a non-LoRA branch and the adapter is never applied.

The base LLM (`meta-llama/Llama-3.1-8B-Instruct`) must be downloaded separately from
[HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

---

## Data

| Split | Source | Volumes |
|---|---|---|
| Pre-training | CT-RATE (train) | 24,128 |
| Internal validation | CT-RATE (held-out) | 1,564 |
| External test | RAD-ChestCT | 3,629 |
| External test | MianYang | 500 |

- **CT-RATE**: <https://huggingface.co/datasets/ibrahimhamamci/CT-RATE> — CT volumes
  with paired free-text reports; disease labels extracted from the reports by an LLM.
- **Organ masks and windowed volumes** come from the
  [RadGenome-Chest CT](https://huggingface.co/datasets/RadGenome/RadGenome-Chest-CT)
  release, which ships nine per-region masks alongside the preprocessed CT. No
  separate segmentation model needs to be run.
- External cohorts are evaluated **without any re-training**.

### Preprocessing

All stages consume a single HDF5 store: one group per study, holding
`ct` `(1, 64, 128, 128)` float32, `mask` `(9, 64, 128, 128)` bool (internal cohort
only), and a `label_18` / `label_16` vector.

**Internal cohort (CT-RATE + RadGenome masks):**

`data_preprocessed.py` reads the **RadGenome-Chest CT** release, not raw CT-RATE.
It expects the `*_preprocessed` volumes, which are already resampled to 1 mm in
plane and 3 mm through plane, and the nine per-region masks that ship beside
them. It does not resample to a target spacing itself, so passing the raw
CT-RATE `*_fixed` volumes silently produces a differently-scaled store.

```bash
cd EXACT_Pretrain/data_preprocessed

python data_preprocessed.py \
    --ct-dir   /path/to/valid_preprocessed \
    --mask-dir /path/to/valid_region_mask \
    --output   valid_total_processed_data.h5

python save_label_18.py --csv labels.csv --h5 valid_total_processed_data.h5
```

Masks are needed for pre-training and for the segmentation branch. Evaluation
does not use them — `test.py` reads `ct` only — so `--mask-dir` may be omitted to
build a CT-only store for zero-shot inference on this cohort.

**Starting from raw CT-RATE instead:** if the RadGenome release is unavailable,
`preprocess_ctrate.py` re-grids the raw volumes onto the same 1 mm / 3 mm grid
first:

```bash
python preprocess_ctrate.py \
    --input        /path/to/valid_fixed \
    --output       /path/to/valid_preprocessed \
    --metadata-csv ../EXACT-CHAT/llava/train/validation_metadata.csv \
    --num-workers  4
```

The metadata table shipped at `EXACT-CHAT/llava/train/validation_metadata.csv` covers
the CT-RATE validation split; `--metadata-csv` is optional, and without it the
spacing is read from each volume's header.

This reproduces the RadGenome grid for most studies but not all: on a 100 study
sample, 71 matched the released shape exactly and agreed to 9e-06 relative
error, while the rest differed along z, because the slice thickness recorded in
the `*_fixed` release — and in the metadata, which repeats it — is not the one
RadGenome resampled from. The difference is largely absorbed by the fixed
`(64, 128, 128)` output grid: zero-shot classification over those 100 studies
scored AUROC 0.8377 against 0.8382 from the RadGenome volumes. Prefer the
RadGenome release where you can get it. Note also that this script produces CT
only — the organ masks come from RadGenome and cannot be regenerated here, so a
store built this way supports evaluation but not pre-training.

**External cohorts (RAD-ChestCT, MianYang):**

```bash
# 1. resample and normalise (.mha and .nii/.nii.gz both supported)
python new_data_preprocessed.py \
    --input  /path/to/volumes \
    --output radchest_processed.h5 \
    --visualize_dir preview --num_processes 4

# 2. check preview/, and align the orientation if it differs from the training set
python flip_data.py --input radchest_processed.h5 --preview --axis 1
python flip_data.py --input radchest_processed.h5 \
                    --output radchest_flipped.h5 --axis 1

# 3. attach labels (16 columns become label_16, expanded to 18 at load time)
python save_label_18.py --csv radchest_labels.csv --h5 radchest_flipped.h5
```

Intensity normalisation is **adaptive**, not a fixed HU window: each volume is
clipped to its own 0.5 / 99.5 percentiles, then CLAHE is applied slice-wise.
Intensities are comparable within a volume but not across volumes.

> The `.mha` and `.nii` readers in `new_data_preprocessed.py` apply different flips
> (two axes vs. one), reflecting how the source cohorts are stored. Always inspect
> the preview images before trusting an orientation.

---

## Reproducing the results

### 1. Multi-disease diagnosis — zero-shot

```bash
conda activate exact
cd EXACT_Pretrain

python test.py \
    --h5         /path/to/valid_total_processed_data.h5 \
    --checkpoint ../checkpoints/01_pretrain/ymamba_pretrain_best.pth \
    --output-dir results/ct_rate
```

Writes `disease_metrics.csv` (per finding), `predictions.csv` and `summary.json`.
The same command evaluates the external cohorts — organ masks are not read, so a
store without a `mask` entry works unchanged.

**Read the [Metric conventions](#metric-conventions) section before comparing
against the published F1 and accuracy.**

### 2. Multi-disease diagnosis — fine-tuned

The backbone is frozen; a lightweight classifier is trained over the exported
AAmaps. First export them with `--export-aamaps` (this writes one directory per
study, containing one NIfTI per finding masked by its organ):

```bash
cd EXACT_Pretrain
python test.py \
    --h5            /path/to/valid_total_processed_data.h5 \
    --checkpoint    ../checkpoints/01_pretrain/ymamba_pretrain_best.pth \
    --output-dir    results/ct_rate \
    --export-aamaps results/ct_rate
```

Then score the frozen classifier over them:

```bash
conda activate exact
cd EXACT_ClassFinetune

python train_heatmap.py \
    --task           test \
    --heatmap-root   ../EXACT_Pretrain/results/ct_rate/prediction_heatmaps/epoch_N \
    --h5-path        /path/to/valid_total_processed_data.h5 \
    --ymamba-ckpt    ../checkpoints/01_pretrain/ymamba_pretrain_best.pth \
    --resume-weights ../checkpoints/02_classification_finetune/classfine_best.pt \
    --save-csv       results/pred.csv \
    --save-probs     results/prob.npz

# per-disease thresholds and metrics from the prediction CSV
python utils/disease_predictions_medics.py --pred-csv results/pred.csv --positive-class absent
```

Training uses the same script with `--task train`.
`disease_predictions_medics.py` always fits its thresholds on the set it scores,
and `--positive-class` selects the F1 convention exactly as in `test.py`
(it defaults to `present`; the manuscript reports `absent`) — see
[Metric conventions](#metric-conventions).

### 3. Zero-shot anomaly localization

No training required — threshold the AAmaps that Stage 1 already exported.

**Ground truth.** ReXGroundingCT ships one mask per finding per study, on the
original CT grid. Merge them into a single binary lesion mask at the resolution
the model predicts on:

```bash
python EXACT-Seg/zero_shot_seg/datasets/resize.py \
    --input  /path/to/ReXGroundingCT/segmentations \
    --output /path/to/lesion_masks
```

This is not just a resize. The CT volumes and the ReXGroundingCT annotations are
stored with opposite in-plane orientation, and both are read with `get_fdata()`,
which ignores the affine — so the masks need an in-plane 180° flip to line up
with the AAmaps. `resize.py` applies it. Skipping it leaves Dice at chance level
rather than producing an obviously broken result, so it is worth checking that
your masks come out of this script and not out of a hand-rolled resample.

**Evaluation.** The disease subset, the aggregation and the threshold are all
cohort-specific; the values below are the ones used in the manuscript.

```bash
conda activate exact
cd EXACT-Seg/zero_shot_seg/evaluation

python overlay_heatmap.py \
    --input-root /path/to/results/ct_rate/prediction_heatmaps/epoch_N \
    --output-dir /path/to/overlaid_heatmaps \
    --res low-res \
    --diseases "Pleural effusion,Bronchiectasis,Peribronchial thickening,Interlobular septal thickening,Atelectasis,Lung opacity,Consolidation,Mosaic attenuation pattern,Pulmonary fibrotic sequela" \
    --aggregate sum

python threshold_overlay.py \
    --in-overlay      /path/to/overlaid_heatmaps \
    --out-seg         /path/to/segmentation_results \
    --thresh-mode     abs \
    --binary-threshold 0.20 \
    --overwrite

python calc_dice.py \
    --pred_dir /path/to/segmentation_results \
    --gt_dir   /path/to/lesion_masks

python calc_aupr.py \
    --pred_dir /path/to/overlaid_heatmaps \
    --gt_dir   /path/to/lesion_masks
```

| Cohort | `--diseases` | `--binary-threshold` |
|---|---|---|
| ReXGroundingCT | the 9 findings above | `0.20` |
| MosMedData / COVID-19-CT-Seg | `"Atelectasis,Lung opacity,Mosaic attenuation pattern,Consolidation"` | `0.15` |

Both use `--res low-res --aggregate sum`. The low-resolution AAmap is the one to
threshold: it is the map the classification head actually attends over, and the
high-resolution upsampling smears mass into the lesion boundary, which inflates
Dice by roughly 40% on ReXGroundingCT.

Dice is computed on the thresholded masks, AUPR on the continuous overlays — so
`calc_aupr.py` reads the output of `overlay_heatmap.py`, not of
`threshold_overlay.py`.

### 4. Supervised segmentation (EXACT-Seg)

```bash
conda activate exact
cd EXACT-Seg/supervised_seg

python train_supervised.py --task test \
    --test-data /path/to/seg_test.h5 \
    --resume_model ../../checkpoints/03_segmentation_finetune/seg_rex_best.pth
# swap in seg_covid_best.pth / seg_mosmed_best.pth for the other cohorts
```

Each checkpoint is fine-tuned on its own cohort, so `--test-data` has to match
the weights you load. The lesion head is wired differently across cohorts --
`seg_rex_best.pth` reads the 18-channel abnormality branch while the COVID and
MosMed checkpoints read the 7-channel organ branch -- and the script picks the
right one by inspecting the checkpoint, so there is nothing to configure.

ReXGroundingCT keeps its masks outside the HDF5 store; for that cohort point
`--mask-dir` at the output of `resize.py`. Paths can also be set in
`configs/config_setting.py` instead of on the command line, and
`zero_shot_seg/configs/config_setting.py` works the same way.

Training uses `--task train` with `--train-data`. Streaming metrics to Weights &
Biases or SwanLab is opt-in via `--track`; without it no account, API key or
network access is needed.

### 5. Report generation (EXACT-CHAT)

**Step 1 — encode CT volumes.** EXACT-CHAT consumes pre-computed visual tokens, not
raw CT. `llava/serve/encode_script.py` runs the frozen Y-Mamba backbone and writes
one `.npz` per study with the embedding under key `"arr"`. The
`--vision_tower openai/clip-vit-large-patch14-336` argument in the training scripts
is an inherited LLaVA placeholder; no CLIP forward pass ever happens.

**Step 2 — generate.** The LoRA adapter is applied at load time, so **no weight
merging step is needed**: pass the adapter as `--model-path` and the base LLM as
`--model-base`.

```bash
conda activate exact-chat
cd EXACT-CHAT

python -m llava.serve.ctchat_validation_llama \
    --model-path     ../checkpoints/04_exactchat_lora/checkpoint-38000 \
    --model-base     meta-llama/Llama-3.1-8B-Instruct \
    --eval-json      /path/to/report_generation.json \
    --encoding-dir   /path/to/encodings \
    --output         preds.json \
    --conv-mode      llama3 \
    --temperature    0.0 \
    --max-new-tokens 1024
```

Invoke it with `python -m` as shown. Running the file by path
(`python llava/serve/ctchat_validation_llama.py`) puts `llava/serve/` on
`sys.path` instead of the repository root and fails on `import llava`.

`--eval-json` is a conversation JSON: one entry per study with an `image` field
and a `conversations` list. `--encoding-dir` holds the matching `.npz` files from
Step 1; the `image` field is looked up there with its extension replaced by
`.npz`. Predictions are written as `{"image": ..., "conversations_out": [...]}`,
with the Llama-3 `<|eot_id|>` marker left in place — the evaluation scripts strip
it.

Each prompt embeds the AAmap-derived per-disease predictions as text; see
[`utils/generate_report_json.py`](EXACT-CHAT/utils/generate_report_json.py) and
[`utils/filter_report_with_predictions.py`](EXACT-CHAT/utils/filter_report_with_predictions.py)
for how the evaluation JSON is assembled.

**Step 3 — evaluate.** One-time setup for the RadBERT clinical-accuracy classifier:

```bash
cd EXACT-CHAT/evaluations
python download_model.py                     # RadBertClassifier.pth
python download_RoBERTa_tokenizer.py         # backbone + tokenizer
mv models/RadBertClassifier.pth roberta_local/
```

Only these downloads need network access. The evaluation itself runs offline:
`classifier.py` and `dataset.py` set `TRANSFORMERS_OFFLINE=1` and read the
backbone and tokenizer from `roberta_local/`.

The evaluation scripts do not read an inference output directly. For each
prediction file they expect a folder under `--base_dir`, named after that file's
stem, holding `result_transformat.json`, `ground_truth.json` and
`ground_truth.csv`. Build them with `prepare_eval_files.py`, passing the cohort's
inference input as the reference — its `gpt` turns are the ground-truth reports:

```bash
python prepare_eval_files.py \
    --prediction_jsons preds.json \
    --base_dir         ctrate_workdir \
    --reference_json   /path/to/report_generation.json \
    --label_csv        /path/to/valid_predicted_labels.csv
```

Then score. Use the same `--base_dir` as above:

```bash
python evaluation.py --prediction_jsons preds.json --base_dir ctrate_workdir
python evaluation_radchest.py  --prediction_jsons preds.json --base_dir radchest_workdir
python evaluation_mianyang.py  --prediction_jsons preds.json --base_dir mianyang_workdir
# append _95ci to any of the above for bootstrap confidence intervals
```

`evaluation.py` and `evaluation_mianyang.py` report classification, CRG and NLG
metrics; `evaluation_radchest.py` is classification-only and drops the two
abnormality columns RAD-ChestCT does not annotate. Results land in
`metrics.json` inside the run directory.

Stand-alone metrics: `calc_scores.py` (classification), `nlg_metrics.py`
(BLEU / METEOR / ROUGE-L / CIDEr), `crg_score.py`.

---

## Training from scratch

Reproducing the released checkpoints, rather than evaluating them. Paths and
hyperparameters live in each module's config file or shell script; edit those
before launching.

```bash
conda activate exact

# Stage 1 - foundation model pre-training (paths in configs/config_setting.py)
python EXACT_Pretrain/train.py

# Stage 2 - AAmap classifier, backbone frozen
python EXACT_ClassFinetune/train_heatmap.py \
    --task         train \
    --heatmap-root /path/to/train_heatmaps \
    --h5-path      /path/to/train_processed_data.h5 \
    --ymamba-ckpt  checkpoints/01_pretrain/ymamba_pretrain_best.pth

# Stage 3 - supervised segmentation
python EXACT-Seg/supervised_seg/train_supervised.py --task train

# Stage 4 - EXACT-CHAT: projector pre-training, then LoRA instruction tuning
conda activate exact-chat && cd EXACT-CHAT
bash scripts/pretrain.sh
bash scripts/finetune_lora.sh
```

Zero-shot anomaly localization (Stage 3a) requires no training — it thresholds
the AAmaps directly, as shown above.

The Stage-4 LoRA configuration is `r=128`, `alpha=256`, `dropout=0.05`; the
adapted modules are discovered automatically by `find_all_linear_names`, which
selects every linear layer except the multimodal projector and the output head.

---

## Metric conventions

Two choices in the evaluation materially affect the reported **F1 and accuracy**.
AUROC is invariant to both and reproduces directly. `EXACT_Pretrain/test.py` exposes
them as flags rather than hard-coding them, so published numbers can be reproduced
and audited:

| Flag | Default | Effect |
|---|---|---|
| `--threshold-source` | `checkpoint` | `checkpoint` uses per-disease thresholds fitted on the validation split. `fit-on-test` refits them on the test set, which is **optimistically biased**. |
| `--positive-class` | `present` | `present` scores disease presence as the positive class (the usual detection convention). `absent` scores disease *absence* as positive; since these cohorts are negative-dominated, this raises F1 substantially. |

The tables in the manuscript were produced with `fit-on-test` + `absent`, available
as a single flag:

```bash
python test.py --h5 ... --checkpoint ... --reproduce-paper
```

The choice of positive class is what separates the published F1 from the one the
usual detection convention gives. Flipping the positive class swaps TP with TN and
FP with FN, so AUROC and accuracy are unchanged and only F1 moves; the threshold
source moves accuracy as well. Running the command above on the CT-RATE held-out
split (1,564 studies, all 18 findings) gives:

| Convention | AUROC | F1 | Accuracy |
|---|---|---|---|
| `--reproduce-paper` (`fit-on-test` + `absent`) | 0.830 | 0.835 | 0.767 |
| defaults (`checkpoint` + `present`) | 0.830 | 0.516 | 0.746 |
| *published, EXACT (Zero-shot) on CT-RATE* | *0.830* | *0.836* | *0.768* |

We report both conventions so the difference is explicit rather than buried.

These are the **zero-shot** numbers, i.e. the output of `EXACT_Pretrain/test.py`.
The Stage 2 classifier (§2 above) is scored by `disease_predictions_medics.py`,
which always fits its thresholds on the set it scores and takes the same
`--positive-class` flag. On the same split:

| Convention | AUROC | F1 | Accuracy |
|---|---|---|---|
| `--positive-class absent` | 0.834 | 0.833 | 0.766 |
| `--positive-class present` (default) | 0.834 | 0.536 | 0.766 |
| *published, EXACT (Fine-tuning) on CT-RATE* | *0.833* | *0.836* | *0.769* |

**Macro-average denominators differ per cohort.** RAD-ChestCT and MianYang annotate
16 of the 18 findings; the two unannotated channels are zero-filled at load time and
carry no ground truth. Any finding with only one class present in a cohort is
excluded from the macro average, so the denominator is 18 (CT-RATE), 16
(RAD-ChestCT) and 17 (MianYang). Concretely, *Coronary artery wall calcification*
and *Mosaic attenuation pattern* are single-class in RAD-ChestCT, and *Mosaic
attenuation pattern* is single-class in MianYang. `test.py` prints
`n_diseases_averaged` and lists the excluded findings in `summary.json` rather than
averaging silently.

---

## Citation

**Chest-OMDL (MIDL 2025):**

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

**EXACT** — citation will be updated upon publication.
