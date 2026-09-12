#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXACT - supervised multi-disease diagnosis over frozen AAmaps.

The pre-trained Y-Mamba backbone is kept frozen; a lightweight classifier is
trained on the Anomaly-aware Maps (AAmaps) exported by ``EXACT_Pretrain/test.py
--export-aamaps``. Each of the 18 findings gets its own single-channel binary
head, so a finding is scored from its own anomaly map rather than from a shared
representation.

Both training and evaluation live in this file:

    # evaluate a released classifier
    python train_heatmap.py --task test \
        --heatmap-root   ../EXACT_Pretrain/results/ct_rate/prediction_heatmaps/epoch_N \
        --h5-path        /path/to/valid_total_processed_data.h5 \
        --ymamba-ckpt    ../checkpoints/01_pretrain/ymamba_pretrain_best.pth \
        --resume-weights ../checkpoints/02_classification_finetune/classfine_best.pt \
        --save-csv       results/pred.csv

    # train from scratch
    python train_heatmap.py --task train \
        --heatmap-root /path/to/train_heatmaps \
        --h5-path      /path/to/train_processed_data.h5 \
        --ymamba-ckpt  ../checkpoints/01_pretrain/ymamba_pretrain_best.pth

``--task test`` prints per-class metrics at a fixed threshold of 0.5. The
published numbers instead fit a per-disease threshold on the prediction CSV;
run ``utils/disease_predictions_medics.py`` on ``--save-csv`` to reproduce them,
and see the "Metric conventions" section of the top-level README for what that
choice implies.
"""

import os
import math
import json
import time
import argparse
import warnings
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import csv
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.ymamba.heatmap import LightweightHeatmapEncoder
from configs.config_setting import setting_config
from datasets.dataset import HeatmapDataset

warnings.filterwarnings("ignore")


# --------------------------
# Name utilities
# --------------------------
def get_disease_names():
    return [
        "Medical_material", "Arterial_wall_calcification",
        "Cardiomegaly", "Pericardial_effusion", "Coronary_artery_wall_calcification",
        "Hiatal_hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung_nodule",
        "Lung_opacity", "Pulmonary_fibrotic_sequela", "Pleural_effusion",
        "Mosaic_attenuation_pattern", "Peribronchial_thickening", "Consolidation",
        "Bronchiectasis", "Interlobular_septal_thickening"
    ]

def get_disease_display_names():
    return [
        "Medical material", "Arterial wall calcification",
        "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
        "Bronchiectasis", "Interlobular septal thickening"
    ]


# --------------------------
# Model
# --------------------------
class HeatmapMultiLabelClassifier(nn.Module):
    """One independent binary head per finding, each over a single AAmap channel.

    The heads do not share weights: channel *i* of the input is the AAmap for
    finding *i*, and only head *i* ever sees it. This keeps a finding's score
    tied to its own anomaly map, at the cost of 18 encoder copies.
    """

    def __init__(self, ymamba_ckpt: str, freeze_encoder: bool = True, dropout_prob: float = 0.3,
                 num_classes: int = 18):
        super().__init__()
        self.num_classes = num_classes

        class SingleChannelBinary(nn.Module):
            def __init__(self, ckpt, freeze, dropout_p):
                super().__init__()
                # Lightweight encoder: the first three Y-Mamba stages, single-channel input.
                self.encoder = LightweightHeatmapEncoder(
                    ymamba_checkpoint_path=ckpt,
                    freeze=freeze
                )
                self.dropout = nn.Dropout(dropout_p)
                self.fc = nn.Linear(768, 1)

            def forward(self, x: torch.Tensor):
                # x: [B,1,D,H,W]
                feats = self.encoder(x)            # [B,16,32,32,768]
                feats = feats.mean(dim=(1, 2, 3))  # [B,768]
                feats = self.dropout(feats)
                return self.fc(feats)              # [B,1]

        self.heads = nn.ModuleList([
            SingleChannelBinary(ymamba_ckpt, freeze_encoder, dropout_prob)
            for _ in range(self.num_classes)
        ])

    def forward(self, images: torch.Tensor, *args, **kwargs):
        """images: [B, 18, D, H, W] -> logits [B, 18]."""
        assert images.size(1) == self.num_classes, \
            f"Expected {self.num_classes} channels, got {images.size(1)}"
        logits = [self.heads[i](images[:, i:i + 1, ...]) for i in range(self.num_classes)]
        return torch.cat(logits, dim=1)  # [B,18]


# --------------------------
# Metrics
# --------------------------
@torch.no_grad()
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(np.int32)
    C = y_true.shape[1] if y_true.size else 18
    per_class = { 'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [] }
    for c in range(C):
        if y_true.size == 0:
            per_class['auc'].append(float('nan'))
            per_class['accuracy'].append(0.0)
            per_class['precision'].append(0.0)
            per_class['recall'].append(0.0)
            per_class['f1'].append(0.0)
            continue
        yt = y_true[:, c]
        yp = y_pred[:, c]
        yp_prob = y_prob[:, c]
        try:
            auc = roc_auc_score(yt, yp_prob)
        except Exception:
            # Undefined when the cohort has only one class for this finding.
            auc = float('nan')
        acc  = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec  = recall_score(yt, yp, zero_division=0)
        f1   = f1_score(yt, yp, zero_division=0)
        per_class['auc'].append(auc)
        per_class['accuracy'].append(acc)
        per_class['precision'].append(prec)
        per_class['recall'].append(rec)
        per_class['f1'].append(f1)

    def nanmean(x):
        x = np.array(x, dtype=np.float64)
        return float(np.nanmean(x)) if np.isnan(x).any() else float(np.mean(x))
    macro = {k: nanmean(v) for k, v in per_class.items()}
    return per_class, macro


# --------------------------
# Train / validation loops
# --------------------------
def _tqdm_total_steps(loader: DataLoader, max_samples: int):
    dataset_size = len(loader.dataset)
    sample_total = max_samples if max_samples and max_samples > 0 else dataset_size
    return max(1, math.ceil(sample_total / loader.batch_size))

def train_one_epoch(model, loader, device, criterion, optimizer, scheduler,
                    start_step: int, max_train_samples_per_epoch=0, epoch=1,
                    grad_clip=1.0, save_every=0, ckpt_dir: Path = None, model_to_save: nn.Module = None,
                    ):
    model.train()
    epoch_loss = 0.0
    processed = 0
    global_step = start_step
    total_steps = _tqdm_total_steps(loader, max_train_samples_per_epoch)

    with tqdm(total=total_steps, desc=f"Train Epoch {epoch}", leave=False, dynamic_ncols=True) as pbar:
        # HeatmapDataset yields (images_18ch, labels_18, sample_name)
        for local_step, (images, labels, sample_names) in enumerate(loader, start=1):
            iter_start = time.time()
            images = images.to(device, non_blocking=True)                         # [B, 18, D, H, W]
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)    # [B, 18]

            logits = model(images)          # [B,18]
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            scheduler(global_step)
            global_step += 1

            bsz = images.size(0)
            epoch_loss += loss.item() * bsz
            processed += bsz

            avg_loss = epoch_loss / max(processed, 1)
            batch_time = time.time() - iter_start
            pbar.set_postfix({"avg_loss": f"{avg_loss:.5f}", "bt(s)": f"{batch_time:.3f}"})
            pbar.update(1)

            if save_every and ckpt_dir is not None and model_to_save is not None:
                if global_step % save_every == 0:
                    ep_tag = f"epoch_{epoch}"
                    step_tag = f"step_{global_step}"
                    torch.save(model_to_save.state_dict(), ckpt_dir / f'checkpoint_{step_tag}_{ep_tag}.pt')
                    torch.save(optimizer.state_dict(),  ckpt_dir / f'optim_{step_tag}_{ep_tag}.pt')
                    print(f"Saved step checkpoint at {ckpt_dir}/checkpoint_{step_tag}_{ep_tag}.pt")

            if max_train_samples_per_epoch and processed >= max_train_samples_per_epoch:
                break

    avg_epoch_loss = epoch_loss / max(processed, 1)
    return avg_epoch_loss, processed, global_step


@torch.no_grad()
def evaluate(model, loader, device, criterion, max_val_samples_per_epoch=0, epoch=1):
    model.eval()
    y_true_list, y_prob_list, names = [], [], []
    processed = 0
    val_loss_sum = 0.0
    total_steps = _tqdm_total_steps(loader, max_val_samples_per_epoch)

    with tqdm(total=total_steps, desc=f"Valid Epoch {epoch}", leave=False, dynamic_ncols=True) as pbar:
        for images, labels, sample_names in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            loss  = criterion(logits, labels).item()

            val_loss_sum += loss * images.size(0)
            y_prob_list.append(probs)
            y_true_list.append(labels.cpu().numpy())
            names.extend(sample_names)

            processed += images.size(0)
            avg_val_loss = val_loss_sum / max(processed, 1)
            pbar.set_postfix({"avg_val_loss": f"{avg_val_loss:.5f}"})
            pbar.update(1)

            if max_val_samples_per_epoch and processed >= max_val_samples_per_epoch:
                break

    y_prob = np.concatenate(y_prob_list, axis=0) if y_prob_list else np.zeros((0, 18), dtype=np.float32)
    y_true = np.concatenate(y_true_list, axis=0).astype(int) if y_true_list else np.zeros((0, 18), dtype=int)
    avg_val_loss = val_loss_sum / max(processed, 1)
    return y_true, y_prob, names, processed, avg_val_loss


# --------------------------
# Test-time output
# --------------------------
def save_predictions_csv(csv_path: Path, names, y_prob: np.ndarray, y_true: np.ndarray, class_names=None):
    """Write the Pred_/GT_ column layout that disease_predictions_medics.py reads."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    C = y_prob.shape[1]
    if class_names is None or len(class_names) != C:
        class_names = get_disease_display_names()[:C]
    header = (["VolumeName"]
              + [f"Pred_{c}" for c in class_names]
              + [f"GT_{c}" for c in class_names])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, name in enumerate(names):
            row = [name] + list(map(float, y_prob[i].tolist())) + list(map(int, y_true[i].tolist()))
            w.writerow(row)

def save_prob_file(prob_path: Path, names, y_prob: np.ndarray, y_true: np.ndarray, class_names=None):
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    if class_names is None:
        class_names = get_disease_display_names()[:y_prob.shape[1]]
    if prob_path.suffix.lower() == ".npy":
        np.save(prob_path, y_prob.astype(np.float32))  # probabilities only
    else:
        np.savez_compressed(
            prob_path,
            volume=np.array(names, dtype=object),
            class_names=np.array(class_names, dtype=object),
            prob=y_prob.astype(np.float32),
            label=y_true.astype(np.int32),
        )

@torch.no_grad()
def run_test(model, loader, device, save_csv: Path = None, save_probs: Path = None):
    model.eval()
    y_true_list, y_prob_list, names = [], [], []
    with tqdm(total=len(loader), desc="Testing", leave=False, dynamic_ncols=True) as pbar:
        for images, labels, sample_names in loader:
            images = images.to(device, non_blocking=True)                       # [B,18,D,H,W]
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)  # [B,18]
            logits = model(images)                           # [B,18]
            probs = torch.sigmoid(logits).cpu().numpy()      # [B,18]
            y_true_list.append(labels.cpu().numpy())
            y_prob_list.append(probs)
            names.extend(sample_names)
            pbar.update(1)

    y_true = np.concatenate(y_true_list, axis=0).astype(int) if y_true_list else np.zeros((0, 18), dtype=int)
    y_prob = np.concatenate(y_prob_list, axis=0).astype(np.float32) if y_prob_list else np.zeros((0, 18), dtype=np.float32)

    per_class_metrics, macro = compute_metrics(y_true, y_prob, threshold=0.5)
    print(f"[Overall] AUC={macro['auc']:.4f} | Acc={macro['accuracy']:.4f} | "
          f"P={macro['precision']:.4f} | R={macro['recall']:.4f} | F1={macro['f1']:.4f}")

    class_names = get_disease_display_names()[:y_prob.shape[1]]
    print("Per-class metrics (threshold fixed at 0.5):")
    for i, cname in enumerate(class_names):
        auc  = per_class_metrics['auc'][i]
        acc  = per_class_metrics['accuracy'][i]
        prec = per_class_metrics['precision'][i]
        rec  = per_class_metrics['recall'][i]
        f1   = per_class_metrics['f1'][i]
        print(f"  - {cname:32s} AUC={auc:.4f} | Acc={acc:.4f} | P={prec:.4f} | R={rec:.4f} | F1={f1:.4f}")

    if save_csv is not None:
        save_predictions_csv(save_csv, names, y_prob, y_true, class_names=class_names)
        print(f"[CSV] Saved to {save_csv}")
        print("      Run utils/disease_predictions_medics.py on this file for the "
              "per-disease thresholds and metrics reported in the paper.")
    if save_probs is not None:
        save_prob_file(save_probs, names, y_prob, y_true, class_names=class_names)
        print(f"[PROBS] Saved to {save_probs}")


# --------------------------
# Arguments
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train or evaluate the AAmap classifier of EXACT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", type=str, default="test", choices=["train", "test"],
                   help="Whether to train the classifier or evaluate a checkpoint")
    p.add_argument("--gpu", type=str, default="0", help="Value for CUDA_VISIBLE_DEVICES")

    # Data and checkpoints
    p.add_argument("--heatmap-root", type=str, required=True,
                   help="Directory of AAmaps exported by EXACT_Pretrain/test.py --export-aamaps; "
                        "this is <export-dir>/prediction_heatmaps/epoch_N")
    p.add_argument("--h5-path", type=str, required=True,
                   help="Preprocessed HDF5 store holding the disease labels (label_18 / label_16)")
    p.add_argument("--ymamba-ckpt", type=str, required=True,
                   help="Pre-trained backbone, e.g. ../checkpoints/01_pretrain/ymamba_pretrain_best.pth")
    p.add_argument("--resume-weights", type=str, default=None,
                   help="Classifier weights. Required for --task test; optional for --task train "
                        "(resumes from an earlier run)")
    p.add_argument("--resume-optim", type=str, default=None,
                   help="Optimizer state to restore alongside --resume-weights")
    p.add_argument("--resume-epoch", type=int, default=None,
                   help="Epochs already completed, used to offset the LR schedule. "
                        "Parsed from the epoch_N pattern in --resume-weights when omitted")

    # Optimisation
    p.add_argument("--freeze-encoder", type=lambda x: str(x).lower() in ("1","true","yes","y"), default=True,
                   help="Freeze the Y-Mamba encoder weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-train-samples-per-epoch", type=int, default=0,
                   help="Cap on training samples per epoch (0 = use the whole split)")
    p.add_argument("--max-val-samples-per-epoch", type=int, default=900,
                   help="Cap on validation samples per epoch (0 = use the whole split)")
    p.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=["fixed", "exponential", "cosine"])
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="Warmup steps (cosine and exponential schedules)")
    p.add_argument("--min-lr-ratio", type=float, default=0.0,
                   help="Minimum cosine LR, as a fraction of --lr")
    p.add_argument("--exp-gamma", type=float, default=0.999,
                   help="Exponential decay factor (smaller decays faster)")
    p.add_argument("--exp-decay-steps", type=int, default=1, help="Decay every N steps")

    # Output
    p.add_argument("--results-root", type=str, default=None,
                   help="Root output directory (default: results/ next to this script)")
    p.add_argument("--run-prefix", type=str, default="ymamba",
                   help="Run directory prefix; the run directory is <prefix>__<timestamp>")
    p.add_argument("--save-every", type=int, default=2000,
                   help="Save a checkpoint every N training steps (0 disables)")
    p.add_argument("--save-csv", type=str, default=None,
                   help="Where to write the test prediction CSV")
    p.add_argument("--save-probs", type=str, default=None,
                   help="Where to write test probabilities (.npy or .npz)")

    # Logging (optional; SwanLab is only imported when enabled)
    p.add_argument("--log-swanlab", action="store_true",
                   help="Log training to SwanLab. Disabled by default so that the script "
                        "runs without a SwanLab account")
    p.add_argument("--sl-project", type=str, default="ct-encoder-classifier")
    p.add_argument("--sl-run-name", type=str, default=None)
    p.add_argument("--sl-mode", type=str, default="local", choices=["cloud", "local"],
                   help="SwanLab mode: local writes to disk only")

    args = p.parse_args()
    if args.task == "test" and not args.resume_weights:
        p.error("--task test requires --resume-weights "
                "(e.g. ../checkpoints/02_classification_finetune/classfine_best.pt)")
    return args


def make_lr_scheduler(
    optimizer,
    strategy: str,
    base_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    exp_gamma: float = 0.999,
    exp_decay_steps: int = 1
):
    def set_lr(lr):
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr

    if strategy == "fixed":
        def schedule(step: int):
            return set_lr(base_lr)
        return schedule

    if strategy == "exponential":
        warmup = int(max(0, warmup_steps))
        decay_steps = max(1, int(exp_decay_steps))
        gamma = float(exp_gamma)

        def schedule(step: int):
            if step < warmup:
                lr = base_lr * float(step) / float(max(1, warmup))
            else:
                k = (step - warmup) // decay_steps
                lr = base_lr * (gamma ** k)
            return set_lr(lr)

        return schedule

    # Default: cosine
    min_lr = base_lr * float(min_lr_ratio)
    warmup = int(max(0, warmup_steps))
    def schedule(step: int):
        if step < warmup:
            lr = base_lr * float(step) / float(max(1, warmup))
        else:
            denom = max(1, total_steps - warmup)
            progress = min(max((step - warmup) / float(denom), 0.0), 1.0)
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = min_lr + (base_lr - min_lr) * cos_decay
        return set_lr(lr)
    return schedule


def make_run_dirs(args: argparse.Namespace):
    script_dir = Path(__file__).resolve().parent
    default_results_root = script_dir / "results"
    results_root = Path(args.results_root) if args.results_root else default_results_root
    results_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%A_%d_%B_%Y_%Hh_%Mm_%Ss")
    run_name = args.sl_run_name if args.sl_run_name else f"{args.run_prefix}__{ts}"

    # Never reuse an existing run directory: an interrupted run keeps its artefacts.
    run_dir = results_root / run_name
    suffix = 1
    _tmp = run_dir
    while _tmp.exists():
        _tmp = results_root / f"{run_name}_{suffix}"
        suffix += 1
    run_dir = _tmp
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Paths] results_root: {results_root}")
    print(f"[Paths] run_dir    : {run_dir}")
    print(f"[Paths] ckpt_dir   : {ckpt_dir}")
    return run_dir, ckpt_dir


def load_state_dict_flexible(model: nn.Module, state_dict: dict, strict: bool = False):
    """Load a state dict written with or without a DataParallel 'module.' prefix."""
    try:
        model.load_state_dict(state_dict, strict=strict)
        return True
    except RuntimeError:
        if any(k.startswith("module.") for k in state_dict.keys()):
            new_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            new_sd = {"module." + k: v for k, v in state_dict.items()}
        model.load_state_dict(new_sd, strict=strict)
        return True


def clean_optimizer_state_shape_mismatch(optimizer: torch.optim.Optimizer) -> int:
    """Drop momentum buffers whose shape no longer matches their parameter.

    Returns the number of removed entries.
    """
    removed = 0
    for p, st in list(optimizer.state.items()):
        if not isinstance(st, dict):
            continue
        for k in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
            t = st.get(k, None)
            if t is not None and hasattr(p, "data") and t.shape != p.data.shape:
                st.pop(k, None)
                removed += 1
    return removed


# --------------------------
# Main flow
# --------------------------
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = setting_config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classifier = HeatmapMultiLabelClassifier(ymamba_ckpt=args.ymamba_ckpt,
                                             freeze_encoder=args.freeze_encoder,
                                             dropout_prob=args.dropout).to(device)

    devices = list(range(torch.cuda.device_count()))
    if len(devices) > 1:
        classifier = torch.nn.DataParallel(classifier, device_ids=devices)

    # ====== Evaluation ======
    if args.task == "test":
        test_dataset = HeatmapDataset(root_dir=args.heatmap_root,
                                      h5_path=args.h5_path, test=True)
        print(f"[Data] {len(test_dataset)} studies from {args.heatmap_root}")

        state = torch.load(args.resume_weights, map_location=device)
        model_to_load = classifier.module if hasattr(classifier, 'module') else classifier
        load_state_dict_flexible(model_to_load, state, strict=False)
        print(f"[Weights] Loaded {args.resume_weights}")

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=config.num_workers, pin_memory=True, drop_last=False)

        run_test(classifier, test_loader, device,
                 save_csv=Path(args.save_csv) if args.save_csv else None,
                 save_probs=Path(args.save_probs) if args.save_probs else None)
        return

    # ====== Training ======
    run_dir, ckpt_dir = make_run_dirs(args)

    logger = None
    if args.log_swanlab:
        import swanlab
        run_name = run_dir.name
        sl_config = {**vars(args), "run_dir": str(run_dir), "ckpt_dir": str(ckpt_dir)}
        try:
            swanlab.init(project=args.sl_project, experiment_name=run_name,
                         config=sl_config, mode=args.sl_mode)
        except Exception as e:
            print(f"[SwanLab] Initialization failed in mode {args.sl_mode}: {e}\n"
                  f"-> Falling back to local mode.")
            swanlab.init(project=args.sl_project, experiment_name=run_name,
                         config=sl_config, mode="local")
        logger = swanlab

    train_dataset = HeatmapDataset(root_dir=args.heatmap_root,
                                   h5_path=args.h5_path, train=True)
    val_dataset   = HeatmapDataset(root_dir=args.heatmap_root,
                                   h5_path=args.h5_path, val=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=config.num_workers, drop_last=False)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, num_workers=config.num_workers, drop_last=False)

    # Per-finding positive weights, computed as neg/pos on the CT-RATE training split.
    # Order matches get_disease_names().
    pos_weight = torch.tensor([
        9.211362733,  2.384068466,  8.295479204, 32.8629776,   2.992233613,
        6.064870808,  3.176470588,  4.187083754, 3.022222222,  1.216071737,
        1.677849552,  3.152851834,  7.123261694, 18.16629381, 13.8480647,
        6.335045662, 10.81701149, 13.40695067
    ], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.max_train_samples_per_epoch and args.max_train_samples_per_epoch > 0:
        steps_per_epoch = math.ceil(args.max_train_samples_per_epoch / max(1, args.batch_size))
    else:
        steps_per_epoch = math.ceil(len(train_dataset) / max(1, args.batch_size))
    total_steps = args.epochs * steps_per_epoch

    scheduler = make_lr_scheduler(
        optimizer=optimizer,
        strategy=args.lr_schedule,
        base_lr=args.lr,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        exp_gamma=args.exp_gamma,
        exp_decay_steps=args.exp_decay_steps
    )

    global_step = 0
    resume_epoch_idx = 0
    if args.resume_weights:
        w_path = Path(args.resume_weights)
        print(f"[Resume] Loading classifier weights: {w_path}")
        state = torch.load(w_path, map_location=device)
        model_to_load = classifier.module if hasattr(classifier, 'module') else classifier
        load_state_dict_flexible(model_to_load, state, strict=False)

        if args.resume_epoch is not None:
            resume_epoch_idx = int(args.resume_epoch)
        else:
            import re
            m = re.search(r"epoch_(\d+)", w_path.name)
            resume_epoch_idx = int(m.group(1)) if m else 0

        global_step = resume_epoch_idx * steps_per_epoch
        print(f"[Resume] Treating epoch {resume_epoch_idx} as complete; global_step = {global_step}")

        if args.resume_optim:
            optim_path = Path(args.resume_optim)
            if optim_path.exists():
                print(f"[Resume] Loading optimizer state: {optim_path}")
                optim_state = torch.load(optim_path, map_location=device)
                try:
                    optimizer.load_state_dict(optim_state)
                    n_removed = clean_optimizer_state_shape_mismatch(optimizer)
                    if n_removed > 0:
                        print(f"[Resume] Cleared {n_removed} momentum states with shape mismatches.")
                except Exception as e:
                    print(f"[Resume] Failed to load optimizer state (continuing without it): {e}")
            else:
                print(f"[Resume] Optimizer state file not found: {optim_path}; skipping.")

    best_macro_f1 = -1.0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_macro_f1": []}

    for local_epoch in range(1, args.epochs + 1):
        total_epoch_num = resume_epoch_idx + local_epoch
        model_to_save = classifier.module if hasattr(classifier, 'module') else classifier

        train_loss, train_processed, global_step = train_one_epoch(
            model=classifier, loader=train_loader, device=device, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, start_step=global_step,
            max_train_samples_per_epoch=args.max_train_samples_per_epoch,
            epoch=total_epoch_num, grad_clip=1.0, save_every=args.save_every,
            ckpt_dir=ckpt_dir, model_to_save=model_to_save
        )

        y_true, y_prob, names, val_processed, val_loss = evaluate(
            classifier, val_loader, device, criterion,
            max_val_samples_per_epoch=args.max_val_samples_per_epoch,
            epoch=total_epoch_num,
        )
        per_class_metrics, macro = compute_metrics(y_true, y_prob, threshold=0.5)

        try:
            current_lr = optimizer.param_groups[0]["lr"]
        except Exception:
            current_lr = args.lr

        print(
            f"[Epoch {total_epoch_num:02d}] "
            f"TrainLoss={train_loss:.5f} ({train_processed} samples) | "
            f"ValLoss={val_loss:.5f} ({val_processed} samples) | "
            f"Val(macro): AUC={macro['auc']:.4f}, Acc={macro['accuracy']:.4f}, "
            f"P={macro['precision']:.4f}, R={macro['recall']:.4f}, F1={macro['f1']:.4f}"
        )

        if logger is not None:
            logger.log({
                "epoch_total": total_epoch_num,
                "epoch_local": local_epoch,
                "lr": float(current_lr),
                "train/epoch_loss": float(train_loss),
                "train/processed": int(train_processed),
                "val/epoch_loss": float(val_loss),
                "val/processed": int(val_processed),
                "val/macro/auc": float(macro["auc"]),
                "val/macro/accuracy": float(macro["accuracy"]),
                "val/macro/precision": float(macro["precision"]),
                "val/macro/recall": float(macro["recall"]),
                "val/macro/f1": float(macro["f1"]),
            })

        torch.save(model_to_save.state_dict(), ckpt_dir / f'epoch_{total_epoch_num}.pt')
        torch.save(optimizer.state_dict(),  ckpt_dir / f'optim_epoch_{total_epoch_num}.pt')
        print(f"Saved epoch checkpoint at {ckpt_dir}/epoch_{total_epoch_num}.pt")

        if macro["f1"] > best_macro_f1:
            best_macro_f1 = macro["f1"]
            torch.save(model_to_save.state_dict(), ckpt_dir / "best.pt")
            with open(run_dir / "best_info.json", "w") as f:
                json.dump({
                    "epoch": total_epoch_num,
                    "macro_f1": best_macro_f1,
                    "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2, ensure_ascii=False)
            print(f"Updated best: epoch={total_epoch_num}, macro F1={best_macro_f1:.4f} -> {ckpt_dir}/best.pt")

        history["epoch"].append(total_epoch_num)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_macro_f1"].append(float(macro["f1"]))

    plt.figure(figsize=(8,5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training/Validation Loss"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(run_dir / "loss_curves.png"); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history["epoch"], history["val_macro_f1"], label="Val Macro F1")
    plt.xlabel("Epoch"); plt.ylabel("Macro F1"); plt.title("Validation Macro F1"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(run_dir / "val_macro_f1.png"); plt.close()

    with open(run_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if logger is not None:
        logger.finish()

    print(f"\nTraining finished. Best model: {ckpt_dir/'best.pt'}")
    print(f"Run directory:        {run_dir}")


if __name__ == "__main__":
    main()
