#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
Use YMamba as the pretrained image encoder, extract global latent vectors (768),
and attach a linear head for 18-class multi-label classification.
Training strategy: AdamW + BCEWithLogits (fixed pos_weight) + cosine LR (with warmup)
+ gradient clipping + tqdm progress bar + SwanLab logging.
Supports: --freeze-encoder, --max-train-samples-per-epoch / --max-val-samples-per-epoch, --save-every.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import csv
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ====== Project modules (based on your project structure) ======
from models.vmunet.ymamba import YMamba
from models.vmunet.heatmap import LightweightHeatmapEncoder
from configs.config_setting import setting_config
from datasets.dataset import HeatmapDataset

warnings.filterwarnings("ignore")

# ====== SwanLab (local/cloud logging, auto-fallback to local on cloud init failure) ======
import swanlab
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
# Cosine learning-rate scheduler (with warmup)
# --------------------------
def cosine_lr(optimizer, base_lr, warmup_length, total_steps, min_lr_ratio=0.0):
    """
    Return a schedule(step) function that sets the learning rate by global step.
    """
    min_lr = base_lr * float(min_lr_ratio)

    def _lr_at(step: int):
        if step < warmup_length:
            return base_lr * float(step) / float(max(1, warmup_length))
        progress = (step - warmup_length) / float(max(1, total_steps - warmup_length))
        progress = min(max(progress, 0.0), 1.0)
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cos_decay

    def schedule(step: int):
        lr = _lr_at(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr

    return schedule


# --------------------------
# YMamba -> latent (768) extraction backbone
# forward(x, return_latents=True) returns (None, latents, None)
# --------------------------
class YMambaLatentBackbone(nn.Module):
    def __init__(self, ymamba: YMamba, freeze_encoder: bool = True):
        super().__init__()
        self.backbone = ymamba
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  # global average pooling

    def forward(self, x: torch.Tensor, return_latents: bool = False):
        ctx = torch.enable_grad() if not self.freeze_encoder else torch.no_grad()
        with ctx:
            bb = self.backbone
            outs = bb.vit(x)
            _enc1 = bb.encoder1(x)
            _enc2 = bb.encoder2(outs[0])
            _enc3 = bb.encoder3(outs[1])
            _enc4 = bb.encoder4(outs[2])
            enc5  = bb.encoder5(outs[3])     # [B,768,D/16,H/16,W/16]
            latents = self.gap(enc5).flatten(1)  # [B,768]

        if return_latents:
            return (None, latents, None)
        else:
            return latents  # [B,768]

class HeatmapBinaryClassifier(nn.Module):
    def __init__(self, ymamba_ckpt: str, freeze_encoder: bool = True, dropout_prob: float = 0.3):
        super().__init__()
        # Lightweight heatmap encoder (uses only the first 3 YMamba stages)
        self.encoder = LightweightHeatmapEncoder(ymamba_checkpoint_path=ymamba_ckpt,
                                                 freeze=freeze_encoder)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(768, 1)  # map pooled feature to 1 logit (binary)

    def forward(self, images: torch.Tensor, *args, **kwargs):
        """
        images: [B, 1, 64, 128, 128] (single-channel heatmap from HeatmapDataset)
        return: [B, 1] logits
        """
        # Encoder output: [B, 16, 32, 32, 768]
        feats = self.encoder(images)
        # Global average pooling over 16x32x32 -> [B, 768]
        feats = feats.mean(dim=(1, 2, 3))
        feats = self.dropout(feats)
        logits = self.fc(feats)  # [B, 1]
        return logits
# class HeatmapMultiLabelClassifier(nn.Module):
#     def __init__(self, ymamba_ckpt: str, freeze_encoder: bool = True, dropout_prob: float = 0.3,
#                  in_channels: int = 18, num_classes: int = 18):
#         super().__init__()
#         # Directly set encoder input channels to 18 (no 1x1x1 fusion)
#         self.encoder = LightweightHeatmapEncoder(
#             ymamba_checkpoint_path=ymamba_ckpt,
#             freeze=freeze_encoder,
#             in_channels=in_channels  # requires LightweightHeatmapEncoder to support this argument
#         )
#         self.dropout = nn.Dropout(dropout_prob)
#         self.fc = nn.Linear(768, num_classes)

#     def forward(self, images: torch.Tensor, *args, **kwargs):
#         """
#         images: [B, 18, D, H, W]
#         return: [B, 18] logits
#         """
#         feats = self.encoder(images)        # expected [B, 16, 32, 32, 768]
#         feats = feats.mean(dim=(1, 2, 3))   # [B, 768]
#         feats = self.dropout(feats)
#         return self.fc(feats)               # [B, 18]
class HeatmapMultiLabelClassifier(nn.Module):
    def __init__(self, ymamba_ckpt: str, freeze_encoder: bool = True, dropout_prob: float = 0.3,
                 in_channels: int = 18, num_classes: int = 18):
        super().__init__()
        self.num_classes = num_classes

        class SingleChannelBinary(nn.Module):
            def __init__(self, ckpt, freeze, dropout_p):
                super().__init__()
                # Independent encoder for each head (single-channel input)
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

        # 18 independent binary heads
        self.heads = nn.ModuleList([
            SingleChannelBinary(ymamba_ckpt, freeze_encoder, dropout_prob)
            for _ in range(self.num_classes)
        ])

    def forward(self, images: torch.Tensor, *args, **kwargs):
        """
        images: [B, 18, D, H, W], each channel goes to its corresponding binary head
        return: [B, 18] logits
        """
        assert images.size(1) == self.num_classes, f"Expected {self.num_classes} channels, got {images.size(1)}"
        logits = []
        for i in range(self.num_classes):
            xi = images[:, i:i+1, ...]   # [B,1,D,H,W]
            li = self.heads[i](xi)       # [B,1]
            logits.append(li)
        return torch.cat(logits, dim=1)  # [B,18]
# --------------------------
# Classifier head with the same interface as the example
# --------------------------
class ImageLatentsClassifier(nn.Module):
    def __init__(self, trained_model: nn.Module, latent_dim: int, num_classes: int,
                 dropout_prob: float = 0.3):
        """
        trained_model: must support forward(x, return_latents=True) -> (None, latents, None)
        """
        super().__init__()
        self.trained_model = trained_model
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, *args, **kwargs):
        kwargs['return_latents'] = True
        _, image_latents, _ = self.trained_model(*args, **kwargs)  # [B,768]
        image_latents = self.relu(image_latents)
        image_latents = self.dropout(image_latents)
        return self.classifier(image_latents)  # [B,18]

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        loaded_state_dict = torch.load(file_path, map_location="cpu")
        self.load_state_dict(loaded_state_dict)


# --------------------------
# Metrics/utilities
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
# Train/validation loops
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
        # HeatmapDataset: (images_18ch, labels_18, sample_name)
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


def save_predictions_csv(csv_path: Path, names, y_prob: np.ndarray, y_true: np.ndarray, class_names=None):
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
def run_test(model, loader, device, save_csv: Path | None = None, save_probs: Path | None = None):
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
    print("Per-class metrics:")
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
    if save_probs is not None:
        save_prob_file(save_probs, names, y_prob, y_true, class_names=class_names)
        print(f"[PROBS] Saved to {save_probs}")
# @torch.no_grad()
# def evaluate(model, loader, device, criterion, max_val_samples_per_epoch=0, epoch=1,pos_weight_map: dict = None):
#     model.eval()
#     y_true_list, y_prob_list, names = [], [], []
#     processed = 0
#     val_loss_sum = 0.0
#     total_steps = _tqdm_total_steps(loader, max_val_samples_per_epoch)

#     with tqdm(total=total_steps, desc=f"Valid Epoch {epoch}", leave=False, dynamic_ncols=True) as pbar:
#         # HeatmapDataset: (image, label, sample_name, abn_name)
#         for images, labels, sample_names, abn_names in loader:
#             images = images.to(device, non_blocking=True)                               # [B, 1, D, H, W]
#             labels = labels.to(device, dtype=torch.float32, non_blocking=True).view(-1, 1)  # [B,1]

#             logits = model(images)                  # [B,1]
#             probs = torch.sigmoid(logits).cpu().numpy()  # [B,1]
#             if pos_weight_map is not None:
#                 pw = pos_weight_from_abn_names(abn_names, pos_weight_map, device)
#                 loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw).item()
#             else:
#                 loss = criterion(logits, labels).item()
#             # loss  = criterion(logits, labels).item()

#             val_loss_sum += loss * images.size(0)
#             y_prob_list.append(probs)
#             y_true_list.append(labels.cpu().numpy())
#             names.extend(sample_names)  # for per-disease evaluation, collect abn_names as well

#             processed += images.size(0)
#             avg_val_loss = val_loss_sum / max(processed, 1)
#             pbar.set_postfix({"avg_val_loss": f"{avg_val_loss:.5f}"})
#             pbar.update(1)

#             if max_val_samples_per_epoch and processed >= max_val_samples_per_epoch:
#                 break

#     y_prob = np.concatenate(y_prob_list, axis=0) if y_prob_list else np.zeros((0, 1), dtype=np.float32)
#     y_true = np.concatenate(y_true_list, axis=0).astype(int) if y_true_list else np.zeros((0, 1), dtype=int)
#     avg_val_loss = val_loss_sum / max(processed, 1)
#     return y_true, y_prob, names, processed, avg_val_loss

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
# Arguments
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze-encoder", type=lambda x: str(x).lower() in ("1","true","yes","y"), default=True,
                   help="Whether to freeze YMamba encoder parameters (default: True)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-train-samples-per-epoch", type=int, default=0,
                   help="Maximum training samples per epoch (0 means no limit)")
    p.add_argument("--max-val-samples-per-epoch", type=int, default=900,
                   help="Maximum validation samples per epoch (0 means no limit)")
    # Learning-rate strategy (consistent with train.py)
    p.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=["fixed", "exponential", "cosine"],
                   help="Learning-rate strategy: fixed / exponential / cosine")
    p.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps (available for cosine/exponential)")
    p.add_argument("--min-lr-ratio", type=float, default=0.0, help="Minimum cosine LR = base_lr * min_lr_ratio")
    p.add_argument("--exp-gamma", type=float, default=0.999, help="Exponential decay factor (<1 decays faster)")
    p.add_argument("--exp-decay-steps", type=int, default=1, help="Decay every N steps (integer)")
    # Class imbalance handling (consistent with train.py)
    p.add_argument("--class-weight-source", type=str, default="fixed",
                   choices=["fixed", "freq"],
                   help="pos_weight source: fixed=use fixed vector; freq=estimate from train-set frequency")
    p.add_argument("--freq-eps", type=float, default=1e-6, help="Numerical floor when computing frequency to avoid division by zero")
    # Save/directories (consistent with train.py)
    p.add_argument("--results-root", type=str, default=None,
                   help="Root output directory (default: results/ under current project)")
    p.add_argument("--run-prefix", type=str, default="ymamba",
                   help="Run directory prefix (generates <prefix>__<timestamp>)")
    p.add_argument("--save-every", type=int, default=2000, help="Save checkpoint every N training steps")
    # Data and checkpoint
    p.add_argument("--ymamba-ckpt", type=str, default="/path/to/ymamba_pretrained/checkpoints/best.pth")
    p.add_argument("--h5-path", type=str, default="/path/to/data/processed_data.h5")
    # SwanLab
    p.add_argument("--sl-project", type=str, default="ct-encoder-classifier")
    p.add_argument("--sl-run-name", type=str, default=None)
    p.add_argument("--sl-mode", type=str, default="cloud", choices=["cloud", "local"],
                   help="SwanLab mode: cloud=remote logging, local=local only")
    # Resume training (consistent with train.py)
    p.add_argument("--resume-weights", type=str, default="/path/to/results/checkpoints/checkpoint_step_N_epoch_N.pt",
                   help="Classifier weights path to load (e.g. /.../checkpoints/epoch_100.pt)")
    p.add_argument("--resume-optim", type=str, default="/path/to/results/checkpoints/optim_step_N_epoch_N.pt",
                   help="Optimizer state path to load (e.g. /.../checkpoints/optim_epoch_100.pt), auto-infer if omitted")
    p.add_argument("--resume-epoch", type=int, default=5,
                   help="Historical epoch count for global_step (if omitted, parse epoch_XX from resume-weights filename)")
    p.add_argument("--task",type=str,default="test",help="Task to run: train / test")
    p.add_argument("--save-csv", type=str, default="/path/to/results/heatmap_results/pred.csv", help="CSV path for saving test predictions")
    p.add_argument("--save-probs", type=str, default="/path/to/results/heatmap_results/prob.npz", help="Probability file path for test output (.npy or .npz)")
    return p.parse_args()

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
    try:
        model.load_state_dict(state_dict, strict=strict)
        return True
    except RuntimeError:
        if any(k.startswith("module.") for k in state_dict.keys()):
            new_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_sd, strict=strict)
            return True
        else:
            new_sd = {"module."+k: v for k, v in state_dict.items()}
            model.load_state_dict(new_sd, strict=strict)
            return True
        
def compute_pos_weight_from_loader(loader, num_classes=18, eps=1e-6):
    total = 0
    pos = np.zeros((num_classes,), dtype=np.float64)
    for _images, _masks, labels, _names in loader:
        labels = labels.numpy()
        total += labels.shape[0]
        pos += labels.sum(axis=0)
    pos = np.clip(pos, eps, None)
    neg = np.clip(total - pos, eps, None)
    pos_weight = neg / pos
    return torch.tensor(pos_weight, dtype=torch.float32)
def compute_pos_weight_from_loader_binary(loader, eps=1e-6):
    total = 0
    pos = 0.0
    for images, labels, sample_names, abn_names in loader:
        labels_np = labels.numpy()
        total += labels_np.shape[0]
        pos += labels_np.sum()
    pos = max(pos, eps)
    neg = max(total - pos, eps)
    return torch.tensor([neg / pos], dtype=torch.float32)
def build_pos_weight_map_from_vector(fixed_pos_weight: torch.Tensor):
    names = get_disease_names()
    assert len(names) == fixed_pos_weight.numel(), "fixed_pos_weight length must match the disease list"
    return {names[i]: float(fixed_pos_weight[i].item()) for i in range(len(names))}

def pos_weight_from_abn_names(abn_names, pos_weight_map: dict, device: torch.device):
    w = [pos_weight_map.get(str(n), 1.0) for n in abn_names]
    return torch.tensor(w, dtype=torch.float32, device=device).view(-1, 1)
def clean_optimizer_state_shape_mismatch(optimizer: torch.optim.Optimizer) -> int:
    """
    Remove momentum states with shape mismatches against current parameters
    (e.g., exp_avg / exp_avg_sq).
    Return the number of removed state entries.
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
    config = setting_config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_dir, ckpt_dir = make_run_dirs(args)


    run_name = run_dir.name
    sl_config = {**vars(args), "run_dir": str(run_dir), "ckpt_dir": str(ckpt_dir)}
    for k in ["resume_weights", "resume_optim", "ymamba_ckpt", "h5_path", "results_root"]:
        if sl_config.get(k) is not None:
            sl_config[k] = str(sl_config[k])
    try:
        swanlab.init(project=args.sl_project, experiment_name=run_name, config=sl_config, mode=args.sl_mode)
    except Exception as e:
        print(f"[SwanLab] Initialization failed in mode {args.sl_mode}: {e}\n-> Falling back to local mode.")
        swanlab.init(project=args.sl_project, experiment_name=run_name, config=sl_config, mode="local")

    # ====== Build YMamba and load weights ======
    model_cfg = {
        'num_classes': config.num_classes,
        'num_abnormal_classes': config.num_abnormal_classes,
        'input_channels': config.input_channels,
    }

    # classifier = HeatmapBinaryClassifier(ymamba_ckpt=args.ymamba_ckpt,
    #                                      freeze_encoder=args.freeze_encoder,
    #                                      dropout_prob=args.dropout).to(device)

    classifier=HeatmapMultiLabelClassifier(ymamba_ckpt=args.ymamba_ckpt,
                                           freeze_encoder=args.freeze_encoder,
                                           dropout_prob=args.dropout).to(device)
    # Multi-GPU
    devices = list(range(torch.cuda.device_count()))
    if len(devices) > 1:
        classifier = torch.nn.DataParallel(classifier, device_ids=devices)

    if args.task=="test":
        test_dataset=HeatmapDataset(root_dir="/path/to/test_results/prediction_heatmaps/epoch_12",
        h5_path="/path/to/data/valid_processed_data.h5",test=True)

        # test_dataset   = HeatmapDataset(root_dir="/path/to/data/heatmap",
        #                            h5_path="/path/to/data/processed_data.h5", val=True)
        # test_dataset=HeatmapDataset(root_dir="/path/to/test_results/ymamba__<timestamp>/test_results/prediction_heatmaps/epoch_12",
        # h5_path="/path/to/data/processed_data.h5",test=True)
        # test_dataset=HeatmapDataset(root_dir="/path/to/test_results/ymamba__<timestamp>/test_results/prediction_heatmaps/epoch_12",
        # h5_path="/path/to/data/processed_ct_data.h5",test=True)

        state=torch.load("/path/to/results/checkpoints/best.pt",map_location=device)
        model_to_load = classifier.module if hasattr(classifier, 'module') else classifier
        load_state_dict_flexible(model_to_load, state, strict=False)

        test_loader=DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=config.num_workers,pin_memory=True,drop_last=False)
        
        save_csv_path = Path(args.save_csv) if args.save_csv else None

        save_probs = Path(args.save_probs) if args.save_probs else None

        run_test(classifier, test_loader, device, save_csv=save_csv_path,save_probs=save_probs)
        swanlab.finish()
        return

    
    # ====== Data ======
    
    train_dataset = HeatmapDataset(root_dir="/path/to/data/heatmap",
                                   h5_path="/path/to/data/processed_data.h5", train=True)
    val_dataset   = HeatmapDataset(root_dir="/path/to/data/heatmap",
                                   h5_path="/path/to/data/processed_data.h5", val=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=config.num_workers, drop_last=False)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, num_workers=config.num_workers, drop_last=False)
    # ====== Imbalanced-class weights (fixed vector; strict order of 18 diseases) ======
    # This weight vector comes from your example and aligns with get_disease_names() order:
    # fixed_pos_weight = torch.tensor([
    #     9.211362733,  2.384068466,  8.295479204, 32.8629776,   2.992233613,
    #     6.064870808,  3.176470588,  4.187083754, 3.022222222,  1.216071737,
    #     1.677849552,  3.152851834,  7.123261694, 18.16629381, 13.8480647,
    #     6.335045662, 10.81701149, 13.40695067
    # ], dtype=torch.float32, device=device)
    # fixed_pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device) 

    # if args.class_weight_source == "fixed":
    #     pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
    #     print("[Class Weights] Using fixed vector (binary=1.0).")
    # else:
    #     print("[Class Weights] Using train-set frequency to compute pos_weight (binary) ...")
    #     pw = compute_pos_weight_from_loader_binary(train_loader, eps=args.freq_eps)
    #     pos_weight = pw.to(device)
    #     print(f"[Class Weights] pos_weight computed. value={pos_weight.item():.4f}")

    # if args.class_weight_source == "fixed":
    pos_weight = torch.tensor([
            9.211362733,  2.384068466,  8.295479204, 32.8629776,   2.992233613,
            6.064870808,  3.176470588,  4.187083754, 3.022222222,  1.216071737,
            1.677849552,  3.152851834,  7.123261694, 18.16629381, 13.8480647,
            6.335045662, 10.81701149, 13.40695067
        ], dtype=torch.float32, device=device)
        # print("[Class Weights] Using fixed vector.")
    # else:
    #     print("[Class Weights] Using train-set frequency to compute pos_weight ...")
    #     pw = compute_pos_weight_from_loader(train_loader, num_classes=18, eps=args.freq_eps)
    #     pos_weight = pw.to(device)
    #     print(f"[Class Weights] pos_weight computed. mean={pos_weight.mean().item():.4f}, "
    #           f"min={pos_weight.min().item():.4f}, max={pos_weight.max().item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=fixed_pos_weight)
    # pos_weight_map = build_pos_weight_map_from_vector(fixed_pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # ====== Optimizer and scheduler ======
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



    # ====== Training loop ======
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
        print(f"[Resume] Inferred completion at epoch {resume_epoch_idx}; set global_step = {global_step}")

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
                    print(f"[Resume] Failed to load optimizer state (ignore and continue): {e}")
            else:
                print(f"[Resume] Optimizer state file not found: {optim_path}; skip.")
        else:
            print("[Resume] Skip optimizer restore (--resume-optim not provided).")

    # ====== Training loop (save to run_dir/ckpt_dir) ======
    best_macro_f1 = -1.0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_macro_f1": []}

    # swanlab.log({
    #     "pos_weight_mean": float(pos_weight.mean().item()),
    #     "pos_weight_min": float(pos_weight.min().item()),
    #     "pos_weight_max": float(pos_weight.max().item())
    # })

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
        print("y_true shape:", y_true.shape)
        print("y_prob shape:", y_prob.shape)
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

        swanlab.log({
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

    # Save curves/history to run_dir
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

    swanlab.finish()

    print(f"\nTraining finished! Best model: {ckpt_dir/'best.pt'}")
    print(f"Run directory (non-overwriting): {run_dir}")
    print(f"Checkpoint directory: {ckpt_dir}")


if __name__ == "__main__":
    main()
