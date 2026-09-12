"""EXACT - multi-disease diagnosis evaluation.

Runs the pre-trained Y-Mamba backbone over a preprocessed HDF5 test set and reports
per-disease AUROC / precision / recall / F1 / accuracy together with macro averages.

The same script covers the internal cohort (CT-RATE) and the external cohorts
(RAD-ChestCT, MianYang): organ masks are not read, so cohorts that ship without
region masks can be evaluated unchanged.

Example
-------
    python test.py \
        --h5 /path/to/valid_total_processed_data.h5 \
        --checkpoint ../checkpoints/01_pretrain/ymamba_pretrain_best.pth \
        --output-dir results/ct_rate

Metric conventions
------------------
Two choices materially affect the reported F1 and accuracy (AUROC is invariant to
both). They are exposed as flags rather than hard-coded so that published numbers
can be reproduced and audited:

  --threshold-source  where the per-disease decision threshold comes from.
                      'checkpoint' uses thresholds fitted on the validation split
                      and stored in the checkpoint (default).
                      'fit-on-test' refits them on the test set itself, which is
                      optimistically biased.
  --positive-class    'present' scores disease presence as the positive class
                      (default, the usual convention for detection).
                      'absent' scores disease absence as positive; because the
                      cohorts are dominated by negatives, this raises F1 markedly.

Passing --reproduce-paper selects the combination used for the tables in the
EXACT manuscript ('fit-on-test' + 'absent').
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import Abnormal_loss, save_prediction_heatmaps  # noqa: E402
from models.ymamba.ymamba import YMamba  # noqa: E402

# Channel order of the segmentation decoder: six organs plus the derived global
# channel. save_prediction_heatmaps masks each finding with the organ it belongs to,
# so this order must match the one the model was trained with.
ORGAN_NAMES = [
    "lung",
    "trachea and bronchie",
    "pleura",
    "mediastinum",
    "heart",
    "esophagus",
    "global",
]

# The 18 findings the backbone predicts, in channel order.
DISEASE_NAMES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

# Positive rates in the CT-RATE training split. Abnormal_loss uses them to weight
# the multi-instance pooling, so they are part of the model definition rather than
# a property of whichever cohort is being evaluated.
DISEASE_FREQUENCIES = [
    0.1020, 0.2837, 0.1072, 0.0705, 0.2476, 0.1420, 0.2534, 0.1939, 0.2558,
    0.4548, 0.3666, 0.2672, 0.1185, 0.0744, 0.1034, 0.1755, 0.0999, 0.0788,
]

# RAD-ChestCT and MianYang annotate 16 of the 18 findings. Their label vectors are
# expanded to 18 channels by inserting zeros at these positions; the affected
# channels carry no ground truth and are excluded from the macro averages.
LABEL_16_TO_18_GAPS = [4, 13]


class DiagnosisH5Dataset(Dataset):
    """CT volumes and disease labels from a preprocessed HDF5 file.

    Each top-level group is one study and must contain a ``ct`` dataset of shape
    (1, D, H, W). Labels are read from ``label_18`` when present, otherwise from
    ``label_16``, which is expanded to 18 channels by inserting zeros at the
    positions listed in LABEL_16_TO_18_GAPS.
    """

    def __init__(self, h5_path):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.keys = sorted(k.strip() for k in f.keys())
            if not self.keys:
                raise RuntimeError(f"No samples in {h5_path}")
            probe = f[self.keys[0]]
            if "label_18" in probe:
                self.label_key = "label_18"
            elif "label_16" in probe:
                self.label_key = "label_16"
            else:
                raise KeyError(
                    f"Sample '{self.keys[0]}' has neither 'label_18' nor 'label_16'. "
                    "Run data_preprocessed/save_label_18.py to attach labels to the "
                    "HDF5 file."
                )
            self.ct_shape = probe["ct"].shape

        # Channels without ground truth in this cohort.
        self.unlabelled = list(LABEL_16_TO_18_GAPS) if self.label_key == "label_16" else []

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        with h5py.File(self.h5_path, "r") as f:
            ct = f[key]["ct"][:]
            label = f[key][self.label_key][:]

        ct = torch.tensor(ct, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.label_key == "label_16":
            expanded = torch.zeros(18, dtype=label.dtype)
            expanded[0:4] = label[0:4]
            expanded[5:13] = label[4:12]
            expanded[14:18] = label[12:16]
            label = expanded

        return ct, label, key


def load_model(checkpoint_path, device):
    """Build YMamba and load weights, tolerating a DataParallel-prefixed state dict.

    Unlike a permissive load, key mismatches are reported and raise: silently
    starting from random weights would still produce a full set of plausible-looking
    metrics.
    """
    model = YMamba(
        in_chans=1,
        num_classes=7,               # 6 organs + global channel
        num_abnormal_classes=18,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size=768,
        norm_name="instance",
        conv_block=True,
        res_block=True,
        spatial_dims=3,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if next(iter(state_dict)).startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the model definition.\n"
            f"  missing keys:    {sorted(missing)[:10]}\n"
            f"  unexpected keys: {sorted(unexpected)[:10]}"
        )

    thresholds = checkpoint.get("best_thresholds")
    if thresholds is not None:
        thresholds = np.asarray(thresholds, dtype=float).reshape(-1)

    model.eval()
    return model, thresholds, checkpoint.get("epoch")


@torch.no_grad()
def predict(model, loader, device, export_dir=None, thresholds=None, epoch=0):
    """Return study-level probabilities (N, 18), labels (N, 18) and study ids.

    When export_dir is set, the voxel-level AAmaps are also written out, one
    directory per study, in the layout EXACT_ClassFinetune and EXACT-Seg expect.
    """
    probs, labels, names = [], [], []

    for step, (images, targets, keys) in enumerate(loader, start=1):
        images, targets = images.to(device), targets.to(device)
        seg_pred, abnormal_preds = model(images)

        # Abnormal_loss pools the voxel-wise disease maps into one probability per
        # study, restricted to the organ each finding belongs to.
        _, pooled = Abnormal_loss(
            seg_pred, abnormal_preds[-1], targets, DISEASE_FREQUENCIES
        )

        probs.append(pooled.cpu().numpy())
        labels.append(targets.cpu().numpy())
        names.extend(keys)

        if export_dir is not None:
            for b in range(images.size(0)):
                save_prediction_heatmaps(
                    predictions=[
                        abnormal_preds[0][b:b + 1].cpu(),   # low resolution
                        abnormal_preds[1][b:b + 1].cpu(),   # high resolution
                    ],
                    segmentation_preds=seg_pred[b:b + 1].cpu(),
                    targets=targets[b:b + 1].cpu(),
                    images=images[b:b + 1].cpu(),
                    epoch=epoch,
                    organ_names=ORGAN_NAMES,
                    sample_idx=keys[b],
                    base_dir=export_dir,
                    abnormal_threshold=thresholds,
                )

        if step % 20 == 0 or step == len(loader):
            print(f"  {step}/{len(loader)} batches", flush=True)

    return np.concatenate(probs), np.concatenate(labels), names


def threshold_at_roc_corner(y_true, y_prob, fallback=0.5):
    """Threshold at the ROC point closest to (0, 1); ties prefer higher TPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if thresholds.size == 0:
        return fallback
    distance = np.sqrt(fpr ** 2 + (1.0 - tpr) ** 2)
    candidates = np.flatnonzero(np.isclose(distance, distance.min()))
    best = candidates[np.argmax(tpr[candidates])]
    return float(thresholds[best])


def evaluate(probs, labels, threshold_source, positive_class, ckpt_thresholds):
    """Per-disease metrics plus macro averages over the evaluable channels."""
    if positive_class == "absent":
        # Score "no finding" as the positive class: both labels and scores flip.
        probs, labels = 1.0 - probs, 1.0 - labels

    rows, skipped = [], []
    for idx, name in enumerate(DISEASE_NAMES):
        y_true = labels[:, idx].astype(int)
        y_prob = probs[:, idx]

        # A channel with one class present carries no signal: AUROC is undefined and
        # a confusion matrix would collapse. Record it and leave it out of the macro.
        if np.unique(y_true).size < 2:
            skipped.append(name)
            rows.append({
                "Disease": name, "Threshold": np.nan, "AUROC": np.nan,
                "Precision": np.nan, "Recall": np.nan, "F1": np.nan,
                "Accuracy": np.nan, "Positives": int(y_true.sum()), "N": len(y_true),
                "Evaluable": False,
            })
            continue

        if threshold_source == "checkpoint":
            threshold = float(ckpt_thresholds[idx])
        else:
            threshold = threshold_at_roc_corner(y_true, y_prob)

        y_pred = (y_prob > threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        rows.append({
            "Disease": name,
            "Threshold": threshold,
            "AUROC": roc_auc_score(y_true, y_prob),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": (tp + tn) / len(y_true),
            "Positives": int(y_true.sum()),
            "N": len(y_true),
            "Evaluable": True,
        })

    table = pd.DataFrame(rows)
    evaluable = table[table["Evaluable"]]
    macro = {
        "AUROC": float(evaluable["AUROC"].mean()),
        "Precision": float(evaluable["Precision"].mean()),
        "Recall": float(evaluable["Recall"].mean()),
        "F1": float(evaluable["F1"].mean()),
        "Accuracy": float(evaluable["Accuracy"].mean()),
        "n_diseases_averaged": int(len(evaluable)),
        "excluded_diseases": skipped,
    }
    return table, macro


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the EXACT Y-Mamba backbone on a preprocessed CT cohort.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5", required=True,
                        help="Preprocessed HDF5 test set")
    parser.add_argument("--checkpoint", required=True,
                        help="Backbone weights, e.g. checkpoints/01_pretrain/ymamba_pretrain_best.pth")
    parser.add_argument("--output-dir", default="results/test",
                        help="Directory for per-disease metrics, predictions and summary")
    parser.add_argument("--gpu", default="0", help="Value for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threshold-source", choices=["checkpoint", "fit-on-test"],
                        default="checkpoint",
                        help="Where per-disease decision thresholds come from")
    parser.add_argument("--positive-class", choices=["present", "absent"],
                        default="present",
                        help="Which class counts as positive for F1 / accuracy")
    parser.add_argument("--export-aamaps", default=None,
                        help="Also write voxel-level AAmaps under "
                             "<dir>/prediction_heatmaps/epoch_<N>/<study>/ , the input "
                             "expected by EXACT_ClassFinetune and EXACT-Seg")
    parser.add_argument("--reproduce-paper", action="store_true",
                        help="Use the convention behind the published tables "
                             "(equivalent to --threshold-source fit-on-test "
                             "--positive-class absent)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.reproduce_paper:
        args.threshold_source = "fit-on-test"
        args.positive_class = "absent"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = DiagnosisH5Dataset(args.h5)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    print(f"Cohort           : {args.h5}")
    print(f"Studies          : {len(dataset)}  (CT shape {dataset.ct_shape})")
    print(f"Labels           : {dataset.label_key}")
    if dataset.unlabelled:
        print("Unlabelled       : "
              + ", ".join(DISEASE_NAMES[i] for i in dataset.unlabelled))
    print(f"Threshold source : {args.threshold_source}")
    print(f"Positive class   : disease {args.positive_class}")

    model, ckpt_thresholds, epoch = load_model(args.checkpoint, device)
    print(f"Checkpoint       : {args.checkpoint} (epoch {epoch})")

    if args.threshold_source == "checkpoint":
        if ckpt_thresholds is None:
            raise SystemExit(
                "This checkpoint has no 'best_thresholds' entry. Re-run with "
                "--threshold-source fit-on-test, noting that thresholds fitted on "
                "the test set are optimistically biased."
            )
        if ckpt_thresholds.size != 18:
            raise SystemExit(
                f"Expected 18 stored thresholds, found {ckpt_thresholds.size}."
            )

    if args.threshold_source == "fit-on-test":
        print("\nNote: thresholds are fitted on this test set, so F1 and accuracy "
              "are optimistically biased. AUROC is unaffected.")
    if args.positive_class == "absent":
        print("Note: F1 and accuracy score disease *absence* as the positive class; "
              "on these negative-dominated cohorts that yields markedly higher "
              "values than the usual convention.")

    print("\nRunning inference...")
    if args.export_aamaps:
        os.makedirs(args.export_aamaps, exist_ok=True)
        print(f"Exporting AAmaps to {args.export_aamaps}/prediction_heatmaps/"
              f"epoch_{epoch or 0}/")

    probs, labels, names = predict(
        model, loader, device,
        export_dir=args.export_aamaps,
        thresholds=ckpt_thresholds,
        epoch=epoch or 0,
    )

    table, macro = evaluate(probs, labels, args.threshold_source,
                            args.positive_class, ckpt_thresholds)

    predictions = pd.DataFrame({"StudyName": names})
    for idx, name in enumerate(DISEASE_NAMES):
        predictions[f"Prob_{name}"] = probs[:, idx]
        predictions[f"GT_{name}"] = labels[:, idx].astype(int)

    metrics_path = os.path.join(args.output_dir, "disease_metrics.csv")
    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")

    table.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    with open(summary_path, "w") as f:
        json.dump({
            "cohort": args.h5,
            "checkpoint": args.checkpoint,
            "n_studies": len(dataset),
            "threshold_source": args.threshold_source,
            "positive_class": args.positive_class,
            "macro": macro,
        }, f, indent=2)

    pd.set_option("display.width", 140)
    print("\n" + table.drop(columns=["Evaluable"]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nMacro over {macro['n_diseases_averaged']} of {len(DISEASE_NAMES)} findings"
          f"  AUROC {macro['AUROC']:.4f}"
          f"  F1 {macro['F1']:.4f}"
          f"  Accuracy {macro['Accuracy']:.4f}")
    if macro["excluded_diseases"]:
        print("Excluded (no positive or no negative case in this cohort): "
              + ", ".join(macro["excluded_diseases"]))
    print(f"\nWritten to {args.output_dir}/"
          f" (disease_metrics.csv, predictions.csv, summary.json)")


if __name__ == "__main__":
    main()
