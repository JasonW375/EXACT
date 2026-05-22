import os
import argparse
import numpy as np
import nibabel as nib
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

PRED_THRESHOLD = 0.5
GT_THRESHOLD = 0.0

# Default data paths (can be overridden by command-line arguments)
gt_dir = "/path/to/%%%/visualization_mask"
pred_dir = "/path/to/%%%/overlaid_heatmaps_covidfull"

# Output path
out_all = f"{pred_dir}/dice_scores_all.csv"


def _strip_nii_suffix(name: str) -> str:
    base = os.path.basename(name)
    if base.endswith(".nii.gz"):
        return base[:-7].strip()
    if base.endswith(".nii"):
        return base[:-4].strip()
    return os.path.splitext(base)[0].strip()


def collect_pred_map(pred_dir):
    """
    Return:
        {normalized_sample_id (with '_overlaid_heatmap' removed): full_prediction_file_path}
    """
    pred_map = {}
    for f in os.listdir(pred_dir):
        if not f.endswith(".nii.gz"):
            continue
        sid_full = _strip_nii_suffix(f)
        sid = sid_full.replace("_overlaid_heatmap", "")
        pred_map[sid] = os.path.join(pred_dir, f)
    return pred_map


def calculate_dice_per_sample(
    pred: np.ndarray,
    gt: np.ndarray,
    pred_threshold: float = 0.5,
    gt_threshold: float = 0.0
) -> float:
    pred_flat = np.asarray(pred, dtype=np.float32).ravel()
    gt_flat = np.asarray(gt, dtype=np.float32).ravel()

    valid = np.isfinite(pred_flat) & np.isfinite(gt_flat)
    if not np.all(valid):
        pred_flat = pred_flat[valid]
        gt_flat = gt_flat[valid]

    pred_bin = pred_flat > pred_threshold
    gt_bin = gt_flat > gt_threshold

    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()

    # If both masks are empty, define Dice as 1.0
    if pred_sum == 0 and gt_sum == 0:
        return 1.0

    denom = pred_sum + gt_sum
    if denom == 0:
        return np.nan

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    return float(2.0 * intersection / denom)


def _bootstrap_mean_ci(
    values: List[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator = None
) -> Tuple[float, float, float]:
    """
    Compute mean and (low, high) confidence interval via bootstrap.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(arr.mean())

    if arr.size == 1:
        return mean, float("nan"), float("nan")

    if rng is None:
        rng = np.random.default_rng()

    boots = np.empty(n_boot, dtype=np.float64)
    n = arr.size
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boots[i] = sample.mean()

    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(boots, alpha))
    high = float(np.quantile(boots, 1.0 - alpha))
    return mean, low, high


def _compute_one_sample(
    sid: str,
    pred_map: dict,
    gt_root: str,
    pred_threshold: float,
    gt_threshold: float
):
    """
    Read one sample and compute Dice.

    Returns:
        (sid, score or np.nan, err_msg or None)
    """
    gt_path = os.path.join(gt_root, f"{sid}.nii.gz")
    pred_path = pred_map.get(sid)

    if not os.path.exists(gt_path):
        return sid, np.nan, f"Missing GT: {sid}"
    if (not pred_path) or (not os.path.exists(pred_path)):
        return sid, np.nan, f"Missing prediction: {sid}"

    try:
        gt = nib.load(gt_path).get_fdata()
        pred = nib.load(pred_path).get_fdata()

        if gt.shape != pred.shape:
            return sid, np.nan, f"Shape mismatch for {sid}: GT {gt.shape} vs Pred {pred.shape}"

        score = calculate_dice_per_sample(
            pred,
            gt,
            pred_threshold=pred_threshold,
            gt_threshold=gt_threshold,
        )

        if np.isnan(score):
            return sid, np.nan, f"{sid}: Dice = NaN."

        return sid, float(score), None

    except Exception as e:
        return sid, np.nan, f"Error processing {sid}: {e}"


def eval_dice_per_sample_mean(
    sample_ids,
    outfile,
    pred_map,
    gt_root,
    pred_threshold=0.5,
    gt_threshold=0.0,
    max_workers=os.cpu_count()
):
    """
    Compute Dice for each sample in parallel, write CSV, and return the mean.
    """
    if not sample_ids:
        print("No samples, skipped.")
        return np.nan

    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    scores = []
    used = 0

    with open(outfile, "w", encoding="utf-8") as rf:
        rf.write("SampleID,Dice\n")

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _compute_one_sample,
                    sid,
                    pred_map,
                    gt_root,
                    pred_threshold,
                    gt_threshold,
                ): sid
                for sid in sample_ids
            }

            for fut in as_completed(futures):
                sid, score, err = fut.result()
                if np.isnan(score):
                    rf.write(f"{sid},NaN\n")
                    if err:
                        print(err)
                else:
                    rf.write(f"{sid},{score:.6f}\n")
                    scores.append(score)
                    used += 1

        if used > 0:
            mean_score, ci_low, ci_high = _bootstrap_mean_ci(scores, n_boot=1000, ci=0.95)
            rf.write(f"MEAN,{mean_score:.6f}\n")
            rf.write(f"CI95_low,{ci_low if not np.isnan(ci_low) else float('nan')}\n")
            rf.write(f"CI95_high,{ci_high if not np.isnan(ci_high) else float('nan')}\n")
            print(
                f"Mean Dice over {used} valid samples = "
                f"{mean_score:.4f} (95% CI: {ci_low:.4f} - {ci_high:.4f})"
            )
        else:
            rf.write("MEAN,NaN\n")
            rf.write("CI95_low,NaN\n")
            rf.write("CI95_high,NaN\n")
            print("No valid samples for mean Dice calculation.")

    return np.mean(scores) if used > 0 else np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Compute voxel-level Dice for all samples"
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="Prediction directory, overrides default"
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="GT directory, overrides default"
    )
    parser.add_argument(
        "--pred_threshold",
        type=float,
        default=PRED_THRESHOLD,
        help="Threshold used to binarize prediction values for Dice calculation"
    )
    parser.add_argument(
        "--gt_threshold",
        type=float,
        default=GT_THRESHOLD,
        help="Threshold used to binarize GT values for Dice calculation"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers"
    )
    args = parser.parse_args()

    gt_d = args.gt_dir if args.gt_dir else gt_dir
    pred_d = args.pred_dir if args.pred_dir else pred_dir
    out_csv = os.path.join(pred_d, "dice_scores_all_per_sample_mean.csv")

    if not os.path.isdir(gt_d):
        raise FileNotFoundError(f"GT directory does not exist: {gt_d}")
    if not os.path.isdir(pred_d):
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_d}")

    pred_map = collect_pred_map(pred_d)
    pred_ids = sorted(pred_map.keys())
    if not pred_ids:
        raise RuntimeError("No .nii.gz files were found in the prediction directory.")

    print("Running Dice evaluation on all samples.")
    mean_score = eval_dice_per_sample_mean(
        pred_ids,
        out_csv,
        pred_map,
        gt_d,
        pred_threshold=args.pred_threshold,
        gt_threshold=args.gt_threshold,
        max_workers=args.max_workers,
    )
    print(f"Sample-level mean Dice (all samples): {mean_score}")
    print(f"Results saved to: {out_csv}")


if __name__ == "__main__":
    main()