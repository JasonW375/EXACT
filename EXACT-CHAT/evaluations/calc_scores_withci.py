#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute multi-label classification metrics with 95% confidence intervals.
Modified version: emits "value [ci_lower, ci_upper]" for every metric.
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Silence sklearn's UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, confidence_level=0.95, random_state=42):
    """
    Compute a confidence interval for a metric via the bootstrap.
    
    Args:
        y_true: ground-truth labels
        y_pred: predicted labels
        metric_func: metric function to evaluate
        n_bootstraps: number of bootstrap resamples
        confidence_level: confidence level
        random_state: random seed

    Returns:
        point_estimate: point estimate of the metric
        ci_lower: lower bound of the confidence interval
        ci_upper: upper bound of the confidence interval
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    
    # Point estimate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point_estimate = metric_func(y_true, y_pred)
    
    # Bootstrap resampling
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = rng.randint(0, n_samples, n_samples)
        
        # Require at least two classes
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = metric_func(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        except:
            continue
    
    # Fall back to the point estimate if every resample failed
    if len(bootstrapped_scores) == 0:
        return point_estimate, point_estimate, point_estimate
    
    # Confidence interval
    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(bootstrapped_scores, alpha * 100)
    ci_upper = np.percentile(bootstrapped_scores, (1 - alpha) * 100)
    
    return point_estimate, ci_lower, ci_upper


def compute_metric_with_ci(y_true, y_pred, metric_func, metric_name, 
                           n_bootstraps=1000, confidence_level=0.95, random_state=42):
    """
    Compute a metric with its confidence interval and format it as a string.
    
    Args:
        y_true: ground-truth labels
        y_pred: predicted labels
        metric_func: metric function
        metric_name: metric name
        n_bootstraps: number of bootstrap resamples
        confidence_level: confidence level
        random_state: random seed
    
    Returns:
        Formatted string: "value [ci_lower, ci_upper]"
    """
    try:
        # Bind the zero_division behaviour required per metric
        if metric_name == 'precision':
            wrapped_func = lambda yt, yp: precision_score(yt, yp, zero_division=0)
        elif metric_name == 'recall':
            wrapped_func = lambda yt, yp: recall_score(yt, yp, zero_division=0)
        elif metric_name == 'f1':
            wrapped_func = lambda yt, yp: f1_score(yt, yp, zero_division=0)
        else:
            wrapped_func = metric_func
        
        point, ci_low, ci_high = bootstrap_ci(
            y_true, y_pred, wrapped_func, 
            n_bootstraps, confidence_level, random_state
        )
        
        # Format as "value [ci_lower, ci_upper]"
        formatted_str = f"{point:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
        
        return formatted_str
        
    except Exception as e:
        # Fall back to a zeroed result on failure
        if metric_name not in ['precision', 'recall', 'f1']:
            print(f"Warning: Could not compute {metric_name}: {e}")
        return "0.0000 [0.0000, 0.0000]"


def evaluate(pred_csv: Path, gt_csv: Path, out_json: Path, 
            n_bootstraps: int = 1000, confidence_level: float = 0.95, 
            random_state: int = 42):
    """
    Evaluate multi-label classification and report metrics with confidence intervals.
    
    Args:
        pred_csv: path to the predictions CSV
        gt_csv: path to the ground-truth CSV
        out_json: path to the output JSON
        n_bootstraps: number of bootstrap resamples
        confidence_level: confidence level
        random_state: random seed
    """
    
    print(f"Loading data...")
    print(f"  Predictions: {pred_csv}")
    print(f"  Ground truth: {gt_csv}")
    
    # Load data
    pred = pd.read_csv(pred_csv)
    gt   = pd.read_csv(gt_csv)

    # Normalise AccessionNo (strip file extensions)
    pred['AccessionNo'] = pred['AccessionNo'].str.replace('.npz',  '', regex=False)
    gt['AccessionNo']   = gt['AccessionNo'].str.replace('.nii.gz', '', regex=False)

    # Index by accession number
    pred.set_index('AccessionNo', inplace=True)
    gt.set_index('AccessionNo',   inplace=True)

    # Align to the ground-truth index and cast to integers
    pred = pred.reindex(gt.index).astype(int)

    print(f"\nData shape:")
    print(f"  Samples: {len(gt)}")
    print(f"  Classes: {len(gt.columns)}")
    print(f"\nComputing metrics with {n_bootstraps} bootstrap samples...")
    print(f"  Confidence level: {confidence_level * 100}%")
    print(f"  Output format: value [ci_lower, ci_upper]")
    
    results = {"per_pathology": []}
    
    # Raw values collected for the macro average
    all_prec_values, all_rec_values, all_f1_values, all_acc_values = [], [], [], []
    all_prec_ci_lows, all_rec_ci_lows, all_f1_ci_lows, all_acc_ci_lows = [], [], [], []
    all_prec_ci_highs, all_rec_ci_highs, all_f1_ci_highs, all_acc_ci_highs = [], [], [], []

    # Compute metrics for each pathology separately
    for i, col in enumerate(gt.columns, 1):
        print(f"  [{i}/{len(gt.columns)}] Processing: {col}", end='')
        
        y_true = gt[col].values
        y_pred = pred[col].values
        
        # Raw values (used for the macro average)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prec_val = precision_score(y_true, y_pred, zero_division=0)
            rec_val = recall_score(y_true, y_pred, zero_division=0)
            f1_val = f1_score(y_true, y_pred, zero_division=0)
            acc_val = accuracy_score(y_true, y_pred)
        
        # Formatted strings carrying the confidence interval
        prec_str = compute_metric_with_ci(
            y_true, y_pred, precision_score, 'precision',
            n_bootstraps, confidence_level, random_state
        )
        rec_str = compute_metric_with_ci(
            y_true, y_pred, recall_score, 'recall',
            n_bootstraps, confidence_level, random_state
        )
        f1_str = compute_metric_with_ci(
            y_true, y_pred, f1_score, 'f1',
            n_bootstraps, confidence_level, random_state
        )
        acc_str = compute_metric_with_ci(
            y_true, y_pred, accuracy_score, 'accuracy',
            n_bootstraps, confidence_level, random_state
        )
        
        # Confidence-interval bounds (used for the macro average)
        def extract_ci(s):
            """Extract value, ci_low and ci_high from the formatted string."""
            parts = s.split('[')
            value = float(parts[0].strip())
            ci_part = parts[1].rstrip(']').split(',')
            ci_low = float(ci_part[0].strip())
            ci_high = float(ci_part[1].strip())
            return value, ci_low, ci_high
        
        _, prec_ci_low, prec_ci_high = extract_ci(prec_str)
        _, rec_ci_low, rec_ci_high = extract_ci(rec_str)
        _, f1_ci_low, f1_ci_high = extract_ci(f1_str)
        _, acc_ci_low, acc_ci_high = extract_ci(acc_str)
        
        # Report whether this class received any positive prediction
        if y_pred.sum() == 0:
            print(" [WARNING: No predictions]")
        else:
            print(f" [OK {y_pred.sum()} predictions]")
        
        # Store the per-class result (compact format)
        results["per_pathology"].append({
            "name": col,
            "precision": prec_str,
            "recall": rec_str,
            "f1": f1_str,
            "accuracy": acc_str
        })
        
        # Collect raw values and CI bounds for the macro average
        all_prec_values.append(prec_val)
        all_rec_values.append(rec_val)
        all_f1_values.append(f1_val)
        all_acc_values.append(acc_val)
        
        all_prec_ci_lows.append(prec_ci_low)
        all_rec_ci_lows.append(rec_ci_low)
        all_f1_ci_lows.append(f1_ci_low)
        all_acc_ci_lows.append(acc_ci_low)
        
        all_prec_ci_highs.append(prec_ci_high)
        all_rec_ci_highs.append(rec_ci_high)
        all_f1_ci_highs.append(f1_ci_high)
        all_acc_ci_highs.append(acc_ci_high)

    print(f"\nPer-class metrics computed")
    print(f"Computing macro-averaged metrics...")
    
    # Macro average
    macro_prec = sum(all_prec_values) / len(all_prec_values)
    macro_rec = sum(all_rec_values) / len(all_rec_values)
    macro_f1 = sum(all_f1_values) / len(all_f1_values)
    macro_acc = sum(all_acc_values) / len(all_acc_values)
    
    macro_prec_ci_low = sum(all_prec_ci_lows) / len(all_prec_ci_lows)
    macro_prec_ci_high = sum(all_prec_ci_highs) / len(all_prec_ci_highs)
    
    macro_rec_ci_low = sum(all_rec_ci_lows) / len(all_rec_ci_lows)
    macro_rec_ci_high = sum(all_rec_ci_highs) / len(all_rec_ci_highs)
    
    macro_f1_ci_low = sum(all_f1_ci_lows) / len(all_f1_ci_lows)
    macro_f1_ci_high = sum(all_f1_ci_highs) / len(all_f1_ci_highs)
    
    macro_acc_ci_low = sum(all_acc_ci_lows) / len(all_acc_ci_lows)
    macro_acc_ci_high = sum(all_acc_ci_highs) / len(all_acc_ci_highs)

    # Format the macro-averaged results
    results["macro"] = {
        "precision": f"{macro_prec:.4f} [{macro_prec_ci_low:.4f}, {macro_prec_ci_high:.4f}]",
        "recall": f"{macro_rec:.4f} [{macro_rec_ci_low:.4f}, {macro_rec_ci_high:.4f}]",
        "f1": f"{macro_f1:.4f} [{macro_f1_ci_low:.4f}, {macro_f1_ci_high:.4f}]",
        "accuracy": f"{macro_acc:.4f} [{macro_acc_ci_low:.4f}, {macro_acc_ci_high:.4f}]"
    }

    # Write the results
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nClassification metrics -> {out_json}")
    
    # Summary
    print(f"\nMacro-averaged Results:")
    print(f"  Precision: {results['macro']['precision']}")
    print(f"  Recall:    {results['macro']['recall']}")
    print(f"  F1 Score:  {results['macro']['f1']}")
    print(f"  Accuracy:  {results['macro']['accuracy']}")
    
    # Count the classes the model actually predicts
    classes_with_predictions = sum(1 for val in all_prec_values if val > 0) + \
                              sum(1 for val in all_rec_values if val > 0)
    classes_with_predictions = min(classes_with_predictions, len(gt.columns))
    
    print(f"\nClasses with predictions: {classes_with_predictions}/{len(gt.columns)}")
    if classes_with_predictions < len(gt.columns) / 2:
        print(f"   WARNING: The model appears to predict very few positive samples!")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description='Compute classification metrics with 95% CI (compact format)'
    )
    ap.add_argument("--pred_csv", required=True, help='Path to predictions CSV')
    ap.add_argument("--gt_csv",   required=True, help='Path to ground truth CSV')
    ap.add_argument("--out_json", required=True, help='Path to output JSON')
    ap.add_argument("--n_bootstraps", type=int, default=1000,
                   help='Number of bootstrap samples (default: 1000)')
    ap.add_argument("--confidence_level", type=float, default=0.95,
                   help='Confidence level (default: 0.95)')
    ap.add_argument("--random_state", type=int, default=42,
                   help='Random seed for reproducibility (default: 42)')
    args = ap.parse_args()

    evaluate(
        Path(args.pred_csv), 
        Path(args.gt_csv), 
        Path(args.out_json),
        args.n_bootstraps,
        args.confidence_level,
        args.random_state
    )