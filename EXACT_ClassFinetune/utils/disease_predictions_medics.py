import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
import os


def choose_best_threshold_lefttop(y_true: np.ndarray, y_prob: np.ndarray, fallback: float = 0.5):
    """
    Pick the ROC point closest to (0, 1); ties break on max TPR, then min FPR.
    
    Args:
        y_true: ground-truth labels
        y_prob: predicted probabilities
        fallback: threshold to use for a degenerate curve
    
    Returns:
        (threshold, auc): the chosen threshold and the AUROC
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        if fpr.size == 0 or tpr.size == 0 or thresholds.size == 0:
            return fallback, np.nan
        dist = np.sqrt(fpr**2 + (1.0 - tpr)**2)
        best_dist = np.min(dist)
        idxs = np.where(np.isclose(dist, best_dist))[0]
        if idxs.size == 0:
            return fallback, roc_auc_score(y_true, y_prob)
        if idxs.size > 1:
            tpr_c = tpr[idxs]
            max_tpr = np.max(tpr_c)
            idxs = idxs[tpr_c >= max_tpr - 1e-12]
            if idxs.size > 1:
                fpr_c = fpr[idxs]
                min_fpr = np.min(fpr_c)
                idxs = idxs[fpr_c <= min_fpr + 1e-12]
        thr = float(thresholds[int(idxs[0])])
        auc = roc_auc_score(y_true, y_prob)
        return thr, float(auc)
    except Exception:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float('nan')
        return fallback, float(auc)


def load_prediction_data(csv_path: str):
    """
    Load a prediction CSV, accepting tabs or commas as the delimiter.
    
    Args:
        csv_path: path to the CSV
    
    Returns:
        df: the loaded DataFrame
    """
    # Sniff the delimiter from the header line
    with open(csv_path, 'r') as f:
        first_line = f.readline()
    sep = '\t' if '\t' in first_line else ','
    
    df = pd.read_csv(csv_path, sep=sep)
    df = df.replace('', np.nan).dropna()
    
    return df


def extract_pred_gt_columns(df: pd.DataFrame):
    """
    Split the columns into predicted probabilities and ground truth.
    
    Args:
        df: input DataFrame
    
    Returns:
        (pred_cols, gt_cols): the Pred_ and GT_ column names
    """
    pred_cols = [col for col in df.columns if col.startswith('Pred_')]
    gt_cols = [col for col in df.columns if col.startswith('GT_')]
    
    return pred_cols, gt_cols


def calculate_optimal_thresholds(df: pd.DataFrame, pred_cols: list, gt_cols: list):
    """
    Fit one threshold per finding on the set being scored.
    
    Args:
        df: input DataFrame
        pred_cols: Pred_ column names
        gt_cols: GT_ column names
    
    Returns:
        thresholds: Pred_ column name -> fitted threshold
    """
    thresholds = {}
    
    for pred_col, gt_col in zip(pred_cols, gt_cols):
        y_prob = df[pred_col].astype(float).values
        y_true = df[gt_col].astype(int).values
        thr, _ = choose_best_threshold_lefttop(y_true, y_prob)
        thresholds[pred_col] = thr
    
    return thresholds


def generate_binary_predictions(df: pd.DataFrame, pred_cols: list, thresholds: dict):
    """
    Binarise the probabilities at the per-finding thresholds.
    
    Args:
        df: input DataFrame
        pred_cols: Pred_ column names
        thresholds: Pred_ column name -> threshold
    
    Returns:
        results_df: one row per study, one binary column per finding
    """
    results = {"VolumeName": df["VolumeName"].values}
    
    for pred_col in pred_cols:
        y_prob = df[pred_col].astype(float).values
        thr = thresholds[pred_col]
        pred_bin = (y_prob >= thr).astype(int)
        
        disease_name = pred_col.replace('Pred_', '')
        results[disease_name] = pred_bin
    
    results_df = pd.DataFrame(results)
    return results_df


def calculate_metrics(df: pd.DataFrame, pred_cols: list, gt_cols: list,
                      thresholds: dict, positive_class: str = "present"):
    """
    Score every finding at its fitted threshold.
    
    Args:
        df: input DataFrame
        pred_cols: Pred_ column names
        gt_cols: GT_ column names
        thresholds: Pred_ column name -> threshold
    
    Returns:
        metrics_df: one row per finding
    """
    metrics = {}
    
    for pred_col, gt_col in zip(pred_cols, gt_cols):
        y_prob = df[pred_col].astype(float).values
        y_true = df[gt_col].astype(int).values
        thr = thresholds[pred_col]
        
        # Binarise at the fitted threshold
        pred_bin = (y_prob >= thr).astype(int)
        
        # AUROC is threshold-free
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
        
        # Which class counts as positive changes precision/recall/F1
        # but not AUROC or accuracy, so those use the raw labels.
        if positive_class == "absent":
            y_eval, pred_eval = 1 - y_true, 1 - pred_bin
        else:
            y_eval, pred_eval = y_true, pred_bin

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_eval, pred_eval, average='binary', zero_division=0
        )
        
        disease_name = pred_col.replace('Pred_', '')
        metrics[disease_name] = {
            "Threshold": thr,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy_score(y_true, pred_bin),
        }
    
    metrics_df = pd.DataFrame(metrics).T
    return metrics_df


def process_disease_prediction(csv_path: str, output_dir: str = None,
                               positive_class: str = "present"):
    """
    Threshold, score, and write disease_predictions.csv / disease_metrics.csv.
    
    Args:
        csv_path: pred.csv from train_heatmap.py --task test
        output_dir: where to write (default: alongside the input CSV)
    
    Returns:
        (results_df, metrics_df): binary predictions and per-finding metrics
    """
    # Default to writing alongside the input CSV
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {csv_path}")
    df = load_prediction_data(csv_path)
    print(f"{len(df)} studies")
    
    print("Extracting prediction and ground-truth columns")
    pred_cols, gt_cols = extract_pred_gt_columns(df)
    print(f"{len(pred_cols)} findings")
    
    print("Fitting per-finding thresholds")
    thresholds = calculate_optimal_thresholds(df, pred_cols, gt_cols)
    
    print("Binarising predictions")
    results_df = generate_binary_predictions(df, pred_cols, thresholds)
    
    print("Scoring")
    metrics_df = calculate_metrics(df, pred_cols, gt_cols, thresholds,
                                   positive_class)
    
    # Save
    pred_save_path = os.path.join(output_dir, 'disease_predictions.csv')
    metrics_save_path = os.path.join(output_dir, 'disease_metrics.csv')
    
    results_df.to_csv(pred_save_path, index=False)
    metrics_df.to_csv(metrics_save_path)
    
    print(f"\nWrote {pred_save_path}")
    print(f"Wrote {metrics_save_path}")
    
    print("\n=== Per-disease metrics ===")
    with pd.option_context('display.width', 200,
                            'display.max_columns', None):
        print(metrics_df.round(4))

    macro = metrics_df.mean(numeric_only=True)
    print("\nMacro over %d findings  AUROC %.4f  F1 %.4f  Accuracy %.4f"
          % (len(metrics_df), macro["AUC"], macro["F1"],
             macro["Accuracy"]))
    
    return results_df, metrics_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score per-disease predictions and write disease_metrics.csv.")
    parser.add_argument("--pred-csv", required=True,
                        help="pred.csv written by train_heatmap.py --task test")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: alongside --pred-csv)")
    parser.add_argument("--positive-class", default="present",
                        choices=["present", "absent"],
                        help="Positive class for precision/recall/F1. The "
                             "manuscript reports 'absent'; AUROC and accuracy "
                             "are unaffected either way.")
    cli_args = parser.parse_args()

    out_dir = cli_args.output_dir or os.path.dirname(
        os.path.abspath(cli_args.pred_csv))

    results_df, metrics_df = process_disease_prediction(
        cli_args.pred_csv, out_dir, cli_args.positive_class)
