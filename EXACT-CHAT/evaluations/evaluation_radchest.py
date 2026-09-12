#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for RAD-ChestCT: classification metrics only.

RAD-ChestCT ships no reference reports, so only the classification stage runs
(no CRG, no NLG). Two of the 18 CT-RATE abnormality columns are placeholders in
this cohort and are dropped from both the predictions and the ground truth
before scoring, leaving 16 classes:

    Coronary artery wall calcification
    Mosaic attenuation pattern

Run prepare_eval_files.py first: it writes the result_transformat.json and
ground_truth.csv that this script consumes.

Example:

    python evaluation_radchest.py \
        --prediction_jsons out_radchest.json \
        --base_dir radchest_workdir
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _parse_cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prediction_jsons", nargs="+", required=True,
                    help="One or more EXACT-CHAT inference output JSONs to evaluate.")
    ap.add_argument("--base_dir", required=True, type=Path,
                    help="Directory containing per-prediction subfolders with prepared "
                         "ground_truth.csv and result_transformat.json.")
    ap.add_argument("--code_dir", type=Path, default=Path(__file__).resolve().parent,
                    help="Directory of the evaluation scripts (defaults to this file's dir).")
    return ap.parse_args()


_ARGS = _parse_cli()
INPUT_FILES = [Path(p) for p in _ARGS.prediction_jsons]
BASE_DIR = _ARGS.base_dir
CODE_DIR = _ARGS.code_dir

# Only these two stages are needed without reference reports.
INFER_SCRIPT = CODE_DIR / "infer.py"
CLS_SCRIPT = CODE_DIR / "calc_scores.py"

# Placeholder columns, excluded from both sides before scoring.
EXCLUDED_CLASSES = [
    "Coronary artery wall calcification",
    "Mosaic attenuation pattern",
]


def extract_checkpoint_number(filename: str) -> int:
    """Pull the checkpoint number out of a prediction filename, for reporting."""
    match = re.search(r'checkpoint(\d+)', filename)
    return int(match.group(1)) if match else 0


def get_output_dir_name(json_file_path: Path) -> str:
    """Derive the prepared-files directory name, matching prepare_eval_files.py."""
    return json_file_path.stem.replace(" ", "_")


def run(script: Path, *args):
    """Run one of the scoring sub-scripts, forwarding its output."""
    cmd = [sys.executable, str(script), *map(str, args)]
    print(">>", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: sub-script failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        raise


def load(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def check_required_files(prep_dir: Path):
    """Return the prepared files that prepare_eval_files.py did not write."""
    required_files = [
        "result_transformat.json",  # reports, in the nesting infer.py reads
        "ground_truth.csv",         # reference labels, for the classification metrics
    ]
    return [name for name in required_files if not (prep_dir / name).exists()]


def remove_excluded_columns(csv_path: Path, output_path: Path, excluded_classes: list) -> int:
    """Copy a label CSV without the placeholder columns. Returns how many were dropped."""
    df = pd.read_csv(csv_path)
    original_cols = len(df.columns)

    cols_to_drop = [col for col in excluded_classes if col in df.columns]
    if cols_to_drop:
        print(f"   dropping: {', '.join(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)

    df.to_csv(output_path, index=False)
    print(f"   columns: {original_cols} -> {len(df.columns)}")
    return len(cols_to_drop)


def check_csv_content(csv_path: Path, excluded_classes: list = None):
    """Summarise a label CSV, so an all-zero or empty table is caught early."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Label CSV summary ({csv_path.name}):")
        print(f"   rows: {len(df)}")

        if len(df) == 0:
            print("   WARNING: the CSV is empty")
            return

        label_cols = [col for col in df.columns if col != 'AccessionNo']
        if excluded_classes:
            label_cols = [col for col in label_cols if col not in excluded_classes]
            print(f"   label columns: {len(label_cols)} (placeholders excluded)")
        else:
            print(f"   label columns: {len(label_cols)}")

        positive_counts = {col: int((df[col] == 1).sum()) for col in label_cols}
        total_positives = sum(positive_counts.values())
        print(f"   positive cells: {total_positives}")

        if total_positives == 0:
            print("   WARNING: every label is 0, classification metrics will be degenerate")

        positive_classes = [(k, v) for k, v in positive_counts.items() if v > 0]
        if positive_classes:
            print("   most frequent classes:")
            for cls, count in sorted(positive_classes, key=lambda x: x[1], reverse=True)[:5]:
                print(f"     {cls}: {count}")

    except Exception as e:
        print(f"   ERROR: could not read the CSV: {e}")


def evaluate_single_file(input_file: Path, file_index: int, total_files: int):
    """Score one prediction file. Returns True on success."""
    checkpoint_num = extract_checkpoint_number(input_file.name)

    print(f"\n{'=' * 60}")
    print(f"File [{file_index}/{total_files}]: {input_file.name}")
    if checkpoint_num > 0:
        print(f"Checkpoint: {checkpoint_num}")
    print(f"{'=' * 60}")
    print(f"Excluded placeholder classes: {', '.join(EXCLUDED_CLASSES)}")

    prep_dir = BASE_DIR / get_output_dir_name(input_file)
    print(f"Prepared files directory: {prep_dir}")

    if not prep_dir.exists():
        print(f"ERROR: prepared files directory does not exist: {prep_dir}")
        print("Run prepare_eval_files.py first.")
        return False

    missing_files = check_required_files(prep_dir)
    if missing_files:
        print("ERROR: the following prepared files are missing:")
        for f in missing_files:
            print(f"   {f}")
        print("Run prepare_eval_files.py first.")
        return False

    # Inputs written by prepare_eval_files.py.
    pred_json_transformed = prep_dir / "result_transformat.json"
    gt_csv = prep_dir / "ground_truth.csv"

    # Intermediates: the full 18-class tables, and the 16-class ones actually scored.
    csv_pred_full = prep_dir / "inferred_full.csv"
    csv_pred = prep_dir / "inferred.csv"
    gt_csv_filtered = prep_dir / "ground_truth_filtered.csv"

    # Outputs.
    cls_json = prep_dir / "classification_scores.json"
    final_json = prep_dir / "metrics.json"

    print(f"Predictions:      {pred_json_transformed.name}")
    print(f"Ground truth CSV: {gt_csv.name}")

    try:
        # 1. Label the generated reports with the RadBERT classifier.
        print("\n1. Running RadBERT inference...")
        run(INFER_SCRIPT,
            "--input_json", pred_json_transformed,
            "--model_path", CODE_DIR / "roberta_local" / "RadBertClassifier.pth",
            "--out_csv", csv_pred_full)
        print(f"   done -> {csv_pred_full.name}")

        # 2. Drop the placeholder columns from both sides.
        print("\n2. Removing placeholder columns...")
        print("\nPredictions, before:")
        check_csv_content(csv_pred_full)
        print("\nGround truth, before:")
        check_csv_content(gt_csv)

        print(f"\n{csv_pred_full.name} -> {csv_pred.name}")
        removed_pred = remove_excluded_columns(csv_pred_full, csv_pred, EXCLUDED_CLASSES)

        print(f"{gt_csv.name} -> {gt_csv_filtered.name}")
        removed_gt = remove_excluded_columns(gt_csv, gt_csv_filtered, EXCLUDED_CLASSES)

        if removed_pred == 0 and removed_gt == 0:
            print("WARNING: no placeholder columns were found to remove")

        print("\nPredictions, after:")
        check_csv_content(csv_pred)
        print("\nGround truth, after:")
        check_csv_content(gt_csv_filtered)

        # 3. Multi-label classification scores over the remaining 16 classes.
        print("\n3. Computing classification scores...")
        run(CLS_SCRIPT,
            "--pred_csv", csv_pred,
            "--gt_csv", gt_csv_filtered,
            "--out_json", cls_json)
        print(f"   done -> {cls_json.name}")

        # 4. Write metrics.json.
        print("\n4. Saving final metrics...")
        cls_metrics = load(cls_json)

        combined = {
            "classification": cls_metrics,
            "metadata": {
                "checkpoint": checkpoint_num if checkpoint_num > 0 else "N/A",
                "input_file": input_file.name,
                "evaluation_type": "classification_only",
                "excluded_classes": EXCLUDED_CLASSES,
            },
        }

        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"\nMetrics written to: {final_json}")

        if cls_metrics:
            print("\nClassification metrics (placeholder classes excluded):")
            print("=" * 50)

            overall = {k: v for k, v in cls_metrics.items() if isinstance(v, (int, float))}
            per_class = {k: v for k, v in cls_metrics.items() if isinstance(v, dict)}

            if overall:
                print("\n   Overall:")
                for key, value in overall.items():
                    print(f"      {key}: {value:.4f}")

            if per_class:
                print("\n   Per class:")
                for metric_name, class_values in per_class.items():
                    print(f"\n   {metric_name}:")
                    for class_name, value in class_values.items():
                        if isinstance(value, (int, float)):
                            print(f"      {class_name}: {value:.4f}")

        return True

    except Exception as e:
        print(f"ERROR while evaluating: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_summary_report(successful_files: list, base_dir: Path):
    """Write evaluation_summary.json comparing every successfully scored file."""
    if not successful_files:
        return

    print(f"\n{'=' * 60}")
    print("Summary report")
    print(f"{'=' * 60}")

    summary_data = []

    for input_file in successful_files:
        metrics_file = base_dir / get_output_dir_name(input_file) / "metrics.json"
        if not metrics_file.exists():
            continue
        try:
            metrics = load(metrics_file)
        except Exception as e:
            print(f"WARNING: could not load {metrics_file}: {e}")
            continue

        checkpoint_num = extract_checkpoint_number(input_file.name)
        summary_item = {
            "checkpoint": checkpoint_num if checkpoint_num > 0 else "N/A",
            "filename": input_file.name,
        }

        for key, value in (metrics.get("classification") or {}).items():
            if isinstance(value, (int, float)):
                summary_item[f"cls_{key}"] = round(value, 4)
            elif isinstance(value, dict) and key.endswith('_per_class') and value:
                # Keep only the mean of a per-class block, not every class.
                numeric = [v for v in value.values() if isinstance(v, (int, float))]
                if numeric:
                    summary_item[f"cls_{key}_avg"] = round(sum(numeric) / len(numeric), 4)

        summary_data.append(summary_item)

    if not summary_data:
        return

    summary_file = base_dir / "evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"Summary written to: {summary_file}")

    print("\nClassification comparison (placeholder classes excluded):")
    print("-" * 100)
    print(f"{'File':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 100)

    for item in summary_data:
        filename = item['filename']
        if len(filename) > 35:
            filename = filename[:32] + "..."

        def fmt(key):
            return f"{item[key]:.3f}" if key in item else "N/A"

        print(f"{filename:<35} {fmt('cls_accuracy'):<12} {fmt('cls_precision'):<12} "
              f"{fmt('cls_recall'):<12} {fmt('cls_f1'):<12}")

    print("-" * 100)
    print("Full numbers: evaluation_summary.json")

    all_metrics = sorted({k for item in summary_data for k in item if k.startswith('cls_')})
    if all_metrics:
        print("\nAvailable classification metrics:")
        for metric in all_metrics:
            print(f"   {metric}")


def main():
    print("EXACT-CHAT evaluation (RAD-ChestCT, classification only)")
    print(f"Base directory:     {BASE_DIR}")
    print(f"Evaluation scripts: {CODE_DIR}")
    print(f"Excluded classes:   {', '.join(EXCLUDED_CLASSES)}")

    if not CODE_DIR.exists():
        print(f"ERROR: evaluation script directory does not exist: {CODE_DIR}")
        return

    missing_scripts = [s for s in (INFER_SCRIPT, CLS_SCRIPT) if not s.exists()]
    if missing_scripts:
        print("ERROR: the following evaluation scripts are missing:")
        for script in missing_scripts:
            print(f"   {script}")
        return

    input_files = INPUT_FILES
    missing_files = [f for f in input_files if not f.exists()]
    if missing_files:
        print("ERROR: the following input files do not exist:")
        for f in missing_files:
            print(f"   {f}")
        return

    print(f"\nEvaluating {len(input_files)} prediction file(s):")
    for i, f in enumerate(input_files, 1):
        checkpoint_num = extract_checkpoint_number(f.name)
        if checkpoint_num > 0:
            print(f"   {i}. {f.name} (checkpoint-{checkpoint_num})")
        else:
            print(f"   {i}. {f.name}")

    successful_files = []
    failed = 0

    for i, input_file in enumerate(input_files, 1):
        if evaluate_single_file(input_file, i, len(input_files)):
            successful_files.append(input_file)
        else:
            failed += 1

    generate_summary_report(successful_files, BASE_DIR)

    print(f"\n{'=' * 60}")
    print("Done")
    print(f"{'=' * 60}")
    print(f"Total:     {len(input_files)}")
    print(f"Succeeded: {len(successful_files)}")
    print(f"Failed:    {failed}")

    if successful_files:
        print("\nResults:")
        for input_file in successful_files:
            print(f"   {BASE_DIR / get_output_dir_name(input_file)}/metrics.json")

        print("\nFiles written per prediction:")
        print("   inferred_full.csv          predicted labels, all 18 classes")
        print("   inferred.csv               predicted labels, placeholders dropped")
        print("   ground_truth_filtered.csv  reference labels, placeholders dropped")
        print("   classification_scores.json per-class and overall metrics")
        print("   metrics.json               merged result")
        print("   evaluation_summary.json    comparison across prediction files")

    if failed:
        print("\nIf a file failed, check in this order:")
        print("1. prepare_eval_files.py ran for that prediction file")
        print("2. ground_truth.csv is not all zeros")
        print("3. roberta_local/RadBertClassifier.pth exists (see download_RoBERTa_tokenizer.py)")


if __name__ == "__main__":
    main()
