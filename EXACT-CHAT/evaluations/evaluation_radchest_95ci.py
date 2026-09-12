#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master evaluation script for the RadChest dataset.
Classification metrics only: placeholder classes are excluded and all metrics carry 95% confidence intervals.
"""

import json, subprocess, sys, glob
from pathlib import Path
import re
import pandas as pd

# === CLI configuration ===
def _parse_cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prediction_jsons", nargs="+", required=True,
                    help="One or more EXACT-CHAT inference output JSONs to evaluate.")
    ap.add_argument("--base_dir", required=True, type=Path,
                    help="Directory containing per-prediction subfolders with prepared "
                         "ground_truth.json/csv and result_transformat.json.")
    ap.add_argument("--code_dir", type=Path, default=Path(__file__).resolve().parent,
                    help="Directory of the evaluation scripts (defaults to this file's dir).")
    return ap.parse_args()

_ARGS = _parse_cli()
MANUAL_INPUT_FILES = [Path(p) for p in _ARGS.prediction_jsons]
BASE_DIR = _ARGS.base_dir
CODE_DIR = _ARGS.code_dir
USE_FOLDER_MODE = False
FOLDER_PATH = None
FILE_PATTERN = None

# Paths to the evaluation helper scripts
INFER_SCRIPT = CODE_DIR / "infer.py"
CLS_SCRIPT   = CODE_DIR / "calc_scores_withci.py"  # CI-enabled classification scorer

# Bootstrap settings
N_BOOTSTRAPS = 1000        # number of bootstrap resamples
CONFIDENCE_LEVEL = 0.95    # confidence level (95% CI)
RANDOM_STATE = 42          # random seed

# === Placeholder classes to exclude ===
EXCLUDED_CLASSES = [
    "Coronary artery wall calcification",
    "Mosaic attenuation pattern"
]

def extract_checkpoint_number(filename: str) -> int:
    """Extract the checkpoint number from a filename so results can be ordered."""
    match = re.search(r'checkpoint(\d+)', filename)
    return int(match.group(1)) if match else 0

def find_prediction_files(folder_path: Path, pattern: str) -> list[Path]:
    """Find prediction files matching a glob pattern inside a folder."""
    print(f"Searching for prediction files in: {folder_path}")
    print(f"   Pattern: {pattern}")

    if not folder_path.exists():
        print(f"Folder does not exist: {folder_path}")
        return []

    if not folder_path.is_dir():
        print(f"Path is not a directory: {folder_path}")
        return []

    # Match files with the glob pattern
    files = list(folder_path.glob(pattern))

    if not files:
        print("   No files matched the pattern")
        print(f"   Check that files named like {pattern} exist in this folder")
        return []

    # Sort by checkpoint number so files are processed in order
    files.sort(key=lambda x: extract_checkpoint_number(x.name))

    print(f"Found {len(files)} prediction file(s):")
    for i, f in enumerate(files, 1):
        checkpoint_num = extract_checkpoint_number(f.name)
        if checkpoint_num > 0:
            print(f"   {i}. {f.name} (checkpoint-{checkpoint_num})")
        else:
            print(f"   {i}. {f.name}")

    return files

def get_input_files() -> list[Path]:
    """Return the list of input files for the current mode."""
    if USE_FOLDER_MODE:
        print("Using folder mode")
        return find_prediction_files(FOLDER_PATH, FILE_PATTERN)
    else:
        print("Using explicit file list mode")
        return MANUAL_INPUT_FILES

def get_output_dir_name(json_file_path: Path) -> str:
    """Derive the output directory name from the JSON filename (kept in sync with the prep script)."""
    name = json_file_path.stem
    return name.replace(" ", "_")

def run(script: Path, *args):
    """Run a helper script as a subprocess."""
    cmd = [sys.executable, str(script), *map(str, args)]
    print(">>", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Script failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        raise

def load(p: Path):
    """Load a JSON file."""
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def check_required_files(prep_dir: Path):
    """Return the prepared files that are required for classification-only evaluation."""
    required_files = [
        "result_transformat.json",  # format expected by infer.py
        "ground_truth.csv"          # reference labels for classification
    ]

    missing = []
    for filename in required_files:
        file_path = prep_dir / filename
        if not file_path.exists():
            missing.append(filename)

    return missing

def remove_excluded_columns(csv_path: Path, output_path: Path, excluded_classes: list):
    """Drop the given placeholder class columns from a CSV file.

    Args:
        csv_path: input CSV path
        output_path: output CSV path
        excluded_classes: class names to drop
    """
    df = pd.read_csv(csv_path)

    # Record the original column count
    original_cols = len(df.columns)

    # Drop the excluded columns when present
    cols_to_drop = [col for col in excluded_classes if col in df.columns]

    if cols_to_drop:
        print(f"   Dropping placeholder columns: {', '.join(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)

    # Write the filtered CSV
    df.to_csv(output_path, index=False)

    print(f"   Columns before: {original_cols}, after: {len(df.columns)}")

    return len(cols_to_drop)

def check_csv_content(csv_path: Path, excluded_classes: list = None):
    """Inspect a CSV file and report anything that could break classification."""
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV summary ({csv_path.name}):")
        print(f"   Rows: {len(df)}")

        if len(df) > 0:
            # Label columns are all columns except AccessionNo
            label_cols = [col for col in df.columns if col != 'AccessionNo']

            # Exclude placeholder classes from the statistics when provided
            if excluded_classes:
                label_cols = [col for col in label_cols if col not in excluded_classes]
                print(f"   Label columns: {len(label_cols)} (after excluding placeholder classes)")
            else:
                print(f"   Label columns: {len(label_cols)}")

            # Count positives per column
            positive_counts = {}
            for col in label_cols:
                if col in df.columns:
                    positive_count = (df[col] == 1).sum()
                    positive_counts[col] = positive_count

            # Report the positive-sample totals
            total_positives = sum(positive_counts.values())
            print(f"   Total positive samples: {total_positives}")

            if total_positives == 0:
                print("   Warning: all labels are 0, which can break classification evaluation")

            # Show the five classes with the most positives
            positive_classes = [(k, v) for k, v in positive_counts.items() if v > 0]
            if positive_classes:
                print("   Classes with positive samples (top 5):")
                for cls, count in sorted(positive_classes, key=lambda x: x[1], reverse=True)[:5]:
                    print(f"     {cls}: {count}")

        else:
            print("   Warning: the CSV file is empty")

    except Exception as e:
        print(f"   Could not analyse the CSV file: {e}")

def evaluate_single_file(input_file: Path, file_index: int, total_files: int):
    """Run the classification-only evaluation for a single input file."""
    checkpoint_num = extract_checkpoint_number(input_file.name)

    print(f"\n{'='*60}")
    print(f"Processing file [{file_index}/{total_files}]: {input_file.name}")
    if checkpoint_num > 0:
        print(f"Checkpoint: {checkpoint_num}")
    print(f"{'='*60}")
    print(f"Note: excluding placeholder classes {', '.join(EXCLUDED_CLASSES)}")

    # Locate the prepared files (same naming convention as the prep script)
    output_dir_name = get_output_dir_name(input_file)
    prep_dir = BASE_DIR / output_dir_name

    print(f"Prepared files directory: {prep_dir}")

    # Check that the prepared directory exists
    if not prep_dir.exists():
        print(f"Prepared file directory does not exist: {prep_dir}")
        print("Run the preparation script first to generate the required files.")
        return False

    # Check that the required files are present
    missing_files = check_required_files(prep_dir)
    if missing_files:
        print("Missing required files:")
        for f in missing_files:
            print(f"   {f}")
        print("Make sure the preparation script completed successfully.")
        return False

    # File paths: inputs produced by the preparation script
    pred_json_transformed = prep_dir / "result_transformat.json"
    gt_csv = prep_dir / "ground_truth.csv"

    # Intermediate files
    csv_pred_full = prep_dir / "inferred_full.csv"          # full predictions (placeholder columns still present)
    csv_pred = prep_dir / "inferred.csv"                    # filtered predictions (placeholder columns dropped)
    gt_csv_filtered = prep_dir / "ground_truth_filtered.csv" # filtered ground truth (placeholder columns dropped)

    # Output files (evaluation results)
    cls_json = prep_dir / "classification_scores.json"
    final_json = prep_dir / "metrics.json"

    print(f"Using transformed predictions: {pred_json_transformed.name}")
    print(f"Ground truth CSV: {gt_csv.name}")

    try:
        # 1. Inference -> CSV (emit the full CSV first)
        print("\n1. Running inference...")
        run(INFER_SCRIPT,
            "--input_json", pred_json_transformed,
            "--model_path", CODE_DIR / "roberta_local" / "RadBertClassifier.pth",
            "--out_csv", csv_pred_full)
        print(f"Inference finished, results written to: {csv_pred_full.name}")

        # 2. Drop the placeholder columns
        print("\n2. Removing placeholder columns...")

        # Inspect the raw predictions
        print("\nRaw prediction summary:")
        check_csv_content(csv_pred_full, excluded_classes=None)

        print("\nRaw ground truth summary:")
        check_csv_content(gt_csv, excluded_classes=None)

        # Drop placeholder columns from the predictions
        print(f"\nFiltering predictions: {csv_pred_full.name} -> {csv_pred.name}")
        removed_pred = remove_excluded_columns(csv_pred_full, csv_pred, EXCLUDED_CLASSES)

        # Drop placeholder columns from the ground truth
        print(f"Filtering ground truth: {gt_csv.name} -> {gt_csv_filtered.name}")
        removed_gt = remove_excluded_columns(gt_csv, gt_csv_filtered, EXCLUDED_CLASSES)

        if removed_pred > 0 or removed_gt > 0:
            print(f"Dropped {removed_pred} placeholder column(s)")
        else:
            print("No placeholder columns needed to be dropped")

        # Inspect the filtered files
        print("\nFiltered prediction summary:")
        check_csv_content(csv_pred, excluded_classes=None)

        print("\nFiltered ground truth summary:")
        check_csv_content(gt_csv_filtered, excluded_classes=None)

        # 3. Multi-label classification scores with 95% CI (using the filtered CSVs)
        print("\n3. Computing classification scores with 95% CI...")
        print("   Using the filtered CSVs (placeholder classes removed)")
        print(f"   Bootstrap samples: {N_BOOTSTRAPS}")
        print(f"   Confidence level: {CONFIDENCE_LEVEL * 100}%")
        run(CLS_SCRIPT,
            "--pred_csv", csv_pred,
            "--gt_csv", gt_csv_filtered,
            "--out_json", cls_json,
            "--n_bootstraps", str(N_BOOTSTRAPS),
            "--confidence_level", str(CONFIDENCE_LEVEL),
            "--random_state", str(RANDOM_STATE))
        print(f"Classification finished (with 95% CI), results written to: {cls_json.name}")

        # 4. Write the final result file
        print("\n4. Saving final metrics...")
        cls_metrics = load(cls_json)

        combined = {
            "classification": cls_metrics,
            "metadata": {
                "checkpoint": checkpoint_num if checkpoint_num > 0 else "N/A",
                "input_file": input_file.name,
                "evaluation_type": "classification_only_with_ci",
                "excluded_classes": EXCLUDED_CLASSES,
                "bootstrap_config": {
                    "n_bootstraps": N_BOOTSTRAPS,
                    "confidence_level": CONFIDENCE_LEVEL,
                    "random_state": RANDOM_STATE
                },
                "note": "Classification metrics only, placeholder classes excluded, 95% confidence intervals included"
            }
        }

        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"\nMetrics written to: {final_json}")

        # Print the detailed results (including confidence intervals)
        print("\nEvaluation results (95% CI, placeholder classes excluded):")
        print("=" * 80)

        if cls_metrics:
            print("Classification metrics:")

            # Macro-averaged metrics, when present
            if 'macro' in cls_metrics:
                print("\n   Macro average:")
                macro = cls_metrics['macro']
                for key in ['precision', 'recall', 'f1', 'accuracy']:
                    if key in macro:
                        value_str = macro[key]  # already a string like "0.0923 [0.0800, 0.1049]"
                        print(f"      {key:12s}: {value_str}")

            # Show the first five per-pathology entries, when present
            if 'per_pathology' in cls_metrics:
                print("\n   Per-class metrics (first 5):")
                per_path = cls_metrics['per_pathology'][:5]
                for item in per_path:
                    print(f"\n      {item['name']}:")
                    for key in ['precision', 'recall', 'f1', 'accuracy']:
                        if key in item:
                            value_str = item[key]  # already a string
                            print(f"         {key:12s}: {value_str}")

                if len(cls_metrics['per_pathology']) > 5:
                    print(f"\n      ... and {len(cls_metrics['per_pathology']) - 5} more classes")

        return True

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_summary_report(successful_files: list[Path], base_dir: Path):
    """Write a summary report across all successfully evaluated files (including confidence intervals)."""
    if not successful_files:
        return

    print(f"\n{'='*60}")
    print("Generating summary report")
    print(f"{'='*60}")

    summary_data = []

    for input_file in successful_files:
        output_dir_name = get_output_dir_name(input_file)
        metrics_file = base_dir / output_dir_name / "metrics.json"

        if metrics_file.exists():
            try:
                metrics = load(metrics_file)
                checkpoint_num = extract_checkpoint_number(input_file.name)

                # Collect the key metrics
                summary_item = {
                    "checkpoint": checkpoint_num if checkpoint_num > 0 else "N/A",
                    "filename": input_file.name
                }

                # Classification metrics (macro average)
                if "classification" in metrics and "macro" in metrics["classification"]:
                    macro = metrics["classification"]["macro"]
                    for key in ['precision', 'recall', 'f1', 'accuracy']:
                        if key in macro:
                            value_str = macro[key]  # e.g. "0.0923 [0.0800, 0.1049]"
                            summary_item[f"cls_{key}"] = value_str  # stored as a string as-is

                summary_data.append(summary_item)

            except Exception as e:
                print(f"Could not load {metrics_file}: {e}")

    # Write the summary report
    if summary_data:
        summary_file = base_dir / "evaluation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"Summary report written to: {summary_file}")

        # Print the comparison table (including confidence intervals)
        print("\nClassification performance (95% CI, placeholder classes excluded):")
        print("-" * 120)
        print(f"{'File':<35} {'Precision':<30} {'Recall':<30} {'F1':<30}")
        print("-" * 120)

        for item in summary_data:
            filename = item['filename'][:32] + "..." if len(item['filename']) > 35 else item['filename']

            # Metrics are already fully formatted strings
            def format_metric(key):
                if f'cls_{key}' in item:
                    return item[f'cls_{key}']  # returned as-is
                return "N/A"

            precision = format_metric('precision')
            recall = format_metric('recall')
            f1 = format_metric('f1')

            print(f"{filename:<35} {precision:<30} {recall:<30} {f1:<30}")

        print("-" * 120)
        print("See evaluation_summary.json for the complete data")
        print(f"Note: every metric above excludes {', '.join(EXCLUDED_CLASSES)}")

        # List every metric that is available
        if summary_data:
            all_metrics = set()
            for item in summary_data:
                all_metrics.update([k for k in item.keys() if k.startswith('cls_') and not k.endswith('_ci')])

            if all_metrics:
                print("\nAvailable classification metrics:")
                for metric in sorted(all_metrics):
                    print(f"   {metric}")

def main():
    """Entry point: evaluate every input file."""
    print("RadChest evaluation script - classification metrics only (95% CI, placeholder classes excluded)")
    print(f"Base directory: {BASE_DIR}")
    print(f"Evaluation script directory: {CODE_DIR}")
    print("Note: this variant computes classification metrics only, it does not score report generation")
    print(f"Excluded placeholder classes: {', '.join(EXCLUDED_CLASSES)}")
    print(f"Bootstrap settings: n={N_BOOTSTRAPS}, CI={CONFIDENCE_LEVEL*100}%")

    # Show the active configuration
    mode_text = "folder mode" if USE_FOLDER_MODE else "explicit file list mode"
    print(f"Mode: {mode_text}")

    if USE_FOLDER_MODE:
        print(f"Search folder: {FOLDER_PATH}")
        print(f"File pattern: {FILE_PATTERN}")

    # Check that the evaluation script directory exists
    if not CODE_DIR.exists():
        print(f"Evaluation script directory does not exist: {CODE_DIR}")
        print(f"Check the path: {CODE_DIR}")
        return

    # Check that the required helper scripts exist
    required_scripts = [INFER_SCRIPT, CLS_SCRIPT]
    missing_scripts = [script for script in required_scripts if not script.exists()]
    if missing_scripts:
        print("Missing evaluation scripts:")
        for script in missing_scripts:
            print(f"   {script}")
        return

    # Collect the input files
    input_files = get_input_files()
    if not input_files:
        print("No input files to process")
        if USE_FOLDER_MODE:
            print("Check that the folder path and file pattern are correct")
        return

    # Check that the input files exist
    missing_files = [f for f in input_files if not f.exists()]
    if missing_files:
        print("These input files do not exist:")
        for f in missing_files:
            print(f"   {f}")
        return

    print(f"\nProcessing {len(input_files)} input file(s):")
    for i, f in enumerate(input_files, 1):
        checkpoint_num = extract_checkpoint_number(f.name)
        if checkpoint_num > 0:
            print(f"   {i}. {f.name} (checkpoint-{checkpoint_num})")
        else:
            print(f"   {i}. {f.name}")

    # Process each file
    successful = 0
    failed = 0
    successful_files = []

    for i, input_file in enumerate(input_files, 1):
        if evaluate_single_file(input_file, i, len(input_files)):
            successful += 1
            successful_files.append(input_file)
        else:
            failed += 1

        # Report progress between files
        if i < len(input_files):
            print(f"\nMoving to the next file... ({i}/{len(input_files)} done)")

    # Generate the summary report
    if successful_files:
        generate_summary_report(successful_files, BASE_DIR)

    # Final summary
    print(f"\n{'='*60}")
    print("Final summary")
    print(f"{'='*60}")
    print(f"Total files: {len(input_files)}")
    print(f"Succeeded: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(input_files)*100:.1f}%" if input_files else "N/A")

    if successful > 0:
        print("\nEvaluation results are available in:")
        for input_file in successful_files:
            output_dir_name = get_output_dir_name(input_file)
            result_dir = BASE_DIR / output_dir_name
            print(f"   {result_dir}/metrics.json")

    print("\nEvaluation complete.")

    # Suggest how to diagnose failures
    if failed > 0:
        print("\nTroubleshooting:")
        print("1. Make sure the preparation script ran successfully and generated the required files")
        print("2. Check that ground_truth.csv contains valid label data")
        print("3. Verify the classifier model path and that the model file exists")
        print("4. Check the CSV format and column names")
        print("5. Confirm that calc_scores_withci.py and confidence_interval.py are in place")

    # Explain the result files
    if successful > 0:
        print("\nResult files:")
        print("   inferred_full.csv - full model predictions (placeholder columns still present)")
        print("   inferred.csv - filtered predictions (placeholder columns dropped)")
        print("   ground_truth_filtered.csv - filtered ground truth (placeholder columns dropped)")
        print("   classification_scores.json - detailed classification metrics (95% CI, computed on the filtered data)")
        print("   metrics.json - final combined evaluation results")
        print("   evaluation_summary.json - summary comparison across all files (95% CI)")
        print(f"\nConfidence interval settings: bootstrap samples={N_BOOTSTRAPS}, CI level={CONFIDENCE_LEVEL*100}%")

    # Exit non-zero when any file failed, so callers can tell success from failure
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())