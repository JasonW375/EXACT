#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master evaluation script for the Mianyang dataset.
Computes classification, CRG and NLG metrics; every supported metric carries a 95% confidence interval.
Supports folder mode: point it at a folder and it will pick up every predictions_checkpoint*.json file.
Placeholder classes are filtered out before scoring.
Classification and NLG metrics are emitted as strings of the form "value [ci_lower, ci_upper]".
NLG scoring runs in parallel (8 workers) with live output.
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

# === Placeholder classes to exclude ===
EXCLUDED_CLASSES = [
    "Mosaic attenuation pattern"
]

# Paths to the evaluation helper scripts
INFER_SCRIPT = CODE_DIR / "infer.py"
CLS_SCRIPT   = CODE_DIR / "calc_scores_withci.py"
CRG_SCRIPT   = CODE_DIR / "crg_score_mianyang.py"
NLG_SCRIPT   = CODE_DIR / "nlg_metrics_withci.py"

# Bootstrap settings
N_BOOTSTRAPS = 1000        # number of bootstrap resamples
CONFIDENCE_LEVEL = 0.95    # confidence level (95% CI)
RANDOM_STATE = 42          # random seed
N_JOBS = 8                 # parallel workers for NLG evaluation

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

def run(script: Path, *args, capture_output=False, realtime=True):
    """Run a helper script as a subprocess.

    Args:
        script: path to the script
        *args: arguments passed to the script
        capture_output: deprecated, kept for backwards compatibility
        realtime: stream output live (True) or buffer and print afterwards (False)
    """
    cmd = [sys.executable, str(script), *map(str, args)]
    print(">>", " ".join(cmd))

    try:
        if realtime:
            # Live output mode (preferred)
            result = subprocess.run(cmd, check=True)
        else:
            # Buffered output mode
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)

    except subprocess.CalledProcessError as e:
        print(f"\nScript failed: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print("STDOUT:", e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print("STDERR:", e.stderr, file=sys.stderr)
        raise

def load(p: Path):
    """Load a JSON file."""
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def check_required_files(prep_dir: Path):
    """Check that the files produced by the preparation script are present."""
    required_files = [
        "result_transformat.json",
        "ground_truth.json",
        "ground_truth.csv"
    ]

    missing = []
    for filename in required_files:
        file_path = prep_dir / filename
        if not file_path.exists():
            missing.append(filename)

    return missing

def remove_excluded_columns(csv_path: Path, output_path: Path, excluded_classes: list):
    """Drop the given placeholder class columns from a CSV file."""
    df = pd.read_csv(csv_path)
    original_cols = len(df.columns)
    cols_to_drop = [col for col in excluded_classes if col in df.columns]

    if cols_to_drop:
        print(f"   Dropping placeholder columns: {', '.join(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)

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
            label_cols = [col for col in df.columns if col != 'AccessionNo']

            if excluded_classes:
                label_cols = [col for col in label_cols if col not in excluded_classes]
                print(f"   Label columns: {len(label_cols)} (after excluding placeholder classes)")
            else:
                print(f"   Label columns: {len(label_cols)}")

            positive_counts = {}
            for col in label_cols:
                if col in df.columns:
                    positive_count = (df[col] == 1).sum()
                    positive_counts[col] = positive_count

            total_positives = sum(positive_counts.values())
            print(f"   Total positive samples: {total_positives}")

            if total_positives == 0:
                print("   Warning: all labels are 0, which can break classification evaluation")

            positive_classes = [(k, v) for k, v in positive_counts.items() if v > 0]
            if positive_classes:
                print("   Classes with positive samples (top 5):")
                for cls, count in sorted(positive_classes, key=lambda x: x[1], reverse=True)[:5]:
                    print(f"     {cls}: {count}")
        else:
            print("   Warning: the CSV file is empty")

    except Exception as e:
        print(f"   Could not analyse the CSV file: {e}")

def parse_metric_string(metric_str):
    """Parse a metric string formatted as: value [ci_lower, ci_upper]."""
    if isinstance(metric_str, (int, float)):
        return {
            'value': float(metric_str),
            'ci_lower': float(metric_str),
            'ci_upper': float(metric_str)
        }

    try:
        parts = metric_str.split('[')
        value = float(parts[0].strip())
        ci_part = parts[1].rstrip(']').split(',')
        ci_lower = float(ci_part[0].strip())
        ci_upper = float(ci_part[1].strip())

        return {
            'value': value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    except (IndexError, ValueError) as e:
        return {
            'value': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0
        }

def evaluate_single_file(input_file: Path, file_index: int, total_files: int):
    """Run the full evaluation for a single input file."""
    checkpoint_num = extract_checkpoint_number(input_file.name)

    print(f"\n{'='*60}")
    print(f"Processing file [{file_index}/{total_files}]: {input_file.name}")
    if checkpoint_num > 0:
        print(f"Checkpoint: {checkpoint_num}")
    print(f"{'='*60}")
    print(f"Note: excluding placeholder classes {', '.join(EXCLUDED_CLASSES)}")

    output_dir_name = get_output_dir_name(input_file)
    prep_dir = BASE_DIR / output_dir_name

    print(f"Prepared files directory: {prep_dir}")

    if not prep_dir.exists():
        print(f"Prepared file directory does not exist: {prep_dir}")
        print("Run the preparation script first to generate the required files.")
        return False

    missing_files = check_required_files(prep_dir)
    if missing_files:
        print("Missing required files:")
        for f in missing_files:
            print(f"   {f}")
        print("Make sure the preparation script completed successfully.")
        return False

    # File paths
    pred_json_transformed = prep_dir / "result_transformat.json"
    gt_json = prep_dir / "ground_truth.json"
    gt_csv = prep_dir / "ground_truth.csv"

    csv_pred_full = prep_dir / "inferred_full.csv"
    csv_pred = prep_dir / "inferred.csv"
    gt_csv_filtered = prep_dir / "ground_truth_filtered.csv"

    cls_json = prep_dir / "classification_scores.json"
    crg_json = prep_dir / "crg_scores.json"
    nlg_json = prep_dir / "nlg_scores.json"
    final_json = prep_dir / "metrics.json"

    print(f"Using transformed predictions: {pred_json_transformed.name}")
    print(f"Ground truth JSON: {gt_json.name}")
    print(f"Ground truth CSV: {gt_csv.name}")

    try:
        # 1. Inference -> CSV
        print("\n1. Running inference...")
        run(INFER_SCRIPT,
            "--input_json", pred_json_transformed,
            "--model_path", CODE_DIR / "roberta_local" / "RadBertClassifier.pth",
            "--out_csv", csv_pred_full,
            realtime=True)
        print(f"Inference finished, results written to: {csv_pred_full.name}")

        # 2. Drop the placeholder columns
        print("\n2. Removing placeholder columns...")
        print("\nRaw prediction summary:")
        check_csv_content(csv_pred_full, excluded_classes=None)

        print("\nRaw ground truth summary:")
        check_csv_content(gt_csv, excluded_classes=None)

        print(f"\nFiltering predictions: {csv_pred_full.name} -> {csv_pred.name}")
        removed_pred = remove_excluded_columns(csv_pred_full, csv_pred, EXCLUDED_CLASSES)

        print(f"Filtering ground truth: {gt_csv.name} -> {gt_csv_filtered.name}")
        removed_gt = remove_excluded_columns(gt_csv, gt_csv_filtered, EXCLUDED_CLASSES)

        if removed_pred > 0 or removed_gt > 0:
            print(f"Dropped {removed_pred} placeholder column(s)")
        else:
            print("No placeholder columns needed to be dropped")

        print("\nFiltered prediction summary:")
        check_csv_content(csv_pred, excluded_classes=None)

        print("\nFiltered ground truth summary:")
        check_csv_content(gt_csv_filtered, excluded_classes=None)

        # 3. Classification scores
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
            "--random_state", str(RANDOM_STATE),
            realtime=True)
        print(f"Classification finished (with 95% CI), results written to: {cls_json.name}")

        # 4. CRG metrics
        print("\n4. Computing CRG metrics...")
        print("   Using the filtered CSVs (placeholder classes removed)")
        print("   Using the dynamic-column variant of the CRG script")
        run(CRG_SCRIPT,
            "--pred_csv", csv_pred,
            "--gt_csv", gt_csv_filtered,
            "--out_json", crg_json,
            realtime=True)
        print(f"CRG evaluation finished, results written to: {crg_json.name}")

        # 5. NLG metrics
        print("\n5. Computing NLG metrics with 95% CI (Parallel Processing)...")
        print("   Using the transformed predictions for NLG evaluation")
        print(f"   Bootstrap samples: {N_BOOTSTRAPS}")
        print(f"   Confidence level: {CONFIDENCE_LEVEL * 100}%")
        print(f"   Parallel workers: {N_JOBS}")
        print()  # blank line

        # Stream output live so the progress bar stays visible
        run(NLG_SCRIPT,
            "--pred_json", pred_json_transformed,
            "--gt_json", gt_json,
            "--out_json", nlg_json,
            "--n_bootstraps", str(N_BOOTSTRAPS),
            "--confidence_level", str(CONFIDENCE_LEVEL),
            "--random_state", str(RANDOM_STATE),
            "--n_jobs", str(N_JOBS),
            realtime=True)

        print()  # blank line
        print(f"NLG evaluation finished (95% CI, parallelised), results written to: {nlg_json.name}")

        # 6. Combine the results
        print("\n6. Combining all metrics...")
        cls_metrics = load(cls_json)
        crg_metrics = load(crg_json)
        nlg_metrics = load(nlg_json)

        nlg_metrics_clean = {k: v for k, v in nlg_metrics.items() if not k.startswith('_')}

        combined = {
            "generation": nlg_metrics_clean,
            "classification": cls_metrics,
            "crg": crg_metrics,
            "metadata": {
                "checkpoint": checkpoint_num if checkpoint_num > 0 else "N/A",
                "input_file": input_file.name,
                "evaluation_type": "full_evaluation_with_ci_parallel",
                "excluded_classes": EXCLUDED_CLASSES,
                "bootstrap_config": {
                    "n_bootstraps": N_BOOTSTRAPS,
                    "confidence_level": CONFIDENCE_LEVEL,
                    "random_state": RANDOM_STATE,
                    "nlg_n_jobs": N_JOBS
                },
                "note": "Classification and NLG metrics include 95% confidence intervals; NLG uses 8 parallel workers; CRG is computed after excluding placeholder classes"
            }
        }

        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"\nAll metrics written to: {final_json}")

        # Print a summary of the results
        print(f"\nEvaluation results (Checkpoint-{checkpoint_num if checkpoint_num > 0 else 'N/A'}, placeholder classes excluded):")
        print("=" * 80)

        if "classification" in combined and combined["classification"]:
            cls_data = combined["classification"]
            print("Classification metrics (95% CI):")

            if 'macro' in cls_data:
                print("\n   Macro average:")
                macro = cls_data['macro']
                for key in ['precision', 'recall', 'f1', 'accuracy']:
                    if key in macro:
                        value_str = macro[key]
                        print(f"      {key:12s}: {value_str}")

            if 'per_pathology' in cls_data and len(cls_data['per_pathology']) > 0:
                print("\n   Per-class metrics (first 3):")
                for item in cls_data['per_pathology'][:3]:
                    print(f"\n      {item['name']}:")
                    for key in ['precision', 'recall', 'f1', 'accuracy']:
                        if key in item:
                            value_str = item[key]
                            print(f"         {key:12s}: {value_str}")

                if len(cls_data['per_pathology']) > 3:
                    print(f"\n      ... and {len(cls_data['per_pathology']) - 3} more classes")

        if "generation" in combined and combined["generation"]:
            gen_metrics = combined["generation"]
            print(f"\nGeneration metrics (95% CI, {N_JOBS} parallel workers):")
            for key, value in gen_metrics.items():
                if isinstance(value, str):
                    print(f"   {key:12s}: {value}")
                elif isinstance(value, (int, float)):
                    print(f"   {key:12s}: {value:.4f}")

        if "crg" in combined and combined["crg"]:
            crg_data = combined["crg"]
            print("\nCRG metrics:")
            for key, value in crg_data.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")

        return True

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_summary_report(successful_files: list[Path], base_dir: Path):
    """Write a summary report across all successfully evaluated files."""
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

                summary_item = {
                    "checkpoint": checkpoint_num if checkpoint_num > 0 else "N/A",
                    "filename": input_file.name
                }

                if "classification" in metrics and "macro" in metrics["classification"]:
                    macro = metrics["classification"]["macro"]
                    for key in ['precision', 'recall', 'f1', 'accuracy']:
                        if key in macro:
                            value_str = macro[key]
                            summary_item[f"cls_{key}"] = value_str

                if "generation" in metrics:
                    gen_metrics = metrics["generation"]
                    for key, value in gen_metrics.items():
                        if isinstance(value, str):
                            summary_item[f"gen_{key}"] = value
                        elif isinstance(value, (int, float)):
                            summary_item[f"gen_{key}"] = round(value, 4)

                if "crg" in metrics:
                    crg_metrics = metrics["crg"]
                    for key, value in crg_metrics.items():
                        if isinstance(value, (int, float)):
                            summary_item[f"crg_{key}"] = round(value, 4)

                summary_data.append(summary_item)

            except Exception as e:
                print(f"Could not load {metrics_file}: {e}")

    if summary_data:
        summary_file = base_dir / "evaluation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"Summary report written to: {summary_file}")

        print(f"\nPerformance comparison (95% CI, {N_JOBS} parallel workers):")
        print("-" * 150)
        print(f"{'Checkpoint':<12} {'File':<30} {'Cls F1 (95% CI)':<35} {'BLEU-1 (95% CI)':<35} {'ROUGE-L (95% CI)':<35}")
        print("-" * 150)

        for item in sorted(summary_data, key=lambda x: x['checkpoint'] if isinstance(x['checkpoint'], int) else 0):
            checkpoint = item['checkpoint']
            filename = item['filename'][:27] + "..." if len(item['filename']) > 30 else item['filename']

            def format_metric(key):
                if key in item:
                    value = item[key]
                    if isinstance(value, str):
                        return value
                    else:
                        return f"{value:.4f}"
                return "N/A"

            cls_f1 = format_metric('cls_f1')
            bleu1 = format_metric('gen_BLEU_1')
            rougeL = format_metric('gen_ROUGE_L')

            print(f"{str(checkpoint):<12} {filename:<30} {cls_f1:<35} {bleu1:<35} {rougeL:<35}")

        print("-" * 150)
        print("See evaluation_summary.json for the complete data")
        print(f"Note: classification and CRG metrics exclude {', '.join(EXCLUDED_CLASSES)}")

        if summary_data:
            all_metrics = set()
            for item in summary_data:
                all_metrics.update([k for k in item.keys() if k not in ['checkpoint', 'filename']])

            if all_metrics:
                print("\nAvailable metrics:")
                cls_metrics = sorted([m for m in all_metrics if m.startswith('cls_')])
                gen_metrics = sorted([m for m in all_metrics if m.startswith('gen_')])
                crg_metrics = sorted([m for m in all_metrics if m.startswith('crg_')])

                if cls_metrics:
                    print(f"   Classification (95% CI): {', '.join(cls_metrics)}")
                if gen_metrics:
                    print(f"   Generation (95% CI, {N_JOBS} parallel workers): {', '.join(gen_metrics)}")
                if crg_metrics:
                    print(f"   CRG: {', '.join(crg_metrics)}")

def main():
    """Entry point: evaluate every input file."""
    print("Mianyang dataset evaluation - full run (classification + NLG with 95% CI, folder mode, placeholder filtering, parallel NLG, live output)")
    print(f"Base directory: {BASE_DIR}")
    print(f"Evaluation script directory: {CODE_DIR}")
    print(f"Excluded placeholder classes: {', '.join(EXCLUDED_CLASSES)}")
    print(f"Bootstrap settings: n={N_BOOTSTRAPS}, CI={CONFIDENCE_LEVEL*100}%")
    print(f"NLG parallel workers: {N_JOBS}")

    mode_text = "folder mode" if USE_FOLDER_MODE else "explicit file list mode"
    print(f"Mode: {mode_text}")

    if USE_FOLDER_MODE:
        print(f"Search folder: {FOLDER_PATH}")
        print(f"File pattern: {FILE_PATTERN}")

    if not CODE_DIR.exists():
        print(f"Evaluation script directory does not exist: {CODE_DIR}")
        return

    required_scripts = [INFER_SCRIPT, CLS_SCRIPT, CRG_SCRIPT, NLG_SCRIPT]
    missing_scripts = [script for script in required_scripts if not script.exists()]
    if missing_scripts:
        print("Missing evaluation scripts:")
        for script in missing_scripts:
            print(f"   {script}")
        return

    input_files = get_input_files()
    if not input_files:
        print("No input files to process")
        if USE_FOLDER_MODE:
            print("Check that the folder path and file pattern are correct")
        return

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

    successful = 0
    failed = 0
    successful_files = []

    for i, input_file in enumerate(input_files, 1):
        if evaluate_single_file(input_file, i, len(input_files)):
            successful += 1
            successful_files.append(input_file)
        else:
            failed += 1

        if i < len(input_files):
            print(f"\nMoving to the next file... ({i}/{len(input_files)} done)")

    if successful_files:
        generate_summary_report(successful_files, base_dir=BASE_DIR)

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

    if failed > 0:
        print("\nTroubleshooting:")
        print("1. Make sure the preparation script ran successfully and generated the required files")
        print("2. Check that ground_truth.csv contains valid label data")
        print("3. Verify the classifier model path and that the model file exists")
        print("4. Check the CSV format and column names")
        print("5. Confirm that calc_scores_withci.py and nlg_metrics_withci.py are in place")
        print("6. Make sure the required NLG libraries are installed: nltk, rouge_score, joblib")

    if successful > 0:
        print("\nResult files:")
        print("   inferred_full.csv - full model predictions (placeholder columns still present)")
        print("   inferred.csv - filtered predictions (placeholder columns dropped)")
        print("   ground_truth_filtered.csv - filtered ground truth (placeholder columns dropped)")
        print("   classification_scores.json - detailed classification metrics (95% CI, computed on the filtered data)")
        print("   crg_scores.json - CRG metrics (computed on the filtered data)")
        print(f"   nlg_scores.json - NLG metrics (95% CI, {N_JOBS} parallel workers)")
        print("   metrics.json - final combined evaluation results")
        print("   evaluation_summary.json - summary comparison across all files (95% CI)")
        print(f"\nConfidence interval settings: bootstrap samples={N_BOOTSTRAPS}, CI level={CONFIDENCE_LEVEL*100}%")
        print(f"NLG speedup: {N_JOBS} parallel workers (progress shown live)")
        print(f"Metrics computed: classification (CI) + CRG + NLG (CI)")

if __name__ == "__main__":
    main()