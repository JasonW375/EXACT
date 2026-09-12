#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master evaluation script for CT-RATE.

Runs the four scoring stages over one or more EXACT-CHAT inference outputs and
merges them into a single metrics.json per prediction file:

    infer.py         labels the generated reports with the RadBERT classifier
    calc_scores.py   multi-label classification metrics against ground_truth.csv
    crg_score.py     CRG score
    nlg_metrics.py   BLEU / ROUGE-L / METEOR / CIDEr against ground_truth.json

Run prepare_eval_files.py first: it writes the result_transformat.json,
ground_truth.json and ground_truth.csv that this script consumes.

Example:

    python evaluation.py \
        --prediction_jsons out_ctrate.json \
        --base_dir ctrate_workdir
"""

import json
import re
import subprocess
import sys
from pathlib import Path


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
INPUT_FILES = [Path(p) for p in _ARGS.prediction_jsons]
BASE_DIR = _ARGS.base_dir
CODE_DIR = _ARGS.code_dir

INFER_SCRIPT = CODE_DIR / "infer.py"
CLS_SCRIPT = CODE_DIR / "calc_scores.py"
CRG_SCRIPT = CODE_DIR / "crg_score.py"
NLG_SCRIPT = CODE_DIR / "nlg_metrics.py"


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
        "ground_truth.json",        # reference reports, for the NLG metrics
        "ground_truth.csv",         # reference labels, for the classification metrics
    ]
    return [name for name in required_files if not (prep_dir / name).exists()]


def check_csv_content(csv_path: Path):
    """Summarise a label CSV, so an all-zero or empty table is caught early."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"Label CSV summary ({csv_path.name}):")
        print(f"   rows: {len(df)}")

        if len(df) == 0:
            print("   WARNING: the CSV is empty")
            return

        label_cols = [col for col in df.columns if col != 'AccessionNo']
        print(f"   label columns: {len(label_cols)}")

        positive_counts = {col: int((df[col] == 1).sum()) for col in label_cols}
        total_positives = sum(positive_counts.values())
        print(f"   positive cells: {total_positives}")

        if total_positives == 0:
            print("   WARNING: every label is 0, classification metrics will be degenerate")

        positive_classes = [(k, v) for k, v in positive_counts.items() if v > 0]
        if positive_classes:
            print("   classes with positives:")
            for cls, count in sorted(positive_classes, key=lambda x: x[1], reverse=True)[:18]:
                print(f"     {cls}: {count}")

    except Exception as e:
        print(f"   ERROR: could not read the CSV: {e}")


def evaluate_single_file(input_file: Path, file_index: int, total_files: int):
    """Score one prediction file end to end. Returns True on success."""
    checkpoint_num = extract_checkpoint_number(input_file.name)

    print(f"\n{'=' * 60}")
    print(f"File [{file_index}/{total_files}]: {input_file.name}")
    print(f"Checkpoint: {checkpoint_num}")
    print(f"{'=' * 60}")

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
    gt_json = prep_dir / "ground_truth.json"
    gt_csv = prep_dir / "ground_truth.csv"

    # Outputs written below.
    csv_pred = prep_dir / "inferred.csv"
    cls_json = prep_dir / "classification_scores.json"
    crg_json = prep_dir / "crg_scores.json"
    nlg_json = prep_dir / "nlg_scores.json"
    final_json = prep_dir / "metrics.json"

    print(f"Predictions:      {pred_json_transformed.name}")
    print(f"Ground truth JSON: {gt_json.name}")
    print(f"Ground truth CSV:  {gt_csv.name}")

    try:
        # 1. Label the generated reports with the RadBERT classifier.
        print("\n1. Running RadBERT inference...")
        run(INFER_SCRIPT,
            "--input_json", pred_json_transformed,
            "--model_path", CODE_DIR / "roberta_local" / "RadBertClassifier.pth",
            "--out_csv", csv_pred)
        print(f"   done -> {csv_pred.name}")

        if csv_pred.exists():
            check_csv_content(csv_pred)

        # 2. Multi-label classification scores.
        print("\n2. Computing classification scores...")
        print("   Precision warnings here usually mean a class has very few predictions.")
        run(CLS_SCRIPT,
            "--pred_csv", csv_pred,
            "--gt_csv", gt_csv,
            "--out_json", cls_json)
        print(f"   done -> {cls_json.name}")

        # 3. CRG score.
        print("\n3. Computing CRG score...")
        run(CRG_SCRIPT,
            "--pred_csv", csv_pred,
            "--gt_csv", gt_csv,
            "--out_json", crg_json)
        print(f"   done -> {crg_json.name}")

        # 4. NLG metrics, scored on the same transformed file used above.
        print("\n4. Computing NLG metrics...")
        run(NLG_SCRIPT,
            "--pred_json", pred_json_transformed,
            "--gt_json", gt_json,
            "--out_json", nlg_json)
        print(f"   done -> {nlg_json.name}")

        # 5. Merge everything into one metrics.json.
        print("\n5. Combining all metrics...")
        combined = {
            "generation": load(nlg_json),
            "classification": load(cls_json),
            "crg": load(crg_json),
            "metadata": {
                "checkpoint": checkpoint_num,
                "input_file": input_file.name,
            },
        }

        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"\nAll metrics written to: {final_json}")

        print(f"\nSummary (checkpoint-{checkpoint_num}):")
        print("=" * 50)
        for section, title in (("classification", "Classification"),
                               ("generation", "Generation"),
                               ("crg", "CRG")):
            metrics = combined.get(section)
            if not metrics:
                continue
            print(f"{title}:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")

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

        summary_item = {
            "checkpoint": extract_checkpoint_number(input_file.name),
            "filename": input_file.name,
        }
        for section, prefix in (("classification", "cls"),
                                ("generation", "gen"),
                                ("crg", "crg")):
            for key, value in (metrics.get(section) or {}).items():
                if isinstance(value, (int, float)):
                    summary_item[f"{prefix}_{key}"] = round(value, 4)

        summary_data.append(summary_item)

    if not summary_data:
        return

    summary_file = base_dir / "evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"Summary written to: {summary_file}")

    # Compact table: checkpoint plus the first three numeric metrics.
    print("\nComparison (first 3 metrics per file):")
    print("-" * 80)
    print(f"{'Checkpoint':<12} {'File':<25} {'Metric 1':<12} {'Metric 2':<12} {'Metric 3':<12}")
    print("-" * 80)

    for item in sorted(summary_data, key=lambda x: x['checkpoint']):
        filename = item['filename']
        if len(filename) > 25:
            filename = filename[:22] + "..."

        numeric_keys = [k for k, v in item.items()
                        if k not in ('checkpoint', 'filename') and isinstance(v, (int, float))][:3]
        values = [f"{item[k]:.3f}" for k in numeric_keys]
        while len(values) < 3:
            values.append("N/A")

        print(f"{item['checkpoint']:<12} {filename:<25} "
              f"{values[0]:<12} {values[1]:<12} {values[2]:<12}")

    print("-" * 80)
    print("Full numbers: evaluation_summary.json")


def main():
    print("EXACT-CHAT evaluation (CT-RATE)")
    print(f"Base directory:       {BASE_DIR}")
    print(f"Evaluation scripts:   {CODE_DIR}")

    if not CODE_DIR.exists():
        print(f"ERROR: evaluation script directory does not exist: {CODE_DIR}")
        return

    missing_scripts = [s for s in (INFER_SCRIPT, CLS_SCRIPT, CRG_SCRIPT, NLG_SCRIPT)
                       if not s.exists()]
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
        print(f"   {i}. {f.name} (checkpoint-{extract_checkpoint_number(f.name)})")

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
    print(f"Total:   {len(input_files)}")
    print(f"Succeeded: {len(successful_files)}")
    print(f"Failed:    {failed}")

    if successful_files:
        print("\nResults:")
        for input_file in successful_files:
            print(f"   {BASE_DIR / get_output_dir_name(input_file)}/metrics.json")

    if failed:
        print("\nIf a file failed, check in this order:")
        print("1. prepare_eval_files.py ran for that prediction file")
        print("2. ground_truth.csv is not all zeros")
        print("3. roberta_local/RadBertClassifier.pth exists (see download_RoBERTa_tokenizer.py)")


if __name__ == "__main__":
    main()
