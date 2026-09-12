#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare the inputs that the evaluation scripts expect.

`evaluation.py`, `evaluation_radchest.py` and `evaluation_mianyang.py` do not read
an inference output directly. For every prediction file they look for a sibling
folder under `--base_dir`, named after the prediction file's stem, containing:

    result_transformat.json   reports wrapped in the structure infer.py reads
    ground_truth.json         reference reports, used by the NLG metrics
    ground_truth.csv          reference labels, used by the classification metrics

This script writes those three files. Run it once per cohort before running the
matching evaluation script.

Example (CT-RATE):

    python prepare_eval_files.py \
        --prediction_jsons out_ctrate.json \
        --base_dir ctrate_workdir \
        --reference_json /path/to/report_generation.json \
        --label_csv /path/to/valid_predicted_labels.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# The 18 CT-RATE abnormality columns. RAD-ChestCT and MianYang are scored on
# subsets of these; the evaluation scripts drop the columns that do not apply.
DEFAULT_LABEL_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction_jsons", nargs="+", required=True, type=Path,
                    help="One or more EXACT-CHAT inference output JSONs.")
    ap.add_argument("--base_dir", required=True, type=Path,
                    help="Output directory; must match the --base_dir passed to "
                         "the evaluation script.")
    ap.add_argument("--reference_json", required=True, type=Path,
                    help="Conversation JSON holding the reference reports. The "
                         "inference input for the same cohort works, since its "
                         "gpt turns are the ground truth.")
    ap.add_argument("--label_csv", type=Path, default=None,
                    help="CSV of reference labels, keyed by a VolumeName column. "
                         "Omit for cohorts without labels; an all-zero "
                         "ground_truth.csv is written instead.")
    ap.add_argument("--label_columns", nargs="+", default=DEFAULT_LABEL_COLS,
                    help="Label columns to copy into ground_truth.csv.")
    return ap.parse_args()


def _strip_eot(text: str) -> str:
    """Drop the Llama-3 end-of-turn marker left in place by the inference script."""
    text = text.strip()
    if text.endswith("<|eot_id|>"):
        text = text[: -len("<|eot_id|>")].strip()
    return text


def _match(name: str, index: dict):
    """Look a study up under the naming variants used across the cohorts."""
    stem = Path(name).stem
    for key in (name, f"{name}.nii.gz", f"{stem}.nii.gz", f"{stem}.npz", stem):
        if key in index:
            return index[key]
    # Fall back to comparing stems, which covers .npz vs .nii.gz mismatches.
    by_stem = {Path(k).stem: v for k, v in index.items()}
    return by_stem.get(stem)


def load_predictions(path: Path) -> list:
    """Read an inference output into [{input_image_name, report}, ...]."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "generated_reports" in data:
        raw = data["generated_reports"]
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError(f"Unrecognised structure in {path}")

    items = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("input_image_name") or entry.get("image")
                or entry.get("image_name") or entry.get("image_id"))
        if not name:
            continue

        # Native inference format: take the answer of the first turn.
        if "conversations_out" in entry:
            report = None
            for turn in entry["conversations_out"]:
                if isinstance(turn, dict) and turn.get("answer"):
                    report = turn["answer"]
                    break
        else:
            report = (entry.get("report") or entry.get("generated_report")
                      or entry.get("text") or entry.get("answer"))

        if report:
            items.append({"input_image_name": str(name).strip(),
                          "report": _strip_eot(str(report))})

    if not items:
        raise ValueError(f"No usable reports found in {path}")
    return items


def load_reference_index(path: Path) -> dict:
    """Map study name -> reference report, read from the gpt turns."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    index = {}
    for entry in data if isinstance(data, list) else []:
        image = entry.get("image")
        turns = entry.get("conversations", [])
        if not image or not isinstance(turns, list):
            continue

        reference = None
        # Prefer the answer that follows an explicit report_generation question.
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict) or turn.get("from") != "human":
                continue
            kind = turn.get("type") or turn.get("conversation_type") or ""
            if str(kind).strip().lower() == "report_generation":
                nxt = turns[i + 1] if i + 1 < len(turns) else None
                if isinstance(nxt, dict) and nxt.get("from") == "gpt":
                    reference = nxt.get("value")
                    break
        if not reference:
            for turn in turns:
                if isinstance(turn, dict) and turn.get("from") == "gpt":
                    reference = turn.get("value")
                    break

        if reference:
            index[image] = reference
    return index


def write_transformed(items: list, out_path: Path):
    """Wrap the reports in the nesting infer.py expects."""
    payload = [{"outputs": [{"value": {"generated_reports": items}}]}]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_ground_truth_json(items: list, index: dict, out_path: Path):
    gt, missing = [], []
    for it in items:
        reference = _match(it["input_image_name"], index)
        if reference:
            gt.append({"input_image_name": Path(it["input_image_name"]).stem,
                       "report": reference})
        else:
            missing.append(it["input_image_name"])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"generated_reports": gt}, f, ensure_ascii=False, indent=2)

    print(f"  ground_truth.json: {len(gt)} reference reports")
    if missing:
        print(f"  WARNING: no reference report for {len(missing)} studies, "
              f"e.g. {missing[:3]}")


def write_ground_truth_csv(items: list, label_csv, label_cols: list, out_path: Path):
    columns = ["AccessionNo"] + label_cols

    table = {}
    if label_csv and Path(label_csv).exists():
        df = pd.read_csv(label_csv)
        if "VolumeName" in df.columns:
            table = {str(r["VolumeName"]): r for _, r in df.iterrows()}
        else:
            print(f"  WARNING: {label_csv} has no VolumeName column")
    elif label_csv:
        print(f"  WARNING: label CSV not found: {label_csv}")

    rows, missing = [], []
    for it in items:
        stem = Path(it["input_image_name"]).stem
        row = _match(it["input_image_name"], table) if table else None
        if row is None:
            missing.append(it["input_image_name"])
            rows.append({"AccessionNo": stem, **{c: 0 for c in label_cols}})
        else:
            rows.append({"AccessionNo": stem,
                         **{c: (row[c] if c in row else 0) for c in label_cols}})

    pd.DataFrame(rows, columns=columns).to_csv(out_path, index=False)
    print(f"  ground_truth.csv: {len(rows)} rows x {len(label_cols)} labels")
    if missing:
        print(f"  WARNING: no labels for {len(missing)} studies, defaulted to 0, "
              f"e.g. {missing[:3]}")


def main():
    args = parse_args()
    index = load_reference_index(args.reference_json)
    print(f"Loaded {len(index)} reference reports from {args.reference_json}")

    for pred_path in args.prediction_jsons:
        out_dir = args.base_dir / pred_path.stem.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{pred_path.name} -> {out_dir}")

        items = load_predictions(pred_path)
        print(f"  {len(items)} generated reports")

        write_transformed(items, out_dir / "result_transformat.json")
        write_ground_truth_json(items, index, out_dir / "ground_truth.json")
        write_ground_truth_csv(items, args.label_csv, args.label_columns,
                               out_dir / "ground_truth.csv")


if __name__ == "__main__":
    main()
