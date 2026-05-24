#!/usr/bin/env python3
"""
csv2generated_reports_json.py
=============================

Convert a CSV mapping (with columns `nii_path` and `concat_text`) into the
JSON schema expected by the evaluation pipeline.

Output schema:
    {
        "name": "Generated reports",
        "type": "Report generation",
        "generated_reports": [
            {"input_image_name": "<filename>.mha", "report": "<concat_text>"},
            ...
        ],
        "version": {"major": 1, "minor": 0}
    }
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv_path", type=Path, required=True,
                    help="Input CSV with columns nii_path and concat_text.")
    ap.add_argument("--out_json", type=Path, required=True,
                    help="Output JSON path.")
    ap.add_argument("--n_rows", type=int, default=None,
                    help="Optional row limit; default is the whole file.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path, nrows=args.n_rows)

    records = []
    for _, row in df.iterrows():
        nii_path = Path(row["nii_path"])
        mha_name = nii_path.name.replace(".nii.gz", ".mha")
        records.append({
            "input_image_name": mha_name,
            "report": row["concat_text"],
        })

    wrapper = {
        "name": "Generated reports",
        "type": "Report generation",
        "generated_reports": records,
        "version": {"major": 1, "minor": 0},
    }

    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(wrapper, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} entries -> {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
