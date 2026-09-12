#!/usr/bin/env python3
"""Attach disease labels from a CSV to a preprocessed HDF5 store.

This is the last preprocessing step. data_preprocessed.py and
new_data_preprocessed.py write only `ct` (and `mask` for the internal cohort); the
dataloaders additionally expect a label vector per study, which this script adds.

The CSV must have the study identifier in its first column and one binary column
per finding, in the channel order the model uses:

    VolumeName,Medical material,Arterial wall calcification,Cardiomegaly,...
    train_1_a_1.nii.gz,0,1,0,...

Any .nii.gz suffix in the identifier is stripped so it matches the HDF5 group name.
The dataset is named after the number of label columns - 18 columns become
`label_18`, and the 16 columns annotated by RAD-ChestCT and MianYang become
`label_16`, which the dataloaders expand to 18 channels at load time.

Studies present in only one of the two files are reported and left untouched, so a
partially labelled store is easy to spot rather than silently mis-scored.

Example
-------
    python save_label_18.py \
        --csv predicted_labels.csv \
        --h5  train_total_processed_data.h5
"""

import argparse
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def save_labels_to_h5(csv_path, h5_path, label_key=None, overwrite=True):
    """Write one label vector per study into the HDF5 store."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 store not found: {h5_path}")

    df = pd.read_csv(csv_path)
    id_column = df.columns[0]
    disease_columns = df.columns[1:]

    if label_key is None:
        label_key = f"label_{len(disease_columns)}"

    print(f"CSV       : {csv_path}")
    print(f"studies   : {len(df)}")
    print(f"findings  : {len(disease_columns)} -> dataset '{label_key}'")
    for i, disease in enumerate(disease_columns, start=1):
        print(f"  {i:2d}. {disease}")

    volume_to_labels = {}
    for _, row in df.iterrows():
        # HDF5 group names carry no file extension.
        sample_key = str(row[id_column]).replace(".nii.gz", "")
        volume_to_labels[sample_key] = row[disease_columns].values.astype(np.int32)

    with h5py.File(h5_path, "r+") as f:
        h5_keys = list(f.keys())
        print(f"\nHDF5      : {h5_path} ({len(h5_keys)} studies)")

        matched = 0
        missing_label = []

        for sample_key in tqdm(h5_keys, desc="labelling"):
            if sample_key not in volume_to_labels:
                missing_label.append(sample_key)
                continue

            if label_key in f[sample_key]:
                if not overwrite:
                    continue
                del f[sample_key][label_key]

            f[sample_key].create_dataset(
                label_key, data=volume_to_labels[sample_key], dtype=np.int32
            )
            matched += 1

        missing_volume = [k for k in volume_to_labels if k not in set(h5_keys)]

    print(f"\nlabelled              : {matched}")
    print(f"in HDF5 but not in CSV: {len(missing_label)}")
    print(f"in CSV but not in HDF5: {len(missing_volume)}")

    for title, keys in (("in HDF5 but not in CSV", missing_label),
                        ("in CSV but not in HDF5", missing_volume)):
        if keys:
            print(f"\nfirst 10 {title}:")
            for key in keys[:10]:
                print(f"  {key}")
            if len(keys) > 10:
                print(f"  ... and {len(keys) - 10} more")

    if missing_label:
        print(f"\nWarning: {len(missing_label)} studies have no '{label_key}' and will "
              f"fail to load. Either extend the CSV or drop them from the store.")

    return label_key


def verify_labels(h5_path, label_key, num_samples=5):
    """Print the label vector of the first few studies as a sanity check."""
    print(f"\nverifying '{label_key}' on the first {num_samples} studies:")
    with h5py.File(h5_path, "r") as f:
        for key in list(f.keys())[:num_samples]:
            if label_key in f[key]:
                labels = f[key][label_key][:]
                print(f"  {key}: {labels} (length {len(labels)})")
            else:
                print(f"  {key}: '{label_key}' missing")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach disease labels from a CSV to a preprocessed HDF5 store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True,
                        help="Label CSV; first column is the study identifier")
    parser.add_argument("--h5", required=True,
                        help="HDF5 store to modify in place")
    parser.add_argument("--label-key", default=None,
                        help="Dataset name to write; defaults to label_<n columns>")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Keep labels that are already present")
    return parser.parse_args()


def main():
    args = parse_args()
    label_key = save_labels_to_h5(args.csv, args.h5, args.label_key,
                                  overwrite=not args.no_overwrite)
    verify_labels(args.h5, label_key)


if __name__ == "__main__":
    main()
