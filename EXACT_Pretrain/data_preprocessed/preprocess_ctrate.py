"""Resample raw CT-RATE volumes into the RadGenome `*_preprocessed` layout.

RadGenome-Chest CT ships CT-RATE re-gridded to 1 mm in plane and 3 mm through
plane, and `data_preprocessed.py` expects that grid: it does not resample to a
target spacing itself, so feeding it the raw CT-RATE volumes silently produces a
differently-scaled store. When the RadGenome release is unavailable, this script
reproduces the same grid directly from CT-RATE.

The transform is the one in new_data_preprocessed.py, with one difference: the
intensities are *not* clamped to [-1000, 1000]. The RadGenome volumes keep their
full dynamic range, and clamping here changes the adaptive window that
data_preprocessed.py fits later.

    read (ITK, LPS)  ->  trilinear resample to (z=3, y=1, x=1)  ->  flip x and y

Volumes are written with an identity affine, matching the RadGenome release; the
1 mm / 3 mm grid is implied by the pipeline rather than recorded in the header.

Example
-------
    python preprocess_ctrate.py \
        --input         data/valid/valid_fixed \
        --output        data/valid/valid_preprocessed \
        --metadata-csv  validation_metadata.csv \
        --num-workers   4

Then continue with the usual pipeline:

    python data_preprocessed.py \
        --ct-dir   data/valid/valid_preprocessed \
        --mask-dir data/valid/valid_region_mask \
        --output   valid_total_processed_data.h5

Note on voxel spacing
---------------------
Spacing is read from the volume header via ITK, or from the release metadata
when `--metadata-csv` is given, which is the more reliable source in plane.

Neither source recovers the RadGenome grid exactly for every study. On a 100
study sample of the validation split, the output shape matched the RadGenome
release for 68 studies from the header alone and 71 with the metadata; for the
matching studies the voxel values agreed to 9e-06 relative error. The remaining
studies differ only along z, because the slice thickness recorded in the
`*_fixed` release (and in the metadata, which carries the same value) is not the
one RadGenome resampled from -- their true thickness is 0.75 mm or 3 mm where
both sources say 1.5 mm.

That residual has little effect downstream, since data_preprocessed.py resizes
every volume to a fixed (64, 128, 128) grid. Zero-shot classification over the
same 100 studies scored AUROC 0.8377 / F1 0.8320 / accuracy 0.7739 from this
script with `--metadata-csv`, against 0.8382 / 0.8336 / 0.7744 from the
RadGenome volumes. Use the RadGenome release when it is available; use this
script when it is not.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm

# (z, y, x), matching the RadGenome grid.
TARGET_SPACING = (3.0, 1.0, 1.0)


def load_lps(file_path):
    """Load a volume in LPS orientation as a (C, X, Y, Z) float32 tensor."""
    import monai
    from monai.data import ITKReader

    loader = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image"], reader=ITKReader()),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Orientationd(axcodes="LPS", keys=["image"]),
        monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
    ])
    return loader({"image": file_path})["image"]


def spacing_zyx(file_path):
    """Voxel spacing as (z, y, x)."""
    spacing = sitk.ReadImage(str(file_path)).GetSpacing()  # (x, y, z)
    return spacing[2], spacing[1], spacing[0]


def read_metadata_spacing(csv_path):
    """Map each study to its (z, y, x) spacing from the CT-RATE metadata table.

    The in-plane spacing recorded here is more reliable than the header's, which
    is rounded and can shift the output extent by a voxel.
    """
    spacings = {}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            study = row["VolumeName"].replace(".nii.gz", "")
            xy = float(row["XYSpacing"].strip("[] ").split(",")[0])
            spacings[study] = (float(row["ZSpacing"]), xy, xy)
    return spacings


def resample(volume, current_spacing, target_spacing):
    """Trilinearly resample a (1, 1, D, H, W) tensor to the target voxel spacing.

    The output extent is truncated rather than rounded, so that the grid matches
    the one used to build the RadGenome release.
    """
    shape = volume.shape[2:]
    new_shape = [int(shape[i] * current_spacing[i] / target_spacing[i])
                 for i in range(len(shape))]
    return F.interpolate(volume, size=new_shape, mode="trilinear",
                         align_corners=False).cpu().numpy()


def process_one(args):
    """Resample one study. Returns a short status string."""
    in_path, out_path, meta_spacing = args
    try:
        if os.path.exists(out_path):
            return f"skipped {os.path.basename(in_path)}: already written"

        image = load_lps(in_path)
        current = meta_spacing if meta_spacing is not None else spacing_zyx(in_path)

        # (C, X, Y, Z) -> (Z, X, Y), the axis order resample() expects.
        array = image[0].cpu().numpy().transpose(2, 0, 1)
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)

        resampled = resample(tensor, current, TARGET_SPACING)[0][0]

        # Back to (X, Y, Z), then match the RadGenome axis directions.
        resampled = resampled.transpose(1, 2, 0)
        resampled = np.flip(resampled, axis=0)
        resampled = np.flip(resampled, axis=1)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        nib.save(nib.Nifti1Image(np.ascontiguousarray(resampled, dtype=np.float32),
                                 np.eye(4)), out_path)
        return f"processed {os.path.basename(in_path)}"

    except Exception as e:
        return f"failed {os.path.basename(in_path)}: {e}"


def build_jobs(input_dir, output_dir, meta_spacings):
    """Pair each input volume with its output path, mirroring the RadGenome layout.

    RadGenome nests each study as <base>/<base><letter>/<study>.nii.gz, for example
    valid_1000/valid_1000a/valid_1000_a_1.nii.gz.
    """
    jobs = []
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if not name.endswith(".nii.gz"):
                continue
            study = name[: -len(".nii.gz")]
            parts = study.split("_")
            if len(parts) < 3:
                continue
            base = "_".join(parts[:2])          # valid_1000
            nested = f"{base}{parts[2]}"        # valid_1000a
            jobs.append((os.path.join(root, name),
                         os.path.join(output_dir, base, nested, name),
                         meta_spacings.get(study)))
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resample raw CT-RATE volumes onto the RadGenome grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Directory searched recursively for <study>.nii.gz")
    parser.add_argument("--output", required=True,
                        help="Destination root, written in the RadGenome layout")
    parser.add_argument("--metadata-csv", default=None,
                        help="CT-RATE metadata table (e.g. validation_metadata.csv). "
                             "Its spacing is preferred over the volume header where "
                             "a study is listed")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Worker processes")
    return parser.parse_args()


def main():
    args = parse_args()

    meta_spacings = {}
    if args.metadata_csv:
        meta_spacings = read_metadata_spacing(args.metadata_csv)
        print(f"read spacing for {len(meta_spacings)} studies from {args.metadata_csv}")

    jobs = build_jobs(args.input, args.output, meta_spacings)
    print(f"found {len(jobs)} volumes under {args.input}")
    if not jobs:
        return

    if meta_spacings:
        missing = sum(1 for job in jobs if job[2] is None)
        if missing:
            print(f"{missing} volumes are absent from the metadata; "
                  "their spacing falls back to the volume header")

    failures = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(process_one, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="resampling"):
            result = future.result()
            if not result.startswith("processed"):
                failures += 1
                tqdm.write(result)

    print(f"done, written to {args.output}")
    if failures:
        print(f"{failures} volumes were skipped or failed; see the messages above")
    print("next step: build the HDF5 store with data_preprocessed.py")


if __name__ == "__main__":
    main()
