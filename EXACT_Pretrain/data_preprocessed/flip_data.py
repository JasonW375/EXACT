#!/usr/bin/env python3
"""Flip the CT volumes in an HDF5 store along one axis.

External cohorts do not always end up in the same orientation as the training data
after new_data_preprocessed.py, because the .mha and .nii readers there apply
different flips. Use --preview first to inspect a handful of studies as three
orthogonal views, then rerun without --preview to write the corrected store.

Only the `ct` dataset is flipped; labels and any other datasets or attributes are
copied through unchanged.

Example
-------
    # 1. inspect the effect on the first ten studies
    python flip_data.py --input data.h5 --preview --axis 1

    # 2. once the orientation looks right, write the flipped store
    python flip_data.py --input data.h5 --output data_flipped.h5 --axis 1

Axis convention for a stored volume of shape (1, 64, 128, 128):
    axis=0  channel   (not normally useful)
    axis=1  Z, 64 slices  - head/feet
    axis=2  Y, 128 rows   - anterior/posterior
    axis=3  X, 128 cols   - left/right
"""

import argparse
import shutil
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AXIS_NAMES = {0: "Channel", 1: "Z-axis (Axial)", 2: "Y-axis (Coronal)", 3: "X-axis (Sagittal)"}


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
def _extract_orthogonal_slices(ct_data):
    """Centre axial, sagittal and coronal slices of a (1, D, H, W) or (D, H, W) volume."""
    if ct_data.ndim == 4:
        data = ct_data[0]
    elif ct_data.ndim == 3:
        data = ct_data
    else:
        raise ValueError(f"Expected a 3D or 4D volume, got {ct_data.ndim} dimensions")

    d, h, w = data.shape
    return {
        "axial": data[d // 2, :, :],
        "sagittal": data[:, :, w // 2],
        "coronal": data[:, h // 2, :],
    }


def _create_comparison_figure(ct_original, ct_flipped, flip_axis, file_key):
    """Three orthogonal views before and after the flip, with per-view difference."""
    slices_orig = _extract_orthogonal_slices(ct_original)
    slices_flip = _extract_orthogonal_slices(ct_flipped)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Flip preview: {file_key}\n"
        f"Shape: {ct_original.shape} | Flip axis: {flip_axis} "
        f"({AXIS_NAMES.get(flip_axis, f'axis {flip_axis}')})",
        fontsize=14, fontweight="bold",
    )

    view_names = ["Axial (Z)", "Sagittal (X)", "Coronal (Y)"]
    view_keys = ["axial", "sagittal", "coronal"]

    for row, (title, slices) in enumerate((("Original", slices_orig), ("Flipped", slices_flip))):
        for i, (view_name, view_key) in enumerate(zip(view_names, view_keys)):
            ax = axes[row, i]
            img = slices[view_key]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{title} - {view_name}", fontsize=11, fontweight="bold")
            ax.axis("off")

            h, w = img.shape
            ax.text(5, 20, f"{h}x{w}", color="yellow", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

            if row == 1:
                diff = np.abs(slices_orig[view_key] - img)
                ax.text(w / 2, h - 10,
                        f"Diff: mean={diff.mean():.4f}, max={diff.max():.4f}",
                        color="cyan", fontsize=8, ha="center",
                        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

    plt.tight_layout()
    return fig


def visualize_flip_preview(input_path, flip_axis, num_samples=10, output_dir="flip_preview"):
    """Render before/after comparisons for the first `num_samples` studies."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"input      : {input_path}")
    print(f"flip axis  : {flip_axis} ({AXIS_NAMES.get(flip_axis, 'custom')})")
    print(f"output dir : {output_dir}")

    with h5py.File(input_path, "r") as hf:
        all_keys = list(hf.keys())
        file_keys = all_keys[:num_samples]
        print(f"studies    : {len(all_keys)} total, previewing {len(file_keys)}\n")

        for idx, file_key in enumerate(tqdm(file_keys, desc="rendering")):
            grp = hf[file_key]
            if "ct" not in grp:
                tqdm.write(f"skipped {file_key}: no 'ct' dataset")
                continue

            ct_original = grp["ct"][:]
            ct_flipped = np.flip(ct_original, axis=flip_axis)

            if idx == 0:
                print(f"\nshape {ct_original.shape}, dtype {ct_original.dtype}, "
                      f"range [{ct_original.min():.2f}, {ct_original.max():.2f}], "
                      f"axis {flip_axis} has size {ct_original.shape[flip_axis]}\n")

            fig = _create_comparison_figure(ct_original, ct_flipped, flip_axis, file_key)
            plt.savefig(output_dir / f"{idx + 1:02d}_{file_key}_flip_preview.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"\nWrote {len(file_keys)} previews to {output_dir}. If the orientation looks "
          f"right, rerun without --preview to write the flipped store.")


# ---------------------------------------------------------------------------
# Flip
# ---------------------------------------------------------------------------
def _process_h5_groups(hf_in, hf_out, flip_axis, verify):
    """Copy every study into a new store, flipping the `ct` dataset."""
    file_keys = list(hf_in.keys())

    for file_key in tqdm(file_keys, desc="flipping"):
        grp_in = hf_in[file_key]
        grp_out = hf_out.create_group(file_key)

        if "ct" in grp_in:
            ct_data = grp_in["ct"][:]
            ct_flipped = np.flip(ct_data, axis=flip_axis)

            if verify and file_key == file_keys[0]:
                print(f"\nshape {ct_data.shape} -> {ct_flipped.shape}")
                print(f"range [{ct_data.min():.2f}, {ct_data.max():.2f}] -> "
                      f"[{ct_flipped.min():.2f}, {ct_flipped.max():.2f}]")

                # The first slice along the flip axis must become the last one.
                axis_size = ct_data.shape[flip_axis]
                original_first = np.take(ct_data, 0, axis=flip_axis)
                flipped_last = np.take(ct_flipped, axis_size - 1, axis=flip_axis)
                if np.allclose(original_first, flipped_last):
                    print("check ok: first slice moved to the end\n")
                else:
                    print("WARNING: flip check failed\n")

            grp_out.create_dataset("ct", data=ct_flipped, compression="gzip", dtype="float32")

        for key in grp_in.keys():
            if key != "ct":
                grp_out.create_dataset(key, data=grp_in[key][:])

        for attr_key, attr_val in grp_in.attrs.items():
            grp_out.attrs[attr_key] = attr_val


def _process_h5_inplace(hf, flip_axis, verify):
    """Flip in place: read every volume, then rewrite the `ct` datasets."""
    file_keys = list(hf.keys())

    print("step 1/2: reading")
    all_data = {}
    for file_key in tqdm(file_keys, desc="reading"):
        grp = hf[file_key]
        if "ct" in grp:
            all_data[file_key] = np.flip(grp["ct"][:], axis=flip_axis)

    print("step 2/2: writing")
    for file_key in tqdm(file_keys, desc="writing"):
        if file_key in all_data:
            grp = hf[file_key]
            if "ct" in grp:
                del grp["ct"]
            grp.create_dataset("ct", data=all_data[file_key], compression="gzip", dtype="float32")


def flip_ct_in_h5(input_path, output_path, flip_axis=1, inplace=False, verify=True):
    """Flip the `ct` dataset of every study along `flip_axis`."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    backup_path = None
    if inplace:
        backup_path = input_path.with_suffix(".h5.backup")
        print(f"backing up to {backup_path}")
        shutil.copy2(input_path, backup_path)
        output_path = input_path
    else:
        if output_path.resolve() == input_path.resolve():
            raise ValueError("--output is the input file; pass --inplace instead")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as hf_in:
        total_files = len(hf_in.keys())

    print(f"input     : {input_path}")
    print(f"output    : {output_path}")
    print(f"flip axis : {flip_axis} ({AXIS_NAMES.get(flip_axis, 'custom')})")
    print(f"studies   : {total_files}")

    if inplace:
        with h5py.File(input_path, "r+") as hf:
            _process_h5_inplace(hf, flip_axis, verify)
    else:
        with h5py.File(input_path, "r") as hf_in, h5py.File(output_path, "w") as hf_out:
            _process_h5_groups(hf_in, hf_out, flip_axis, verify)

    print(f"\ndone, written to {output_path}")
    if backup_path:
        print(f"original preserved at {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Flip the CT volumes in an HDF5 store along one axis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input HDF5 store")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HDF5 store (default: <input>_flipped.h5)")
    parser.add_argument("--axis", type=int, default=1,
                        help="Axis to flip; 1 is Z for a (1, 64, 128, 128) volume")
    parser.add_argument("--inplace", action="store_true",
                        help="Modify the input file, keeping a .backup copy")
    parser.add_argument("--preview", action="store_true",
                        help="Only render before/after views; do not modify any file")
    parser.add_argument("--num_visualize", type=int, default=10,
                        help="Studies to render in preview mode")
    parser.add_argument("--preview_dir", type=str, default="flip_preview",
                        help="Directory for the preview images")

    args = parser.parse_args()

    if args.preview:
        visualize_flip_preview(
            input_path=args.input,
            flip_axis=args.axis,
            num_samples=args.num_visualize,
            output_dir=args.preview_dir,
        )
        return

    if args.output is None and not args.inplace:
        input_path = Path(args.input)
        args.output = input_path.parent / f"{input_path.stem}_flipped.h5"

    flip_ct_in_h5(
        input_path=args.input,
        output_path=args.output if not args.inplace else args.input,
        flip_axis=args.axis,
        inplace=args.inplace,
        verify=True,
    )


if __name__ == "__main__":
    main()
