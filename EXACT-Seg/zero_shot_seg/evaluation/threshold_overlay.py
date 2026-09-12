import os
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="Binarize the averaged heatmaps written by overlay_heatmap.py."
    )
    # Thresholding mode: abs = absolute threshold; rel = relative threshold; both = intersection of both
    p.add_argument(
        "--thresh-mode",
        choices=["abs", "rel", "both"],
        default="abs",
        help="Binarization mode: abs = absolute threshold; rel = relative threshold; both = intersection of both.",
    )
    p.add_argument(
        "--binary-threshold",
        type=float,
        default=0.004,
        help="Absolute threshold in [0, 1] (used when --thresh-mode includes abs).",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="Relative threshold ratio in (0, 1], e.g., 0.2 means top 20% highest values (used when --thresh-mode includes rel).",
    )
    p.add_argument(
        "--in-overlay",
        required=True,
        help="Input directory of averaged heatmaps (reads *_overlaid_heatmap.nii.gz).",
    )
    p.add_argument(
        "--out-seg",
        required=True,
        help="Output directory for the binary masks.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite target files if they already exist.",
    )
    return p.parse_args()


def topk_threshold(vol: np.ndarray, ratio: float) -> float:
    """Return the threshold such that approximately ratio of voxels fall into the top-valued set (computed only on voxels > 0)."""
    flat = vol.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return np.inf
    nonzero = finite[finite > 0]
    if nonzero.size == 0:
        return np.inf
    k = max(1, int(np.ceil(ratio * nonzero.size)))
    idx = nonzero.size - k  # Smallest value among the top-k largest values
    thr = float(np.partition(nonzero, idx)[idx])
    return thr


def main():
    args = parse_args()

    if "abs" in args.thresh_mode:
        if not (0.0 <= args.binary_threshold <= 1.0):
            raise ValueError("--binary-threshold must be in [0, 1].")
    if "rel" in args.thresh_mode:
        if not (0.0 < args.ratio <= 1.0):
            raise ValueError("--ratio must be in (0, 1], e.g., 0.2 means the top 20%.")

    # Only create the segmentation output directory; average heatmaps are no longer written
    os.makedirs(args.out_seg, exist_ok=True)

    in_dir = args.in_overlay
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Pre-overlaid heatmap directory not found: {in_dir}")

    # Prefer *_overlaid_heatmap.nii.gz; if none exist, fall back to all .nii.gz files
    overlay_files = sorted(glob.glob(os.path.join(in_dir, "*_overlaid_heatmap.nii.gz")))
    if not overlay_files:
        overlay_files = sorted(glob.glob(os.path.join(in_dir, "*.nii.gz")))
    if not overlay_files:
        raise FileNotFoundError(f"No .nii.gz files were found in directory: {in_dir}")

    # Build {sample_id: fullpath}
    file_map = {}
    for ofile in overlay_files:
        base = os.path.basename(ofile)
        sid = base.replace("_overlaid_heatmap.nii.gz", "")
        if sid.endswith(".nii.gz"):
            sid = sid[:-7]
        file_map[sid] = ofile

    selected_ids = sorted(file_map.keys())
    if not selected_ids:
        print("No samples available for processing.")
        return

    total_samples = len(selected_ids)
    success_samples = 0

    for sid in tqdm(selected_ids, desc="thresholding"):
        ofile = file_map[sid]
        try:
            img = nib.load(ofile)
            avg_vol = img.get_fdata().astype(np.float32)
            affine = img.affine
            header = img.header.copy()

            # Absolute-threshold mask
            mask_abs = None
            if args.thresh_mode in ("abs", "both"):
                mask_abs = (avg_vol >= args.binary_threshold).astype(np.uint8)

            # Relative-threshold mask
            mask_rel = None
            if args.thresh_mode in ("rel", "both"):
                thr_rel = topk_threshold(avg_vol, args.ratio)
                if np.isfinite(thr_rel):
                    mask_rel = (avg_vol >= thr_rel).astype(np.uint8)
                else:
                    mask_rel = np.zeros_like(avg_vol, dtype=np.uint8)

            # Compose final mask
            if args.thresh_mode == "abs":
                seg_mask = mask_abs
            elif args.thresh_mode == "rel":
                seg_mask = mask_rel
            else:  # both -> intersection
                if mask_abs is None or mask_rel is None:
                    seg_mask = np.zeros_like(avg_vol, dtype=np.uint8)
                else:
                    seg_mask = (mask_abs & mask_rel).astype(np.uint8)

            seg_header = header.copy()
            seg_header.set_data_dtype(np.uint8)
            seg_path = os.path.join(args.out_seg, f"{sid}.nii.gz")
            if args.overwrite or (not os.path.exists(seg_path)):
                nib.save(nib.Nifti1Image(seg_mask, affine, seg_header), seg_path)
                success_samples += 1
        except Exception as e:
            print(f"Error processing {sid}: {e}")

    print(f"Done: thresholded segmentation for {success_samples}/{total_samples} samples")
    print(f"Input average heatmap directory -> {in_dir}")
    print(f"Segmentation output directory -> {args.out_seg}")


if __name__ == "__main__":
    main()
