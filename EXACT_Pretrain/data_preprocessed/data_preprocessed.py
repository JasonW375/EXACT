"""Build the training/evaluation HDF5 store from RadGenome-Chest CT.

Inputs are taken directly from the RadGenome-Chest CT release, which already ships
windowed CT volumes and the nine per-region organ masks, so no separate
segmentation step is required:

    <ct-dir>/**/<study>.nii.gz
    <mask-dir>/seg_<study>/{lung,trachea and bronchie,pleura,mediastinum,heart,
                            esophagus,bone,thyroid,abdomen}.nii.gz

The CT volumes must be the RadGenome `*_preprocessed` release, which is already
resampled to 1 mm isotropic, rather than the raw CT-RATE `*_fixed` volumes: this
script does not itself resample to a target spacing, so feeding it the raw volumes
silently produces a differently-scaled store.

Each study becomes one HDF5 group holding

    ct    float32 (1, 64, 128, 128)
    mask  bool    (9, 64, 128, 128)   only when --mask-dir is given

Masks are needed for pre-training and for the segmentation branch. Evaluation does
not use them -- test.py reads `ct` only -- so --mask-dir may be omitted to build a
CT-only store for inference on this cohort.

Disease labels are attached afterwards by save_label_18.py; the dataloaders expect
a `label_18` entry that this script does not write.

The store is appended to, so interrupted runs can simply be restarted: studies
already present are skipped.

Examples
--------
    # Pre-training / segmentation: CT plus organ masks.
    python data_preprocessed.py \
        --ct-dir   data/valid/valid_preprocessed \
        --mask-dir data/valid/valid_region_mask \
        --output   valid_total_processed_data.h5

    # Evaluation only: CT alone, no masks required.
    python data_preprocessed.py \
        --ct-dir data/valid/valid_preprocessed \
        --output valid_ct_only.h5

Intensity normalisation is adaptive rather than a fixed HU window: each volume is
clipped to its own 0.5 / 99.5 percentiles before CLAHE. Intensities are therefore
comparable within a volume but not across volumes, and the mapping cannot be
reproduced from a fixed window specification.
"""

import argparse
import os
from multiprocessing import Manager, Pool

import cv2
import h5py
import nibabel as nib
import numpy as np
import torchio as tio
from tqdm import tqdm

# Channel order of the stored mask; the dataloader selects the first six plus a
# derived global channel.
REGION_NAMES = [
    "lung", "trachea and bronchie", "pleura", "mediastinum", "heart",
    "esophagus", "bone", "thyroid", "abdomen",
]


def normalize_data(image, mask, clip_limit=2.0, tile_grid_size=(8, 8),
                   window_min=-2000, window_max=1000):
    def adaptive_windowing(img, min_val=-2000, max_val=1000):
        # float32 rather than the input dtype: the percentile search below marks
        # the air background with NaN, which an integer array cannot hold.
        img_filtered = np.array(img, dtype=np.float32)
        img_filtered[img_filtered < img_filtered.min() + 100] = np.nan
        min_display = np.nanpercentile(img_filtered, 0.5)
        max_display = np.nanpercentile(img_filtered, 99.5)
        img_windowed = np.clip(img, min_display, max_display)
        img_windowed = ((img_windowed - min_display) / (max_display - min_display) * 255).astype(np.uint8)
        return img_windowed

    def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe_slices = []
        for i in range(img.shape[0]):
            slice_img = img[i, :, :]
            clahe_img = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(slice_img)
            clahe_slices.append(clahe_img)
        return np.stack(clahe_slices, axis=0)

    def normalize(img):
        return img.astype(np.float32) / 255.0

    image_windowed = adaptive_windowing(image, window_min, window_max)
    image_clahe = apply_clahe(image_windowed, clip_limit, tile_grid_size)
    image_normalized = normalize(image_clahe)
    image_normalized = np.expand_dims(image_normalized, axis=0)
    return image_normalized, mask


def resize_data(image, mask, target_shape=(64, 128, 128)):
    resize_transform = tio.Resize(target_shape)
    image = resize_transform(image)
    if mask is not None:
        mask = resize_transform(mask)
    return image, mask


def resample_data(image, mask, target_spacing=(1, 1, 1)):
    resample_transform = tio.Resample(target_spacing)
    image = resample_transform(image)
    if mask is not None:
        mask = resample_transform(mask)
    return image, mask


def process_single_file(args):
    """Preprocess one study and append it to the store."""
    file_name, ct_base_path, mask_base_path, output_path, region_names, process_lock = args

    try:
        ct_file_path = None
        for root, dirs, files in os.walk(ct_base_path):
            for file in files:
                if file == f"{file_name}.nii.gz":
                    ct_file_path = os.path.join(root, file)
                    break
            if ct_file_path:
                break

        if not ct_file_path:
            return f"skipped {file_name}: CT volume not found"

        ct_img = nib.load(ct_file_path).get_fdata()
        ct_img = np.transpose(ct_img, (2, 0, 1))

        masks_4d = None
        if mask_base_path is not None:
            mask_folder = os.path.join(mask_base_path, f"seg_{file_name}")
            if not os.path.isdir(mask_folder):
                return f"skipped {file_name}: mask folder seg_{file_name} not found"

            masks = []
            for region in region_names:
                mask_file_path = os.path.join(mask_folder, f"{region}.nii.gz")
                if os.path.exists(mask_file_path):
                    mask_img = nib.load(mask_file_path).get_fdata()
                    mask_img = np.transpose(mask_img, (2, 0, 1))
                    masks.append(mask_img.astype(bool))
                else:
                    return f"skipped {file_name}: missing mask {region}.nii.gz"

            if len(masks) != len(region_names):
                return f"skipped {file_name}: incomplete mask set"

            masks_4d = np.stack(masks, axis=0)

        ct_img, masks_4d = normalize_data(ct_img, masks_4d)
        ct_img, masks_4d = resample_data(ct_img, masks_4d)
        ct_img, masks_4d = resize_data(ct_img, masks_4d)

        # One writer at a time: HDF5 is not safe for concurrent appends.
        with process_lock:
            with h5py.File(output_path, "a") as h5f:
                if file_name not in h5f:
                    grp = h5f.create_group(file_name)
                    grp.create_dataset("ct", data=ct_img.astype(np.float32),
                                       compression="gzip", dtype="float32")
                    if masks_4d is not None:
                        grp.create_dataset("mask", data=masks_4d,
                                           compression="gzip", dtype="bool")

        return f"processed {file_name}"

    except Exception as e:
        return f"failed {file_name}: {e}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess RadGenome-Chest CT volumes and organ masks into an HDF5 store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ct-dir", required=True,
                        help="Directory searched recursively for <study>.nii.gz")
    parser.add_argument("--mask-dir", default=None,
                        help="Directory holding one seg_<study>/ folder per study. "
                             "Required for pre-training and the segmentation branch; "
                             "omit it to build a CT-only store for evaluation")
    parser.add_argument("--output", required=True,
                        help="HDF5 file to create or append to")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Worker processes")
    return parser.parse_args()


def main():
    args = parse_args()

    processed_files = set()
    if os.path.exists(args.output):
        with h5py.File(args.output, "r") as h5f:
            processed_files = set(h5f.keys())

    all_files = set()
    for root, dirs, files in os.walk(args.ct_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                file_name = os.path.splitext(os.path.splitext(file)[0])[0]
                all_files.add(file_name)

    remaining_files = sorted(all_files - processed_files)

    print(f"found {len(all_files)} studies under {args.ct_dir}")
    print(f"already in {args.output}: {len(processed_files)}")
    print(f"to process: {len(remaining_files)}")
    if args.mask_dir is None:
        print("no --mask-dir given: writing CT only, which is enough for test.py "
              "but not for pre-training")

    if not remaining_files:
        return

    manager = Manager()
    process_lock = manager.Lock()

    args_list = [(file_name, args.ct_dir, args.mask_dir, args.output,
                  REGION_NAMES, process_lock)
                 for file_name in remaining_files]

    failures = 0
    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_file, args_list),
                           total=len(args_list), desc="preprocessing"):
            if not result.startswith("processed"):
                failures += 1
                tqdm.write(result)

    print(f"done, written to {args.output}")
    if failures:
        print(f"{failures} studies were skipped or failed; see the messages above")
    print("next step: attach disease labels with save_label_18.py")


if __name__ == "__main__":
    main()
