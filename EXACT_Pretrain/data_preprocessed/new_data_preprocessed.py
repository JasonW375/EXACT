#!/usr/bin/env python3
"""Preprocess external CT cohorts (RAD-ChestCT, MianYang) into an HDF5 store.

Accepts .mha and .nii/.nii.gz volumes, resamples them to the training geometry and
writes one group per study containing a `ct` array of shape (1, 64, 128, 128).

Unlike data_preprocessed.py, no organ masks are read: external cohorts are only
used for diagnosis, which needs the image and the label vector alone.

Example
-------
    python new_data_preprocessed.py \
        --input  /path/to/rad_chest/volumes \
        --output rad_chest_processed.h5 \
        --num_processes 4

The store is appended to, so an interrupted run can be restarted; pass
--force_reprocess to rebuild from scratch.

Orientation
-----------
The two readers below do NOT flip identically: preprocess_mha_file flips axes 2 and
1, while preprocess_nii_file flips only axis 2. This asymmetry is deliberate and
reflects how the source cohorts are stored. Cohorts that still do not match the
training orientation afterwards are corrected with flip_data.py, whose axis has to
be chosen by inspection.

Disease labels are attached afterwards by save_label_18.py.
"""

# Thread limits must be set before the numerical libraries are imported.
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import warnings
from multiprocessing import Manager, Pool
from pathlib import Path

import cv2
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio as tio
from tqdm import tqdm

warnings.filterwarnings("ignore")

torch.set_num_threads(2)
torch.set_num_interop_threads(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def strip_known_suffixes(name: str) -> str:
    """Strip image suffixes, including the double .nii.gz."""
    n = name
    while True:
        nl = n.lower()
        matched = False
        for sfx in (".nii.gz", ".nii", ".mha", ".mhd"):
            if nl.endswith(sfx):
                n = n[: -len(sfx)]
                matched = True
                break
        if not matched:
            break
    return n


def normalize_stem(p) -> str:
    """HDF5 group name for a volume path."""
    return strip_known_suffixes(Path(p).name)


def resize_array(array, current_spacing, target_spacing):
    """Trilinearly resample a (1, 1, D, H, W) tensor to the target voxel spacing."""
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear',
                                  align_corners=False).cpu().numpy()
    return resized_array


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def _load_as_tensor(file_path):
    """Load a volume in LPS orientation as a (C, D, H, W) float tensor."""
    import monai
    from monai.data import ITKReader

    monai_loader = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=['image'], reader=ITKReader()),
        monai.transforms.EnsureChannelFirstd(keys=['image']),
        monai.transforms.Orientationd(axcodes="LPS", keys=['image']),
        monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
    ])
    dictionary = monai_loader({'image': file_path})
    return dictionary['image']


def _get_spacing_from_itk(file_path):
    """Voxel spacing as (z, y, x)."""
    image = sitk.ReadImage(str(file_path))
    spacing = image.GetSpacing()  # (x, y, z)
    return spacing[2], spacing[1], spacing[0]


def preprocess_mha_file(file_path):
    """Read and resample an .mha volume."""
    file_path_str = str(file_path)

    img_data = _load_as_tensor(file_path_str)
    current = _get_spacing_from_itk(file_path_str)
    target = (3.0, 1.0, 1.0)

    img_data = torch.clamp(img_data, min=-1000, max=1000)

    img_np = img_data[0].cpu().numpy()
    img_np = img_np.transpose(2, 0, 1)
    tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)

    resized_array = resize_array(tensor, current, target)
    resized_array = resized_array[0][0]

    # Two flips here, one in the NIfTI path below - see the module docstring.
    resized_array = np.flip(resized_array, axis=2)
    resized_array = np.flip(resized_array, axis=1)

    return {
        'data': resized_array.astype(np.float32),
        'spacing': (np.float32(1.0), np.float32(1.0), np.float32(1.0)),
        'original_spacing': current,
    }


def preprocess_nii_file(file_path):
    """Read and resample a .nii/.nii.gz volume."""
    file_path_str = str(file_path)

    img_data = _load_as_tensor(file_path_str)
    current = _get_spacing_from_itk(file_path_str)
    target = (3.0, 1.0, 1.0)

    img_data = torch.clamp(img_data, min=-1000, max=1000)

    img_np = img_data[0].cpu().numpy()
    img_np = img_np.transpose(2, 0, 1)
    tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)

    resized_array = resize_array(tensor, current, target)
    resized_array = resized_array[0][0]

    # Only one flip, unlike the .mha path.
    resized_array = np.flip(resized_array, axis=2)

    return {
        'data': resized_array.astype(np.float32),
        'spacing': (np.float32(1.0), np.float32(1.0), np.float32(1.0)),
        'original_spacing': current,
    }


# ---------------------------------------------------------------------------
# Intensity normalisation
# ---------------------------------------------------------------------------
def adaptive_windowing(image, window_min=-2000, window_max=1000):
    """Clip to the volume's own 0.5 / 99.5 percentiles and scale to uint8."""
    img_filtered = np.array(image, dtype=np.float32)
    img_filtered[img_filtered < img_filtered.min() + 100] = np.nan

    min_display = np.nanpercentile(img_filtered, 0.5)
    max_display = np.nanpercentile(img_filtered, 99.5)

    img_windowed = np.clip(image, min_display, max_display)
    img_windowed = ((img_windowed - min_display) / (max_display - min_display) * 255).astype(np.uint8)

    return img_windowed


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Slice-wise contrast-limited adaptive histogram equalisation."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_slices = []
    for i in range(img.shape[0]):
        slice_img = img[i, :, :]
        clahe_img = clahe.apply(slice_img)
        clahe_slices.append(clahe_img)
    return np.stack(clahe_slices, axis=0)


def normalize_data(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    image_windowed = adaptive_windowing(image)
    image_clahe = apply_clahe(image_windowed, clip_limit, tile_grid_size)
    image_normalized = image_clahe.astype(np.float32) / 255.0
    image_normalized = np.expand_dims(image_normalized, axis=0)
    return image_normalized


def resize_data(image, target_shape=(64, 128, 128)):
    resize_transform = tio.Resize(target_shape)
    return resize_transform(image)


def resample_data(image, target_spacing=(1, 1, 1)):
    resample_transform = tio.Resample(target_spacing)
    return resample_transform(image)


# ---------------------------------------------------------------------------
# Per-study worker
# ---------------------------------------------------------------------------
def process_single_file(args):
    file_path, output_path, visualize_dir, should_visualize, process_lock = args

    try:
        file_path_str = str(file_path)
        file_key = normalize_stem(file_path)

        if file_path_str.lower().endswith('.mha'):
            processed_data = preprocess_mha_file(file_path)
        elif file_path_str.lower().endswith(('.nii.gz', '.nii')):
            processed_data = preprocess_nii_file(file_path)
        else:
            return f"unsupported format: {file_path.name}", None

        ct_img = processed_data['data']
        spacing = processed_data['spacing']

        ct_normalized = normalize_data(ct_img)

        ct_img_tensor = torch.from_numpy(ct_normalized)
        ct_subject = tio.Subject(
            image=tio.ScalarImage(tensor=ct_img_tensor, spacing=spacing)
        )

        ct_resampled_subject = resample_data(ct_subject.image)
        ct_resized_subject = resize_data(ct_resampled_subject, target_shape=(64, 128, 128))

        ct_tensor = ct_resized_subject.data.numpy()

        # One writer at a time: HDF5 is not safe for concurrent appends.
        with process_lock:
            with h5py.File(output_path, 'a') as hf:
                if file_key not in hf:
                    grp = hf.create_group(file_key)
                    grp.create_dataset("ct", data=ct_tensor, compression="gzip", dtype="float32")
                    grp.create_dataset("original_spacing", data=processed_data['original_spacing'])
                    grp.create_dataset("final_spacing", data=spacing)
                    grp.attrs['filename'] = file_path.name

        if should_visualize and visualize_dir:
            save_as_nifti(ct_tensor, Path(visualize_dir) / f"{file_key}_preprocessed.nii.gz")

        return f"ok {file_path.name}", file_key

    except Exception as e:
        return f"failed {file_path.name}: {e}", None


def save_as_nifti(data, output_path):
    """Write a preprocessed volume out for visual inspection."""
    affine = np.eye(4)

    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]

    data_transposed = np.transpose(data, (1, 2, 0))
    nib.save(nib.Nifti1Image(data_transposed, affine), output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess external CT cohorts into an HDF5 store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Directory of .mha / .nii / .nii.gz volumes")
    parser.add_argument("--output", type=str, required=True,
                        help="HDF5 file to create or append to")
    parser.add_argument("--visualize_dir", type=str, default=None,
                        help="Write the first N preprocessed volumes here as NIfTI "
                             "so the orientation can be checked")
    parser.add_argument("--num_visualize", type=int, default=10,
                        help="How many volumes to write to --visualize_dir")
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Worker processes")
    parser.add_argument("--force_reprocess", action="store_true",
                        help="Rebuild from scratch instead of resuming")

    args = parser.parse_args()

    input_path = Path(args.input)
    all_files = sorted([p for p in input_path.iterdir()
                        if p.suffix.lower() in {".mha", ".nii", ".gz"}
                        or (p.suffix.lower() == ".gz" and p.with_suffix("").suffix.lower() == ".nii")])

    if not all_files:
        raise SystemExit(f"no CT volumes found in {args.input}")

    processed_files = set()
    output_path = Path(args.output)

    if not args.force_reprocess and output_path.exists():
        with h5py.File(output_path, 'r') as hf:
            processed_files = set(hf.keys())

    all_file_keys = {normalize_stem(f) for f in all_files}
    remaining_file_keys = all_file_keys - processed_files
    files_to_process = [f for f in all_files if normalize_stem(f) in remaining_file_keys]

    print(f"found {len(all_files)} volumes under {args.input}")
    print(f"already in {output_path}: {len(processed_files)}")
    print(f"to process: {len(files_to_process)} using {args.num_processes} workers")

    if not files_to_process:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        with h5py.File(output_path, 'w'):
            pass

    if args.visualize_dir:
        Path(args.visualize_dir).mkdir(parents=True, exist_ok=True)

    manager = Manager()
    process_lock = manager.Lock()

    args_list = []
    for idx, file_path in enumerate(files_to_process):
        should_visualize = bool(args.visualize_dir) and (idx < args.num_visualize)
        args_list.append((file_path, str(output_path), args.visualize_dir,
                          should_visualize, process_lock))

    successful = failed = 0
    with Pool(processes=args.num_processes) as pool:
        for result, file_key in tqdm(pool.imap_unordered(process_single_file, args_list),
                                     total=len(args_list), desc="preprocessing"):
            if file_key:
                successful += 1
            else:
                failed += 1
                tqdm.write(result)

    print(f"done: {successful} succeeded, {failed} failed, written to {output_path}")
    if args.visualize_dir:
        print(f"inspect {args.visualize_dir} and correct the orientation with "
              f"flip_data.py if it does not match the training cohort")
    print("next step: attach disease labels with save_label_18.py")


if __name__ == "__main__":
    main()
