import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
import torchio as tio


def normalize_stem(p) -> str:
    """Extract the filename stem from a path (without extension)."""
    return Path(p).stem.replace('.nii', '')  # Handle .nii.gz cases


def read_mha_files(directory):
    """Read all .mha files in a directory."""
    mha_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".mha"):
                mha_files.append(os.path.join(root, f))
    return mha_files


def resize_array(array, current_spacing, target_spacing):
    """Resize an array using nearest-neighbor interpolation."""
    original_shape = array.shape[2:]  # (D, H, W)
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    # Use nearest-neighbor interpolation instead of trilinear interpolation
    resized_array = F.interpolate(array, size=new_shape, mode='nearest').cpu().numpy()
    return resized_array


def _load_mha_as_tensor(file_path):
    """Load an MHA file as a tensor using MONAI."""
    monitor_pid("_load_mha_as_tensor start")
    import monai
    from monai.data import ITKReader

    monai_loader = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=['image'], reader=ITKReader()),
            monai.transforms.EnsureChannelFirstd(keys=['image']),
            monai.transforms.Orientationd(axcodes="LPS", keys=['image']),
            monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )
    dictionary = monai_loader({'image': file_path})
    monitor_pid("_load_mha_as_tensor end")
    return dictionary['image']  # (C, D, H, W)


def _get_spacing_from_itk(file_path):
    """Get voxel spacing from an ITK image."""
    monitor_pid("_get_spacing_from_itk start")
    image = sitk.ReadImage(str(file_path))
    spacing = image.GetSpacing()  # (x, y, z)
    monitor_pid("_get_spacing_from_itk end")
    return spacing[2], spacing[1], spacing[0]  # Return (z, y, x)


def first_stage_preprocess(file_path):
    """First-stage preprocessing: process multi-channel mask files and merge channels."""
    monitor_pid("first-stage preprocessing start")

    file_path_str = str(file_path)
    key = normalize_stem(file_path_str)

    try:
        print(f"Using first-stage preprocessing: MHA file conversion - {file_path_str}")

        # Load the MHA file
        img_data = _load_mha_as_tensor(file_path_str)  # (C, D, H, W), i.e. (C, x, y, z)
        print(f"Original data shape: {img_data.shape}")

        # Get voxel spacing
        current = _get_spacing_from_itk(file_path_str)  # (z, y, x)
        target = (3.0, 1.0, 1.0)
        print(f"Original voxel spacing: {current}")

        # Take the union of all channels (logical OR)
        # Merge multiple lesion masks into a single mask
        mask_union = torch.max(img_data, dim=0, keepdim=False)[0]  # (D, H, W), i.e. (x, y, z)

        # Ensure binary mask (0 or 1)
        mask_union = (mask_union > 0).float()

        # Convert to numpy and reorder axes
        img_np = mask_union.cpu().numpy()                 # (D, H, W), i.e. (x, y, z)
        img_np = img_np.transpose(2, 0, 1)               # (W, D, H), i.e. (z, x, y)
        tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)  # (1, 1, z, x, y)

        # Resample to target spacing
        resized_array = resize_array(tensor, current, target)  # (1, 1, z', x', y')
        resized_array = resized_array[0][0]                    # (z', x', y')

        # Flip along y-axis while keeping (z, x, y) order
        resized_array = np.flip(resized_array, axis=2)

        # Flip along x-axis
        resized_array = np.flip(resized_array, axis=1)

        print(f"First-stage preprocessing completed: shape={resized_array.shape}")

        # Return processed array in (z, x, y) order
        monitor_pid("first-stage preprocessing end")
        return {
            'data': resized_array.astype(np.float32),  # (z, x, y)
            'spacing': (np.float32(1.0), np.float32(1.0), np.float32(1.0)),  # Standardized voxel spacing
        }

    except Exception as e:
        print(f"First-stage preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def preprocess_nii_file(file_path):
    """Process a NIfTI file (supports multi-channel masks)."""
    monitor_pid("before loading NIfTI file")
    print(f"Using nibabel to read NIfTI file: {file_path}")

    nii_img = nib.load(file_path)
    mask_data = nii_img.get_fdata()

    # Check whether the data is multi-channel
    if len(mask_data.shape) == 4:
        # 4D data (c, x, y, z) - multi-channel mask
        print(f"Detected multi-channel mask: {mask_data.shape}")
        # Take the union across all channels
        mask_union = np.max(mask_data, axis=0)  # (x, y, z)
    else:
        # 3D data (x, y, z)
        mask_union = mask_data

    # Ensure binary mask
    mask_union = (mask_union > 0).astype(np.float32)

    # NIfTI is usually (x, y, z), convert to (z, x, y)
    mask_union = np.transpose(mask_union, (2, 0, 1))

    # Get voxel spacing
    spacing = nii_img.header.get_zooms()
    spacing = (spacing[2], spacing[0], spacing[1])  # (x, y, z) -> (z, x, y)

    monitor_pid("after loading NIfTI file")
    print(f"NIfTI data loaded: shape={mask_union.shape}, voxel spacing={spacing}")

    return {
        'data': mask_union.astype(np.float32),
        'spacing': spacing,
    }


def resize_data(image, target_shape=(64, 128, 128)):
    """
    Resize an image using TorchIO (nearest-neighbor interpolation).

    Input: torchio.ScalarImage object
    Output: resized torchio.ScalarImage object
    """
    monitor_pid("before resize")
    # Use nearest-neighbor interpolation for masks
    resize_transform = tio.Resize(target_shape, image_interpolation='nearest')
    result = resize_transform(image)
    monitor_pid("after resize")
    return result


def resample_data(image, target_spacing=(1, 1, 1)):
    """
    Resample an image using TorchIO (nearest-neighbor interpolation).

    Input: torchio.ScalarImage object
    Output: resampled torchio.ScalarImage object
    """
    monitor_pid("before resample")
    # Use nearest-neighbor interpolation for masks
    resample_transform = tio.Resample(target_spacing, image_interpolation='nearest')
    result = resample_transform(image)
    monitor_pid("after resample")
    return result


def preprocess_volume(file_path):
    """Process a single mask volume, including channel merging and size adjustment."""
    try:
        file_path_str = str(file_path)
        print(f"Processing file: {file_path_str}")
        monitor_pid("preprocessing start")

        # Determine file type and apply corresponding preprocessing
        if file_path_str.lower().endswith('.mha'):
            # MHA file - use first-stage preprocessing
            processed_data = first_stage_preprocess(file_path)
        elif file_path_str.endswith('.nii.gz') or file_path_str.endswith('.nii'):
            # NIfTI file - use NIfTI loading
            processed_data = preprocess_nii_file(file_path_str)
        else:
            raise ValueError(f"Unsupported file format: {file_path_str}. Only .nii, .nii.gz, and .mha are supported.")

        # Get processed data
        mask_data = processed_data['data']   # Expected to be in (z, x, y)
        spacing = processed_data['spacing']  # Expected to be in (z, x, y)

        print(f"Processed data shape: {mask_data.shape} - expected format: (z, x, y)")
        print(f"Processed voxel spacing: {spacing} mm - expected format: (z, x, y)")

        # Expand dimensions to match expected input format
        mask_data = np.expand_dims(mask_data, axis=0)  # (1, z, x, y)

        # Create TorchIO object with correct spatial information
        monitor_pid("before creating TorchIO object")
        mask_tensor = torch.from_numpy(mask_data)
        mask_subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=mask_tensor,
                spacing=spacing
            )
        )
        monitor_pid("after creating TorchIO object")

        # Apply resampling and resizing
        print("Applying resampling...")
        mask_resampled_subject = resample_data(mask_subject.image)
        print(f"Shape after resampling: {mask_resampled_subject.shape}")

        print("Resizing image to 64x128x128...")
        mask_resized_subject = resize_data(mask_resampled_subject, target_shape=(64, 128, 128))
        print(f"Shape after resizing: {mask_resized_subject.shape}")

        # Get processed tensor
        mask_tensor = mask_resized_subject.data

        # Ensure output remains binary
        mask_tensor = (mask_tensor > 0.5).float()

        monitor_pid("after preprocessing")
        return mask_tensor

    except Exception as e:
        print(f"Error while preprocessing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def monitor_pid(checkpoint_name=""):
    """Monitor process ID and thread count (helper function)."""
    import psutil
    process = psutil.Process()
    thread_count = process.num_threads()
    print(f"[{checkpoint_name}] PID: {process.pid}, thread count: {thread_count}")
    return thread_count


def strip_known_suffixes(filename):
    """Remove known file suffixes (helper function).

    Note:
    The original implementation stripped .gz first, causing .nii.gz to become .nii,
    and then appending .nii.gz again produced .nii.nii.gz.
    Fix:
    Strip the compound suffix .nii.gz first, and remove known suffixes repeatedly
    when possible until no known suffix remains.
    """
    name = filename
    lowered = name.lower()
    # Loop at most twice to handle cases like .nii.gz / .mha / .nii
    for _ in range(2):
        if lowered.endswith('.nii.gz'):
            name = name[:-len('.nii.gz')]
        elif lowered.endswith('.mha'):
            name = name[:-len('.mha')]
        elif lowered.endswith('.nii'):
            name = name[:-len('.nii')]
        else:
            break
        lowered = name.lower()
    return name


def process_mask_files(input_dir, output_dir):
    """Process all mask files in a directory."""
    import os
    from pathlib import Path

    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    # Only select medical-image mask files and avoid treating arbitrary .gz files as input
    mask_files = sorted([
        p for p in input_path.iterdir()
        if p.name.lower().endswith(('.mha', '.nii', '.nii.gz'))
    ])

    mask_files = [f for f in mask_files if "valid" in f.name]
    if not mask_files:
        print(f"No mask files found in {input_dir}")
        return

    print(f"Found {len(mask_files)} mask files")

    for file_path in mask_files:
        try:
            print(f"\nProcessing: {file_path.name}")

            # Preprocess the mask file (including channel merging and resizing)
            processed_mask = preprocess_volume(file_path)

            print(f"Final mask shape: {processed_mask.shape}")
            print(f"Mask value range: [{processed_mask.min():.2f}, {processed_mask.max():.2f}]")
            print(f"Non-zero ratio: {(processed_mask > 0).sum() / processed_mask.numel():.4f}")

            # Normalize output basename to avoid cases like .nii.nii.gz
            base_name = strip_known_suffixes(file_path.name)
            output_file = Path(output_dir) / f"{base_name}.nii.gz"
            if output_file.exists():
                print(f"Target file already exists, skipping: {output_file}")
                continue

            # Convert tensor to numpy and adjust dimensions to match NIfTI format
            # Current format: (1, z, x, y) -> output format kept as processed array
            mask_numpy = processed_mask.squeeze(0).cpu().numpy()  # (z, x, y)
            mask_numpy = mask_numpy[..., ::-1, ::-1]

            # Create a NIfTI image object
            # Use identity affine since data has already been resampled
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(mask_numpy.astype(np.uint8), affine)

            # Save the NIfTI file
            nib.save(nifti_img, output_file)
            print(f"Saved to: {output_file}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    print("\nAll files have been processed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segmentation mask preprocessing")
    parser.add_argument("--input", type=str, default="/path/to/%%%/segmentations", help="Input mask directory")
    parser.add_argument("--output", type=str, default="/path/to/%%%/lesion_mask", help="Output directory")

    args = parser.parse_args()
    process_mask_files(args.input, args.output)