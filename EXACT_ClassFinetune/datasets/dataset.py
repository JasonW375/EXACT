from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
import json
import nibabel as nib
import time

# Legacy dataset implementation removed for readability.
    
from torch.utils.data import Dataset
import os
import torch
import json
import time

import h5py
from torch.utils.data import Dataset
    
class HeatmapDataset(Dataset):
    def __init__(self, root_dir, h5_path, train=False, val=False, test=False,
                 file_suffix="high-res_combined_pred.nii.gz", raise_if_missing=False):
        """
    root_dir: heatmap root directory; each subfolder is one sample (e.g. train_xxx)
    h5_path: HDF5 file path containing label_18
    Each item is a stacked multi-channel tensor of 18 disease heatmaps: [18, D, H, W]
    Returns: (heatmaps_18ch, labels_18, sample_name)
        """
        super().__init__()
        self.root_dir = root_dir
        self.h5_path = h5_path
        self.file_suffix = file_suffix
        self.raise_if_missing = raise_if_missing

        # 18-class order aligned with label_18 in HDF5 (also the channel order)
        self.disease_names = [
            "Medical material", "Arterial wall calcification",
            "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",
            "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",
            "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
            "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
            "Bronchiectasis", "Interlobular septal thickening"
        ]
        self.num_diseases = len(self.disease_names)

        # Collect sample directories
        self.samples = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No sample directory found under: {root_dir}")

        # Split
        torch.manual_seed(42)
        indices = torch.randperm(len(self.samples)).tolist()
        if train or val:
            total_size = len(self.samples)
            train_size = int(total_size * 15 / 16)
            self.train_indices = indices[:train_size]
            self.val_indices = indices[train_size:]
            self.subset_indices = self.train_indices if train else self.val_indices
        elif test:
            self.subset_indices = indices
        else:
            raise ValueError("One of train, val, or test must be True")

        # Warn once for each missing item
        self._warned_missing = set()

    def __len__(self):
        # Count by sample (each sample returns 18 channels)
        return len(self.subset_indices)

    def _first_available_shape(self, sample_dir: str):
        # To create zero-filled fallback channels, infer spatial shape first
        # print("sample_dir:", sample_dir)
        for fn in os.listdir(sample_dir):
            # print("fn:", fn)
            
            if fn.endswith(self.file_suffix):
                path = os.path.join(sample_dir, fn)
                # print("path:", path)
                try:
                    data = nib.load(path).get_fdata()
                    return data.shape
                except Exception:
                    continue
        return None

    def __getitem__(self, idx):
        base_idx = self.subset_indices[idx]
        sample_name = self.samples[base_idx]
        sample_dir = os.path.join(self.root_dir, sample_name)

        spatial_shape = self._first_available_shape(sample_dir)
        if spatial_shape is None:
            raise FileNotFoundError(f"No readable heatmap found in sample directory to infer shape: {sample_dir}")

        # Preallocate [18, D, H, W]
        heatmaps = np.zeros((self.num_diseases,) + spatial_shape, dtype=np.float32)

        # Load each disease channel
        for c, abn_name in enumerate(self.disease_names):
            # Match heatmap file for current disease
            matched = [fn for fn in os.listdir(sample_dir)
                       if (abn_name in fn) and fn.endswith(self.file_suffix)]
            if len(matched) > 0:
                hpath = os.path.join(sample_dir, matched[0])
                try:
                    data = nib.load(hpath).get_fdata().astype(np.float32)
                    # If shape mismatches, resampling can be added here; currently assumes consistent shapes
                    if data.shape != spatial_shape:
                        msg = f"[HeatmapDataset] Shape mismatch: {sample_name} | {abn_name} got {data.shape}, expect {spatial_shape}"
                        if self.raise_if_missing:
                            raise RuntimeError(msg)
                        if (sample_name, abn_name) not in self._warned_missing:
                            print(msg + " -> replacing with a zero channel.")
                            self._warned_missing.add((sample_name, abn_name))
                        # Keep this channel as zero
                    else:
                        heatmaps[c] = data
                except Exception as e:
                    msg = f"[HeatmapDataset] Read failed: {hpath}, err={e}"
                    if self.raise_if_missing:
                        raise
                    if (sample_name, abn_name) not in self._warned_missing:
                        print(msg + " -> replacing with a zero channel.")
                        self._warned_missing.add((sample_name, abn_name))
            else:
                # Missing heatmap for this disease -> zero channel
                if (sample_name, abn_name) not in self._warned_missing:
                    print(f"[HeatmapDataset] Missing heatmap: {sample_name} | {abn_name} -> replacing with a zero channel.")
                    self._warned_missing.add((sample_name, abn_name))

        heatmaps = torch.from_numpy(heatmaps)  # [18, D, H, W], float32

        # Load 18-d label
        with h5py.File(self.h5_path, 'r') as f:
            if sample_name not in f:
                raise KeyError(f"Sample {sample_name} not found in HDF5: {self.h5_path}")
            if "label_18" not in f[sample_name].keys():
                label_16= f[sample_name]['label_16'][:]  # 16-d disease label
                label_16=torch.tensor(label_16,dtype=torch.float32)
                label_18 = torch.zeros(18, dtype=torch.float32)

                # Fill label_18 from label_16 using index mapping
                label_18[0:4] = label_16[0:4]
                label_18[5:13] = label_16[4:12]
                label_18[14:18] = label_16[12:16] 
            else:
                label_18 = f[sample_name]['label_18'][:]  # 18-d disease label  
            # label_18 = f[sample_name]['label_18'][:]  # (18,)
        labels = torch.tensor(label_18, dtype=torch.float32)  # [18]

        

        # Return: 18-channel heatmaps, 18-d label, sample name
        return heatmaps, labels, sample_name