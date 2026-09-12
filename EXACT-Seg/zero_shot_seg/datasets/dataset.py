"""Dataset for zero-shot anomaly localization.

Reads the same HDF5 store as the other stages: one group per study holding
``ct`` and a ``label_18`` (or ``label_16``) vector. No segmentation masks are
needed here -- the AAmaps this dataset feeds are produced without any
voxel-level supervision.
"""
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class My_datasets(Dataset):
    def __init__(self, h5_path, train=False, val=False, test=False, seed=42):
        super(My_datasets, self).__init__()
        self.h5_path = h5_path

        # The 18 findings, in the channel order the backbone was trained with.
        self.disease_names = [
            'Medical material','Arterial wall calcification', 'Cardiomegaly', 
            'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia',
            'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity',
            'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
            'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
            'Interlobular septal thickening'
        ]

        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"HDF5 store not found: {self.h5_path}")
        print("H5 path:", self.h5_path)
        with h5py.File(self.h5_path, 'r') as f:
            print("Groups in store:", len(f.keys()))
            all_keys = [k.strip() for k in f.keys()]  # drop stray whitespace
        self.data_keys = sorted([k for k in all_keys])

        if "covid_ct" in self.data_keys:
            self.data_keys = [k for k in self.data_keys
                              if k.startswith('coronacases')]

        if len(self.data_keys) == 0:
            raise RuntimeError(f"No usable studies in {self.h5_path}.")

        # Shuffle once under a fixed seed so the split is reproducible.
        torch.manual_seed(seed)
        indices = torch.randperm(len(self.data_keys)).tolist()

        if train or val:
            total_size = len(self.data_keys)
            train_size = int(total_size * 15 / 16)
            self.train_indices = indices[:train_size]
            self.val_indices = indices[train_size:]
            self.subset_indices = self.train_indices if train else self.val_indices
        elif test:
            self.subset_indices = indices
        else:
            raise ValueError("Set exactly one of train, val or test to True")

        print("==========================================")
        print("Studies:", len(self.subset_indices))
        print("==========================================")

    def __getitem__(self, index):
        data_index = self.subset_indices[index]
        sample_key = self.data_keys[data_index]

        with h5py.File(self.h5_path, 'r') as f:
            ct_img = f[sample_key]['ct'][:]
            if 'label_18' not in f[sample_key] and 'label_16' in f[sample_key]:
                # Cohorts annotating 16 findings omit channels 4 and 13
                # (coronary artery wall calcification, mosaic attenuation);
                # zero-fill them so the channel order still matches the model.
                label_16 = f[sample_key]['label_16'][:]
                label_18 = np.zeros((18,), dtype=label_16.dtype)
                label_18[0:4] = label_16[0:4]
                label_18[4] = 0
                label_18[5:13] = label_16[4:12]
                label_18[13] = 0
                label_18[14:] = label_16[12:]
            elif 'label_18' in f[sample_key]:
                label_18 = f[sample_key]['label_18'][:]
            else:
                label_18 = np.zeros((18,), dtype=np.float32)

        ct_img = torch.tensor(ct_img, dtype=torch.float32)
        assert len(ct_img.shape) in [3, 4], f"ct_img shape error: {ct_img.shape}"
        if len(ct_img.shape) == 3:
            ct_img = ct_img.unsqueeze(0)
        label_18 = torch.tensor(label_18, dtype=torch.float)

        return ct_img, label_18, sample_key

    def __len__(self):
        return len(self.subset_indices)

    @property
    def disease_list(self):
        return self.disease_names
