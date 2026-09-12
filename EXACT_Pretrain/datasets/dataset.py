from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import h5py
import torch
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

import nibabel as nib
from torch.utils.data import Dataset
import os
import torch

import h5py
from torch.utils.data import Dataset

class my_datasets(Dataset):  
    def __init__(self, h5_path, train=False, val=False, test=False, seed=42):   
        """Dataset over the preprocessed HDF5 store.

        :param h5_path: path to the HDF5 file
        :param train: use the training split
        :param val: use the validation split
        :param test: use every sample
        :param seed: seed for the split, so it stays the same across runs
        """
        super(my_datasets, self).__init__()
        self.h5_path = h5_path  

        # Organ channels present in the store
        self.organ_mapping = {  
            "lung": 0,  
            "trachea and bronchie": 1,  
            "pleura": 2,  
            "mediastinum": 3,  
            "heart": 4,  
            "esophagus": 5,  
            "bone": 6,  
            "thyroid": 7,  
            "abdomen": 8  
        }  
        
        # The 18 disease labels, in channel order
        self.disease_names = [ 
            "Medical material","Arterial wall calcification", 
            "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
            "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
            "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
            "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
            "Bronchiectasis", "Interlobular septal thickening"  
        ]  
        
        # Organs kept for this task
        self.required_organs = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", "esophagus"]  
        
        # Their channel indices
        self.required_indices = [self.organ_mapping[organ] for organ in self.required_organs]  

        # The store has to exist
        if not os.path.exists(self.h5_path):  
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        # Record every sample name
        with h5py.File(self.h5_path, 'r') as f:  
            self.data_keys = list(f.keys())  

        # Shuffle with a fixed seed so the split is reproducible
        torch.manual_seed(seed)  
        indices = torch.randperm(len(self.data_keys)).tolist()  

        # Pick the requested subset
        if train or val:  
            # 15/16 train, 1/16 val
            total_size = len(self.data_keys)
            train_size = int(total_size * 15 / 16)
            val_size = total_size - train_size  

            self.train_indices = indices[:train_size]  
            self.val_indices = indices[train_size:]  

            if train:  
                self.subset_indices = self.train_indices  
            else:  # val  
                self.subset_indices = self.val_indices  
        
        elif test:  
            # Test uses every sample
            self.subset_indices = indices  
        
        else:  
            raise ValueError("exactly one of train, val or test must be True")

    def __getitem__(self, index):  
        """Fetch one sample.

        :param index: index into the current subset
        :return: CT volume, mask (with the extra global channel), 18 disease
            labels, and the sample key
        """
        # Key of the requested sample
        data_index = self.subset_indices[index]  
        sample_key = self.data_keys[data_index]  

        # Read it out of the store
        with h5py.File(self.h5_path, 'r') as f:  
            ct_img = f[sample_key]['ct'][:]  
            mask_np = f[sample_key]['mask'][:]  # [9, D, H, W]
            label_18 = f[sample_key]['label_18'][:]  # 18 disease labels

        # The global mask is the union of every organ channel: cast to float,
        # treat anything > 0 as foreground, then reduce over the channel axis.
        all_organs_mask = torch.tensor(mask_np, dtype=torch.float32)         # [9, D, H, W]
        global_mask = (all_organs_mask > 0).any(dim=0, keepdim=True).float() # [1, D, H, W]

        # Keep only the organ channels this task uses
        selected_mask = all_organs_mask[self.required_indices]               # [6, D, H, W]

        # Append the global channel, giving [7, D, H, W]
        mask = torch.cat([selected_mask, global_mask], dim=0)

        # The rest becomes tensors as-is
        ct_img = torch.tensor(ct_img, dtype=torch.float32)  
        label_18 = torch.tensor(label_18, dtype=torch.float)  

        return ct_img, mask, label_18, sample_key

    def __len__(self):  
        """Number of samples in the current subset."""
        return len(self.subset_indices)  

    @property  
    def organ_names(self):  
        """Organ names kept for this task."""
        return self.required_organs  

    @property  
    def disease_list(self):  
        """Disease names, in label order."""
        return self.disease_names

