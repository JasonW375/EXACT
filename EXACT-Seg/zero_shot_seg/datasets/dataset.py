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

    
from torch.utils.data import Dataset
import os
import torch
import json
import time

import h5py
from torch.utils.data import Dataset

    # Cardiomegaly -> 心脏肥大
    # Pericardial effusion -> 心包积液
    # Coronary artery wall calcification -> 冠状动脉壁钙化
    # Hiatal hernia -> 食管裂孔疝
    # Lymphadenopathy -> 淋巴结病变/淋巴结肿大
    # Emphysema -> 肺气肿
    # Atelectasis -> 肺不张
    # Lung nodule -> 肺结节
    # Lung opacity -> 肺部片状影/肺不透明
    # Pulmonary fibrotic sequela -> 肺纤维化后遗症/肺纤维化后遗表现
    # Pleural effusion -> 胸腔积液
    # Mosaic attenuation pattern -> 马赛克样衰减/马赛克样减低密度
    # Peribronchial thickening -> 支气管周围增厚
    # Consolidation -> 肺实变
    # Bronchiectasis -> 支气管扩张
    # Interlobular septal thickening -> 小叶间隔增厚

class My_datasets(Dataset):  
    def __init__(self, h5_path, train=False, val=False, test=False, seed=42,
                 segmentations_dir="/path/to/bxg/storage/ReXGroundingCT/lesion_mask"):   
        # ...existing code...
        # 读取 HDF5 的所有键
        super(My_datasets, self).__init__()  
        self.h5_path = h5_path  
        
        # 定义18种疾病名称  
        self.disease_names = [  
            'Medical material','Arterial wall calcification', 'Cardiomegaly', 
            'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia',
            'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity',
            'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
            'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
            'Interlobular septal thickening'
        ]  

        # 验证 HDF5 文件是否存在  
        if not os.path.exists(self.h5_path):  
            raise FileNotFoundError(f"HDF5 文件未找到: {self.h5_path}")  
        print("H5 path:", self.h5_path)
        with h5py.File(self.h5_path, 'r') as f:  
            print("HDF5 键总数:", len(f.keys()))
            all_keys = [k.strip() for k in f.keys()]  # 去掉意外空白字符
        self.data_keys = sorted([k for k in all_keys])
       
        if "covid_ct" in self.data_keys:
            self.data_keys=[k for k in self.data_keys if k.startswith('coronacases')]
        def _strip_nii_suffix(name: str) -> str:
            # 统一去掉 .nii.gz 或 .nii
            base = os.path.basename(name)
            if base.endswith(".nii.gz"):
                return base[:-7].strip()
            if base.endswith(".nii"):
                return base[:-4].strip()
            return os.path.splitext(base)[0].strip()

        def _iter_entries(obj):
            # 兼容 {split: [..]} 或直接 [..] 的结构
            if isinstance(obj, list):
                for it in obj:
                    if isinstance(it, dict):
                        yield it
            elif isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict):
                                yield it

        pixel_sums = {}
        

      
        if len(self.data_keys) == 0:
            raise RuntimeError("过滤后无可用样本，请检查 HDF5 键、分割文件名与 dataset.json 的 name 是否一致，以及像素阈值条件。")

        # 设置随机种子并随机打乱索引（在过滤之后）
        torch.manual_seed(seed)  
        indices = torch.randperm(len(self.data_keys)).tolist()  

        # 根据参数选择当前子集  
        if train or val:  
            total_size = len(self.data_keys)  
            train_size = int(total_size * 15 / 16)  
            self.train_indices = indices[:train_size]  
            self.val_indices = indices[train_size:]  
            self.subset_indices = self.train_indices if train else self.val_indices  
        elif test:  
            self.subset_indices = indices
    
        else:  
            raise ValueError("必须指定 train, val 或 test 中的一个为 True")  

        print("==========================================")
        print("样本数:", len(self.subset_indices))
        print("==========================================")
        # assert False
    def __getitem__(self, index):  
        data_index = self.subset_indices[index]  
        sample_key = self.data_keys[data_index]  

        with h5py.File(self.h5_path, 'r') as f:  
            ct_img = f[sample_key]['ct'][:]  
            if 'label_18' not in f[sample_key] and 'label_16' in f[sample_key]:
                # print(f"[My_datasets] 警告: 样本 {sample_key} 缺少 label_18，使用 label_16 填充")
                label_16=f[sample_key]['label_16'][:]
                label_18=np.zeros((18,),dtype=label_16.dtype)
                label_18[0:4]=label_16[0:4]
                label_18[4]=0
                label_18[5:13]=label_16[4:12]
                label_18[13]=0
                label_18[14:]=label_16[12:]
            elif 'label_18' in f[sample_key]:
                label_18 = f[sample_key]['label_18'][:]  
            else:
                label_18=np.zeros((18,),dtype=np.float32)
            
            

        ct_img = torch.tensor(ct_img, dtype=torch.float32)  
        # print("ct_img shape",ct_img.shape)
        assert len(ct_img.shape) in [3,4], f"ct_img shape error: {ct_img.shape}"
        if len(ct_img.shape)==3:
            ct_img=ct_img.unsqueeze(0)
        label_18 = torch.tensor(label_18, dtype=torch.float)  


        return ct_img, label_18, sample_key  

    def __len__(self):  
        return len(self.subset_indices)

    @property  
    def organ_names(self):  
        """  
        返回当前使用的器官名称列表  
        """  
        return self.required_organs  

    @property  
    def disease_list(self):  
        """  
        返回疾病名称列表  
        """  
        return self.disease_names




def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
        
    