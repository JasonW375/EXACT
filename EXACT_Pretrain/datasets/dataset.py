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
class my_datasets(Dataset):  
    def __init__(self, h5_path, train=False, val=False, test=False, seed=42):   
        """  
        初始化数据集  
        :param h5_path: HDF5 文件路径  
        :param train: 是否为训练集  
        :param val: 是否为验证集  
        :param test: 是否为测试集  
        :param seed: 随机种子，用于划分一致性  
        """  
        super(NW_datasets, self).__init__()  
        self.h5_path = h5_path  

        # 定义需要的器官索引  
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
        
        # 定义18种疾病名称  
        self.disease_names = [ 
            "Medical material","Arterial wall calcification", 
            "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
            "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
            "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
            "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
            "Bronchiectasis", "Interlobular septal thickening"  
        ]  
        
        # 需要保留的器官列表  
        self.required_organs = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", "esophagus"]  
        
        # 获取需要保留的器官对应的索引  
        self.required_indices = [self.organ_mapping[organ] for organ in self.required_organs]  

        # 验证 HDF5 文件是否存在  
        if not os.path.exists(self.h5_path):  
            raise FileNotFoundError(f"HDF5 文件未找到: {self.h5_path}")  

        # 打开 HDF5 文件，记录所有样本的名称  
        with h5py.File(self.h5_path, 'r') as f:  
            self.data_keys = list(f.keys())  

        # 设置随机种子并随机打乱索引  
        torch.manual_seed(seed)  
        indices = torch.randperm(len(self.data_keys)).tolist()  

        # 根据参数选择当前子集  
        if train or val:  
            # 训练和验证时按照 11:1 划分  
            total_size = len(self.data_keys)  
            train_size = int(total_size * 15 / 16)  # 15/16（原注释保留，未改动）  
            val_size = total_size - train_size  

            self.train_indices = indices[:train_size]  
            self.val_indices = indices[train_size:]  

            if train:  
                self.subset_indices = self.train_indices  
            else:  # val  
                self.subset_indices = self.val_indices  
        
        elif test:  
            # 测试时使用全部数据  
            self.subset_indices = indices  
        
        else:  
            raise ValueError("必须指定 train, val 或 test 中的一个为 True")  

    # 其余方法保持不变，__getitem__、__len__、organ_names、disease_list 方法不需要修改

    def __getitem__(self, index):  
        """  
        根据索引获取数据项  
        :param index: 数据索引  
        :return: CT 图像、掩码(增加global通道)、16种疾病标签和样本键  
        """  
        # 获取样本的键  
        data_index = self.subset_indices[index]  
        sample_key = self.data_keys[data_index]  

        # 从 HDF5 文件加载数据  
        with h5py.File(self.h5_path, 'r') as f:  
            ct_img = f[sample_key]['ct'][:]  
            mask_np = f[sample_key]['mask'][:]  # 原始掩码 [9, D, H, W]  
            label_18 = f[sample_key]['label_18'][:]  # 18种疾病的标签  

        # ===== 修改处开始：先对“全部器官通道”的并集生成 global_mask =====
        # 将全部器官通道转为 tensor（float32），>0 判定为前景，然后对通道维做 any
        all_organs_mask = torch.tensor(mask_np, dtype=torch.float32)         # [9, D, H, W]
        global_mask = (all_organs_mask > 0).any(dim=0, keepdim=True).float() # [1, D, H, W]

        # 只选择需要的器官通道
        selected_mask = all_organs_mask[self.required_indices]               # [6, D, H, W]

        # 将全局掩码添加到原始掩码后面（得到 [7, D, H, W]）
        mask = torch.cat([selected_mask, global_mask], dim=0)
        # ===== 修改处结束 =====

        # 转换其他数据为 PyTorch 张量  
        ct_img = torch.tensor(ct_img, dtype=torch.float32)  
        label_18 = torch.tensor(label_18, dtype=torch.float)  

        return ct_img, mask, label_18, sample_key

    def __len__(self):  
        """  
        返回数据集的大小  
        """  
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

