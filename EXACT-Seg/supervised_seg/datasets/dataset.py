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

# class NPY_datasets(Dataset):
#     def __init__(self, path_Data, config, train=True):
#         super(NPY_datasets, self).__init__()
#         # 定义九个区域掩码的文件名列表
#         region_names = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", 
#                         "esophagus", "bone", "thyroid", "abdomen"]

#         if train:
#             #使用 os.walk() 来递归遍历所有子文件夹，找到图像文件
#             errors = []

#             images_list = []
#             train_preprocessed_path = os.path.join(path_Data, 'train', 'train_preprocessed')
#             for root, dirs, files in os.walk(train_preprocessed_path):
#                 for file in files:
#                     if file.endswith('.nii.gz'):  # 只匹配实际的图像文件
#                         images_list.append(os.path.join(root, file))  # 完整路径加入列表

#             self.data = []
#             for img_path in images_list:
#                 try:
#                     # 获取图像文件的基础文件名
#                     base_name = os.path.basename(img_path).replace('.nii.gz', '')  # 去掉扩展名
                    
#                     # 在文件名前加上 'seg_'
#                     base_name = 'seg_' + base_name  # 只是在文件名开头加上 'seg_'
#                     mask_folder_path = os.path.join(path_Data + '/train/train_region_mask', base_name)

#                     # 检查 mask_folder_path 是否存在，并确保它是一个文件夹
#                     if os.path.isdir(mask_folder_path):
#                         # 初始化一个空的掩膜文件列表
#                         mask_files = []
#                         for region in region_names:
#                             mask_file = os.path.join(mask_folder_path, f"{region}.nii.gz")
#                             if os.path.exists(mask_file):
#                                 mask_files.append(mask_file)
#                             else:
#                                 # 如果掩膜文件不存在，记录错误并继续
#                                 error_message = f"掩膜文件 {mask_file} 不存在！"
#                                 errors.append(error_message)
#                                 continue  # 继续处理下一个 region

#                         # 将图像路径与掩膜文件夹中的所有掩膜文件配对
#                         if mask_files:
#                             self.data.append([img_path, mask_files])  # 保存图像和所有掩膜文件路径
#                     else:
#                         # 如果掩膜文件夹不存在，记录错误并继续
#                         error_message = f"掩膜文件夹 {mask_folder_path} 不存在！"
#                         errors.append(error_message)
#                 except Exception as e:
#                     # 捕获所有其他异常并记录
#                     errors.append(f"处理 {img_path} 时发生错误: {str(e)}")

#             # 在处理完所有文件后，打印所有收集到的错误信息
#             if errors:
#                 print("以下是处理过程中遇到的错误：")
#                 for error in errors:
#                     print(error)
#                 print(f"处理过程中共遇到 {len(errors)} 个错误：")

#             self.transformer = config.train_transformer
#         else:
#             # 使用 os.walk() 来递归遍历所有子文件夹，找到图像文件
#             errors = []

#             images_list = []
#             valid_preprocessed_path = os.path.join(path_Data, 'valid', 'valid_preprocessed')
#             for root, dirs, files in os.walk(valid_preprocessed_path):
#                 for file in files:
#                     if file.endswith('.nii.gz'):  # 只匹配实际的图像文件
#                         images_list.append(os.path.join(root, file))  # 完整路径加入列表

#             self.data = []
#             for img_path in images_list:
#                 try:
#                     # 获取图像文件的基础文件名
#                     base_name = os.path.basename(img_path).replace('.nii.gz', '')  # 去掉扩展名
                    
#                     # 在文件名前加上 'seg_'
#                     base_name = 'seg_' + base_name  # 只是在文件名开头加上 'seg_'
#                     mask_folder_path = os.path.join(path_Data + '/valid/valid_region_mask', base_name)

#                     # 检查 mask_folder_path 是否存在，并确保它是一个文件夹
#                     if os.path.isdir(mask_folder_path):
#                         # 初始化一个空的掩膜文件列表
#                         mask_files = []
#                         for region in region_names:
#                             mask_file = os.path.join(mask_folder_path, f"{region}.nii.gz")
#                             if os.path.exists(mask_file):
#                                 mask_files.append(mask_file)
#                             else:
#                                 # 如果掩膜文件不存在，记录错误并继续
#                                 error_message = f"掩膜文件 {mask_file} 不存在！"
#                                 errors.append(error_message)
#                                 continue  # 继续处理下一个 region

#                         # 将图像路径与掩膜文件夹中的所有掩膜文件配对
#                         if mask_files:
#                             self.data.append([img_path, mask_files])  # 保存图像和所有掩膜文件路径
#                     else:
#                         # 如果掩膜文件夹不存在，记录错误并继续
#                         error_message = f"掩膜文件夹 {mask_folder_path} 不存在！"
#                         errors.append(error_message)
#                 except Exception as e:
#                     # 捕获所有其他异常并记录
#                     errors.append(f"处理 {img_path} 时发生错误: {str(e)}")

#             # 在处理完所有文件后，打印所有收集到的错误信息
#             if errors:
#                 print("以下是处理过程中遇到的错误：")
#                 for error in errors:
#                     print(error)
#                 print(f"处理过程中共遇到 {len(errors)} 个错误：")

#             self.transformer = config.test_transformer
        
#     def __getitem__(self, indx):
#         img_path, msk_paths = self.data[indx]

#         # 使用 nibabel 加载图像文件
#         img = nib.load(img_path).get_fdata()  # 加载图像数据
#         img = np.array(img)  # 转换为 numpy 数组

#         # 加载掩膜文件（多个掩膜）
#         masks = []
#         for msk_path in msk_paths:
#             msk = nib.load(msk_path).get_fdata()  # 加载掩膜数据
#             msk = np.array(msk)
#             masks.append(msk)

#         # 合并所有掩膜（假设需要堆叠到一个多通道掩膜中）
#         msk = np.stack(masks, axis=-1)  # 将掩膜堆叠到最后一个维度，形状为 (H, W, D, C)

#         # 调整图像的形状，增加通道维度，使其变为 (1, D, H, W)
#         img = np.expand_dims(img, axis=0)  # 形状变为 (1, H, W, D)
#         img = np.transpose(img, (0, 3, 1, 2))  # 重新排列维度，使其形状变为 (1, D, H, W)

#         # 调整掩码的形状，将其从 (H, W, D, C) 直接变为 (C, D, H, W)
#         msk = np.transpose(msk, (3, 2, 0, 1))

#         # print(f"Transformed image type: {type(img)}, shape: {img.shape}")
#         # print(f"Transformed mask type: {type(msk)}, shape: {msk.shape}")

#         # 使用转换器对图像和掩膜进行转换
#         img, msk = self.transformer((img, msk))
#         # 调试信息：检查转换后的数据类型和形状
#         # print(f"Transformed image type: {type(img)}, shape: {img.shape}")
#         # print(f"Transformed mask type: {type(msk)}, shape: {msk.shape}")


#         return img, msk


#     def __len__(self):
#         return len(self.data)
    
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

class NW_datasets_bad(Dataset):  
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
            train_size = int(total_size * 15 / 16)#15/16  
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
            mask = f[sample_key]['mask'][:]  # 原始掩码 [9, D, H, W]  
            label_18 = f[sample_key]['label_18'][:]  # 18种疾病的标签  

        # 只选择需要的器官通道  
        mask = mask[self.required_indices]  # 现在变成 [6, D, H, W]  

        # 创建全局掩码（所有器官的并集）  
        global_mask = torch.zeros_like(torch.tensor(mask[0:1]))  # 创建一个和单个通道相同形状的零张量  
        for i in range(len(self.required_indices)):  
            global_mask = global_mask | (mask[i:i+1] > 0)  # 使用位运算计算并集  
        
        # 将全局掩码添加到原始掩码后面  
        mask = torch.tensor(mask, dtype=torch.float)  # 先转换原始掩码为tensor  
        mask = torch.cat([mask, global_mask], dim=0)  # 拼接全局掩码，现在变成 [7, D, H, W]  

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

class NW_datasets(Dataset):  
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

class NW_datasets_5fold(Dataset):  
    def __init__(self, h5_path, train=False, val=False, test=False, full_dataset=False, seed=42):   
        """  
        初始化数据集  
        :param h5_path: HDF5 文件路径  
        :param train: 是否为训练集（传统模式）  
        :param val: 是否为验证集（传统模式）  
        :param test: 是否为测试集  
        :param full_dataset: 是否返回完整数据集（用于交叉验证）
        :param seed: 随机种子，用于划分一致性  
        """  
        super(NW_datasets_5fold, self).__init__()  
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

        if full_dataset:
            # 交叉验证模式：返回完整数据集
            self.subset_indices = indices
        elif train or val:  
            # 传统模式：训练和验证时按照 15:1 划分  
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
            # 测试时使用全部数据  
            self.subset_indices = indices  
        else:  
            raise ValueError("必须指定 train, val, test 或 full_dataset 中的一个为 True")  

    def __getitem__(self, index):  
        """  
        根据索引获取数据项  
        :param index: 数据索引  
        :return: CT 图像、掩码(增加global通道)、18种疾病标签和样本键  
        """  
        # 获取样本的键  
        data_index = self.subset_indices[index]  
        sample_key = self.data_keys[data_index]  

        # 从 HDF5 文件加载数据  
        with h5py.File(self.h5_path, 'r') as f:  
            ct_img = f[sample_key]['ct'][:]  
            mask_np = f[sample_key]['mask'][:]  # 原始掩码 [9, D, H, W]  
            label_18 = f[sample_key]['label_18'][:]  # 18种疾病的标签  

        # ===== 修改处开始：先对"全部器官通道"的并集生成 global_mask =====
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

    def get_sample_id(self, index):
        """
        根据索引获取样本ID
        """
        data_index = self.subset_indices[index]
        return self.data_keys[data_index]

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

# class NW_datasets_supervised(Dataset):  
#     def __init__(self, h5_path,train=False, val=False, test=False,
#         mask_dir="/path/to/bxg/storage/ReXGroundingCT/lesion_mask", seed=42):   
#         """  
#         初始化数据集  
#         :param h5_path: HDF5 文件路径  
#         :param train: 是否为训练集  
#         :param val: 是否为验证集  
#         :param test: 是否为测试集  
#         :param seed: 随机种子，用于划分一致性  
#         """  
#         super(NW_datasets_supervised, self).__init__()  
#         self.h5_path = h5_path  
#         self.mask_dir=mask_dir
#         # 定义需要的器官索引  

        


#         # 验证 HDF5 文件是否存在  
#         if not os.path.exists(self.h5_path):  
#             raise FileNotFoundError(f"HDF5 文件未找到: {self.h5_path}")  

#         # 打开 HDF5 文件，记录所有样本的名称  
#         with h5py.File(self.h5_path, 'r') as f:  
#             self.data_keys = list(f.keys())  
#         # self.data_keys=[k for k in self.data_keys if k.startswith('coronacases')]
#         if "total_processed" in self.h5_path.lower():
#             assert os.path.exists(self.mask_dir) and os.path.isdir(self.mask_dir),f"mask目录不存在: {self.mask_dir}"
#             # 只计算有mask的样本key
            
#             seg_files = [fn for fn in os.listdir(mask_dir) if fn.endswith(".nii.gz")]
#             seg_keys = {
#                 os.path.splitext(os.path.splitext(fn)[0])[0].strip()  # 去掉 .nii.gz 并 strip
#                 for fn in seg_files
#             }
#             json_path = "/path/to/bxg/storage/ReXGroundingCT/dataset.json"
#             assert os.path.exists(json_path),f"json文件不存在: {json_path}"

#             self.allowed_keys=seg_keys&set(self.data_keys)
#             self.data_keys = list(self.allowed_keys)
#             assert len(self.data_keys)>0,f"没有找到有mask的样本"
#         if "covid_ct" in self.h5_path.lower():
#             self.data_keys=[k for k in self.data_keys if k.startswith('coronacases')]
#         self.data_keys.sort()
#         # 设置随机种子并随机打乱索引  
#         torch.manual_seed(seed)  


#         if "covid_ct" in self.h5_path.lower():
#             coronas = [k for k in self.data_keys if k.startswith("coronacases")]
#             radios   = [k for k in self.data_keys if k.startswith("radio")]

#             # 保证排序与可重复性
#             coronas.sort()
#             radios.sort()

#             # 用户需求：仅使用指定两个 coronacases 进行训练，其他全部用于测试
#             fixed_train_coronas = ["coronacases_003", "coronacases_007"]
#             # 过滤出存在于数据集的指定键
#             train_coronas = [k for k in fixed_train_coronas if k in coronas]
#             if len(train_coronas) == 0:
#                 raise ValueError("指定的训练样本未在数据集中找到：coronacases_003, coronacases_007")

#             # 训练集只包含指定 coronacases，不混入 radio
#             train_keys = train_coronas

#             # 测试样本：除训练指定外的全部样本（包括剩余 coronacases 与所有 radio）
#             test_coronas = [k for k in coronas if k not in train_coronas]
#             test_radios  = radios  # radios 全部进测试
#             test_keys = test_coronas + test_radios

#             if train:
#                 self.subset_indices = [self.data_keys.index(k) for k in train_keys]
#             elif val or test:
#                 # val 与 test 都使用剩余样本
#                 self.subset_indices = [self.data_keys.index(k) for k in test_keys]
#             else:
#                 raise ValueError("必须指定 train, val 或 test 中的一个为 True")

#         else:
#             # ===== 原有通用划分逻辑 =====
#             indices = torch.randperm(len(self.data_keys)).tolist()
#             if train or val:
#                 if "mosmed" in self.h5_path.lower():
#                     train_size = 10
#                 elif "tbad" in self.h5_path.lower():
#                     train_size = 62
#                 else:
#                     total_size = len(self.data_keys)
#                     train_size = int(total_size * 15 / 16)
#                 val_size = len(self.data_keys) - train_size

#                 self.train_indices = indices[:train_size]
#                 self.val_indices = indices[train_size:train_size + val_size]

#                 self.subset_indices = self.train_indices if train else self.val_indices
#             elif test:
#                 self.subset_indices = indices
#             else:
#                 raise ValueError("必须指定 train, val 或 test 中的一个为 True") 
#         print("============================================")
#         print(f"NW_datasets_supervised initialized with {len(self.subset_indices)} samples.")
#         print("============================================")
#     # 其余方法保持不变，__getitem__、__len__、organ_names、disease_list 方法不需要修改

#     def __getitem__(self, index):  
#         """  
#         根据索引获取数据项  
#         :param index: 数据索引  
#         :return: CT 图像、掩码(增加global通道)、16种疾病标签和样本键  
#         """  
#         # 获取样本的键  
#         data_index = self.subset_indices[index]  
#         sample_key = self.data_keys[data_index]  

#         # 从 HDF5 文件加载数据  
#         with h5py.File(self.h5_path, 'r') as f:  
#             ct_img = f[sample_key]['ct'][:]  
#             # mask_np = f[sample_key]['mask'][:]  
#             # print("ct_img shape:", ct_img.shape)
#             # print("mask_np shape:", mask_np.shape)
#             # 从mask_dir中读取nii.gz文件作为掩码
#             if "total_processed" in self.h5_path:
#                 mask_np=nib.load(os.path.join(self.mask_dir, sample_key+'.nii.gz')).get_fdata()
#             elif "tbad" in self.h5_path.lower():
#                 mask_np=f[sample_key]['mask'][0:2,...] # 只取前两个通道作为掩码 
#                 ct_img=ct_img[:,:,24:88,22:102]
#                 mask_np=mask_np[:,:,24:88,22:102]
#             else:
#                 mask_np = f[sample_key]['mask'][:]  
#             # print("ct_img shape:", ct_img.shape)
#             # print("mask_np shape:", mask_np.shape)


#         mask = torch.tensor(mask_np, dtype=torch.float32)         # [D, H, W]
#         if len(mask.shape)==3:
#             mask=mask.unsqueeze(0)  # [1, D, H, W]


#         # 转换其他数据为 PyTorch 张量  
#         ct_img = torch.tensor(ct_img, dtype=torch.float32)  
#         if len(ct_img.shape)==3:
#             ct_img=ct_img.unsqueeze(0)  # [1, D, H, W]
#         # print("ct_img shape:", ct_img.shape)
#         # print("mask_np shape:", mask_np.shape)
#         return ct_img, mask, sample_key

#     def __len__(self):  
#         """  
#         返回数据集的大小  
#         """  
#         return len(self.subset_indices)  

#     @property  
#     def organ_names(self):  
#         """  
#         返回当前使用的器官名称列表  
#         """  
#         return self.required_organs  

#     @property  
#     def disease_list(self):  
#         """  
#         返回疾病名称列表  
#         """  
#         return self.disease_names


class NW_datasets_supervised(Dataset):  
    def __init__(self, h5_path,train=False, val=False, test=False,
        mask_dir="/path/to/bxg/storage/ReXGroundingCT/lesion_mask", seed=42):   
        """  
        初始化数据集  
        :param h5_path: HDF5 文件路径  
        :param train: 是否为训练集  
        :param val: 是否为验证集  
        :param test: 是否为测试集  
        :param seed: 随机种子，用于划分一致性  
        """  
        super(NW_datasets_supervised, self).__init__()  
        self.h5_path = h5_path  
        self.mask_dir=mask_dir
        # 定义需要的器官索引  

        


        # 验证 HDF5 文件是否存在  
        if not os.path.exists(self.h5_path):  
            raise FileNotFoundError(f"HDF5 文件未找到: {self.h5_path}")  

        # 打开 HDF5 文件，记录所有样本的名称  
        with h5py.File(self.h5_path, 'r') as f:  
            self.data_keys = list(f.keys())  
        # self.data_keys=[k for k in self.data_keys if k.startswith('coronacases')]
        if "total_processed" in self.h5_path.lower():
            assert os.path.exists(self.mask_dir) and os.path.isdir(self.mask_dir),f"mask目录不存在: {self.mask_dir}"
            # 只计算有mask的样本key
            
            seg_files = [fn for fn in os.listdir(mask_dir) if fn.endswith(".nii.gz")]
            seg_keys = {
                os.path.splitext(os.path.splitext(fn)[0])[0].strip()  # 去掉 .nii.gz 并 strip
                for fn in seg_files
            }
            json_path = "/path/to/bxg/storage/ReXGroundingCT/dataset.json"
            assert os.path.exists(json_path),f"json文件不存在: {json_path}"

            self.allowed_keys=seg_keys&set(self.data_keys)
            self.data_keys = list(self.allowed_keys)
            assert len(self.data_keys)>0,f"没有找到有mask的样本"

            print("num of samples with masks:", len(self.data_keys))
        if "covid_ct" in self.h5_path.lower():
            self.data_keys=[k for k in self.data_keys if k.startswith('coronacases')]
        self.data_keys.sort()
        # 设置随机种子并随机打乱索引  
        torch.manual_seed(seed)  


        if "covid_ct" in self.h5_path.lower():
            coronas = [k for k in self.data_keys if k.startswith("coronacases")]
            radios   = [k for k in self.data_keys if k.startswith("radio")]

            # 保证排序与可重复性
            coronas.sort()
            radios.sort()

            # 用户需求：仅使用指定两个 coronacases 进行训练，其他全部用于测试
            fixed_train_coronas = ["coronacases_003", "coronacases_007"]
            # 过滤出存在于数据集的指定键
            train_coronas = [k for k in fixed_train_coronas if k in coronas]
            if len(train_coronas) == 0:
                raise ValueError("指定的训练样本未在数据集中找到：coronacases_003, coronacases_007")

            # 训练集只包含指定 coronacases，不混入 radio
            train_keys = train_coronas

            # 测试样本：除训练指定外的全部样本（包括剩余 coronacases 与所有 radio）
            test_coronas = [k for k in coronas if k not in train_coronas]
            test_radios  = radios  # radios 全部进测试
            test_keys = test_coronas + test_radios

            if train:
                self.subset_indices = [self.data_keys.index(k) for k in train_keys]
            elif test:
                # 仅使用 coronacases_002 做测试
                chosen = ["coronacases_002"]
                chosen = [k for k in chosen if k in self.data_keys]
                if len(chosen) == 0:
                    raise ValueError("指定的测试样本 coronacases_002 不在数据集中。")
                self.subset_indices = [self.data_keys.index(k) for k in chosen]

                # 原有逻辑（测试集使用剩余样本）保留但注释掉
                # elif val or test:
                #     # val 与 test 都使用剩余样本
                #     self.subset_indices = [self.data_keys.index(k) for k in test_keys]
            elif val:
                # 验证沿用原默认逻辑（剩余样本集）
                self.subset_indices = [self.data_keys.index(k) for k in test_keys]
            else:
                raise ValueError("必须指定 train, val 或 test 中的一个为 True")

        else:
            # ===== 原有通用划分逻辑 =====
            indices = torch.randperm(len(self.data_keys)).tolist()
            if train or val:
                if "mosmed" in self.h5_path.lower():
                    train_size = 10
                elif "tbad" in self.h5_path.lower():
                    train_size = 62
                else:
                    total_size = len(self.data_keys)
                    train_size = int(total_size * 15 / 16)
                val_size = len(self.data_keys) - train_size

                self.train_indices = indices[:train_size]
                self.val_indices = indices[train_size:train_size + val_size]

                self.subset_indices = self.train_indices if train else self.val_indices
            elif test:
                # 通用分支下保持原逻辑（使用全部数据）；如需精确筛选，可按上面 covid_ct 分支方式仿造
                self.subset_indices = indices
            else:
                raise ValueError("必须指定 train, val 或 test 中的一个为 True") 
        print("============================================")
        print(f"NW_datasets_supervised initialized with {len(self.subset_indices)} samples.")
        print("============================================")
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
            # mask_np = f[sample_key]['mask'][:]  
            # print("ct_img shape:", ct_img.shape)
            # print("mask_np shape:", mask_np.shape)
            # 从mask_dir中读取nii.gz文件作为掩码
            if "total_processed" in self.h5_path:
                mask_np=nib.load(os.path.join(self.mask_dir, sample_key+'.nii.gz')).get_fdata()
            elif "tbad" in self.h5_path.lower():
                mask_np=f[sample_key]['mask'][0:2,...] # 只取前两个通道作为掩码 
                ct_img=ct_img[:,:,24:88,22:102]
                mask_np=mask_np[:,:,24:88,22:102]
            else:
                mask_np = f[sample_key]['mask'][:]  
            # print("ct_img shape:", ct_img.shape)
            # print("mask_np shape:", mask_np.shape)


        mask = torch.tensor(mask_np, dtype=torch.float32)         # [D, H, W]
        if len(mask.shape)==3:
            mask=mask.unsqueeze(0)  # [1, D, H, W]


        # 转换其他数据为 PyTorch 张量  
        ct_img = torch.tensor(ct_img, dtype=torch.float32)  
        if len(ct_img.shape)==3:
            ct_img=ct_img.unsqueeze(0)  # [1, D, H, W]
        # print("ct_img shape:", ct_img.shape)
        # print("mask_np shape:", mask_np.shape)
        return ct_img, mask, sample_key

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