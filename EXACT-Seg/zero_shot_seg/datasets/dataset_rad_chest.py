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

# class NW_datasets(Dataset):  
#     def __init__(self, h5_path, train=False, val=False, test=False, seed=42):   
#         """  
#         初始化数据集  
#         :param h5_path: HDF5 文件路径  
#         :param train: 是否为训练集  
#         :param val: 是否为验证集  
#         :param test: 是否为测试集  
#         :param seed: 随机种子，用于划分一致性  
#         """  
#         super(NW_datasets, self).__init__()  
#         self.h5_path = h5_path  
        
#         # 定义18种疾病名称  
#         self.disease_names = [  
#             'Medical material','Arterial wall calcification', 'Cardiomegaly', 
#         'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia',
#         'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity',
#           'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
#           'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
#           'Interlobular septal thickening'
#         ]  

#         # 验证 HDF5 文件是否存在  
#         if not os.path.exists(self.h5_path):  
#             raise FileNotFoundError(f"HDF5 文件未找到: {self.h5_path}")  

#         # 打开 HDF5 文件，记录所有样本的名称  
#         with h5py.File(self.h5_path, 'r') as f:  
#             self.data_keys = list(f.keys())  

#         # 设置随机种子并随机打乱索引  
#         torch.manual_seed(seed)  
#         indices = torch.randperm(len(self.data_keys)).tolist()  

#         # 根据参数选择当前子集  
#         if train or val:  
#             # 训练和验证时按照 11:1 划分  
#             total_size = len(self.data_keys)  
#             train_size = int(total_size * 15 / 16)  
#             val_size = total_size - train_size  

#             self.train_indices = indices[:train_size]  
#             self.val_indices = indices[train_size:]  

#             if train:  
#                 self.subset_indices = self.train_indices  
#             else:  # val  
#                 self.subset_indices = self.val_indices  
        
#         elif test:  
#             # 测试时使用全部数据  
#             self.subset_indices = indices
        
#         else:  
#             raise ValueError("必须指定 train, val 或 test 中的一个为 True")  

    # def __getitem__(self, index):  
    #     """  
    #     根据索引获取数据项  
    #     :param index: 数据索引  
    #     :return: CT 图像、14种疾病标签和样本键  
    #     """  
    #     # 获取样本的键  
    #     data_index = self.subset_indices[index]  
    #     sample_key = self.data_keys[data_index]  

    #     # 从 HDF5 文件加载数据  
    #     with h5py.File(self.h5_path, 'r') as f:  
    #         ct_img = f[sample_key]['ct'][:]  
    #         # label_14 = f[sample_key]['label_14'][:]  # 14种疾病的标签  
    #         label_18 = f[sample_key]['label_18'][:]  # 14种疾病的标签  

    #     # 转换为 PyTorch 张量  
    #     ct_img = torch.tensor(ct_img, dtype=torch.float32)  
    #     # label_14 = torch.tensor(label_14, dtype=torch.float)  
    #     label_18 = torch.tensor(label_18, dtype=torch.float)  

    #     return ct_img, label_18, sample_key  

    # def __len__(self):  
    #     """  
    #     返回数据集大小  
    #     """  
    #     return len(self.subset_indices)

# ...existing code...
import os
import h5py
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np


class NW_datasets(Dataset):
    def __init__(self, h5_path, train=False, val=False, test=False, seed=42, 
                 visualize_samples=True, num_viz_samples=10, viz_save_dir='./visualizations'):
        """
        简化版数据集类，用于测试16种疾病标签
        
        Args:
            h5_path: H5文件路径
            train: 是否为训练集
            val: 是否为验证集
            test: 是否为测试集
            seed: 随机种子
            visualize_samples: 是否可视化前几个样本
            num_viz_samples: 可视化样本的数量
            viz_save_dir: 可视化结果保存目录
        """
        super(NW_datasets, self).__init__()
        self.h5_path = h5_path
        self.visualize_samples = visualize_samples
        self.num_viz_samples = num_viz_samples
        self.viz_save_dir = viz_save_dir
        
        # 如果需要可视化，创建保存目录
        if self.visualize_samples and not os.path.exists(self.viz_save_dir):
            os.makedirs(self.viz_save_dir)
            print(f"创建可视化保存目录: {self.viz_save_dir}")
        
        # 定义18种疾病名称（完整版本，用于保持代码兼容性）
        self.disease_names = [
            'Medical material',                      # 0
            'Arterial wall calcification',           # 1
            'Cardiomegaly',                          # 2
            'Pericardial effusion',                  # 3
            'Coronary artery wall calcification',    # 4 (缺失，占位)
            'Hiatal hernia',                         # 5
            'Lymphadenopathy',                       # 6
            'Emphysema',                             # 7
            'Atelectasis',                           # 8
            'Lung nodule',                           # 9
            'Lung opacity',                          # 10
            'Pulmonary fibrotic sequela',            # 11
            'Pleural effusion',                      # 12
            'Mosaic attenuation pattern',            # 13 (缺失，占位)
            'Peribronchial thickening',              # 14
            'Consolidation',                         # 15
            'Bronchiectasis',                        # 16
            'Interlobular septal thickening'         # 17
        ]
        
        # 16种疾病到18种疾病的映射索引
        self.label_16_to_18_mapping = [
            0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17
        ]
        
        # 验证 HDF5 文件是否存在
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"HDF5 文件未找到: {self.h5_path}")
        
        print("=" * 80)
        print(f"初始化数据集: {h5_path}")
        print("=" * 80)
        
        # 读取 HDF5 的所有键
        with h5py.File(self.h5_path, 'r') as f:
            all_keys = sorted([k.strip() for k in f.keys()])
            print(f"HDF5 键总数: {len(all_keys)}")
            
            # 检查是否有label_16数据集
            sample_key = all_keys[0] if all_keys else None
            if sample_key:
                if 'label_16' not in f[sample_key]:
                    raise KeyError(f"样本 {sample_key} 中未找到 'label_16' 数据集")
                else:
                    print(f"✓ 验证成功：样本包含 label_16 数据集")
                    
                # 打印CT图像的形状信息
                ct_shape = f[sample_key]['ct'].shape
                print(f"✓ CT图像形状: {ct_shape}")
        
        # 根据test参数筛选键（如果只要测试集）
        if test:
            self.data_keys = all_keys
            print(f"测试模式：使用所有 {len(self.data_keys)} 个样本")
        else:
            self.data_keys = all_keys
        
        if len(self.data_keys) == 0:
            raise RuntimeError("没有可用样本，请检查 HDF5 文件")
        
        # 设置随机种子并随机打乱索引
        torch.manual_seed(seed)
        indices = torch.randperm(len(self.data_keys)).tolist()
        
        # 根据参数选择当前子集
        if train or val:
            total_size = len(self.data_keys)
            train_size = int(total_size * 15 / 16)
            self.train_indices = indices[:train_size]
            self.val_indices = indices[train_size:]
            self.subset_indices = self.train_indices if train else self.val_indices
            print(f"{'训练集' if train else '验证集'}样本数: {len(self.subset_indices)}")
        elif test:
            self.subset_indices = indices
            print(f"测试集样本数: {len(self.subset_indices)}")
        else:
            raise ValueError("必须指定 train, val 或 test 中的一个为 True")
        
        print("=" * 80)
        print(f"数据集初始化完成")
        print(f"  - 总样本数: {len(self.data_keys)}")
        print(f"  - 当前子集样本数: {len(self.subset_indices)}")
        print(f"  - 疾病类别数: {len(self.disease_names)} (18种)")
        print(f"  - 标签来源: label_16 (自动扩展为 label_18)")
        print("=" * 80)
        
        # 如果需要可视化，执行可视化
        if self.visualize_samples:
            self._visualize_initial_samples()
    
    def _convert_label_16_to_18(self, label_16):
        """
        将16维标签转换为18维标签
        在索引4和13位置插入0（对应缺失的两种疾病）
        
        Args:
            label_16: 16维标签数组 [16]
        
        Returns:
            label_18: 18维标签数组 [18]
        """
        label_18 = torch.zeros(18, dtype=label_16.dtype)
        
        # 按照映射填充label_18
        label_18[0:4] = label_16[0:4]
        label_18[5:13] = label_16[4:12]
        label_18[14:18] = label_16[12:16]
        
        return label_18
    
    def visualize_ct_slices(self, ct_img, sample_key, save_path=None):
        """
        可视化CT图像的三个正交平面的中心切片
        
        Args:
            ct_img: CT图像张量 [C, D, H, W] 或 [D, H, W]
            sample_key: 样本ID
            save_path: 保存路径，如果为None则不保存
        """
        # 转换为numpy数组
        if isinstance(ct_img, torch.Tensor):
            ct_img = ct_img.numpy()
        
        # 处理通道维度
        if ct_img.ndim == 4:  # [C, D, H, W]
            ct_img = ct_img[0]  # 取第一个通道 [D, H, W]
        
        D, H, W = ct_img.shape
        
        # 获取三个维度的中心切片
        axial_slice = ct_img[D // 2, :, :]      # 轴向切片 (Axial) [H, W]
        coronal_slice = ct_img[:, H // 2, :]    # 冠状切片 (Coronal) [D, W]
        sagittal_slice = ct_img[:, :, W // 2]   # 矢状切片 (Sagittal) [D, H]
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'CT Slices - Sample: {sample_key}\nShape: [D={D}, H={H}, W={W}]', 
                     fontsize=14, fontweight='bold')
        
        # 轴向切片 (从上往下看)
        im0 = axes[0].imshow(axial_slice, cmap='gray', aspect='auto')
        axes[0].set_title(f'Axial (横断面)\nSlice {D // 2}/{D}')
        axes[0].set_xlabel('Width')
        axes[0].set_ylabel('Height')
        axes[0].axis('on')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 冠状切片 (从前往后看)
        im1 = axes[1].imshow(coronal_slice, cmap='gray', aspect='auto')
        axes[1].set_title(f'Coronal (冠状面)\nSlice {H // 2}/{H}')
        axes[1].set_xlabel('Width')
        axes[1].set_ylabel('Depth')
        axes[1].axis('on')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 矢状切片 (从左往右看)
        im2 = axes[2].imshow(sagittal_slice, cmap='gray', aspect='auto')
        axes[2].set_title(f'Sagittal (矢状面)\nSlice {W // 2}/{W}')
        axes[2].set_xlabel('Height')
        axes[2].set_ylabel('Depth')
        axes[2].axis('on')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存可视化图像: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def _visualize_initial_samples(self):
        """
        可视化数据集的前几个样本
        """
        print("\n" + "=" * 80)
        print(f"开始可视化前 {self.num_viz_samples} 个样本...")
        print("=" * 80)
        
        num_samples = min(self.num_viz_samples, len(self.subset_indices))
        
        for i in range(num_samples):
            # 获取样本
            ct_img, label_18, sample_key = self.__getitem__(i)
            
            # 显示标签信息
            positive_labels = []
            for idx, val in enumerate(label_18):
                if val > 0:
                    positive_labels.append(f"{self.disease_names[idx]} ({val:.2f})")
            
            print(f"\n样本 {i+1}/{num_samples} - ID: {sample_key}")
            print(f"  - CT形状: {ct_img.shape}")
            print(f"  - 阳性标签: {', '.join(positive_labels) if positive_labels else '无阳性标签'}")
            
            # 生成保存路径
            save_path = os.path.join(self.viz_save_dir, f'sample_{i+1}_{sample_key}.png')
            
            # 可视化
            self.visualize_ct_slices(ct_img, sample_key, save_path)
        
        print("\n" + "=" * 80)
        print(f"可视化完成！图像已保存至: {self.viz_save_dir}")
        print("=" * 80 + "\n")
    
    def __getitem__(self, index):
        """
        获取单个样本
        
        Returns:
            ct_img: CT图像 [C, D, H, W]
            label_18: 18维疾病标签 [18]
            sample_key: 样本ID
        """
        data_index = self.subset_indices[index]
        sample_key = self.data_keys[data_index]
        
        # 从H5文件读取数据
        with h5py.File(self.h5_path, 'r') as f:
            ct_img = f[sample_key]['ct'][:]
            label_16 = f[sample_key]['label_16'][:]
        
        # 转换为tensor
        ct_img = torch.tensor(ct_img, dtype=torch.float32)
        label_16 = torch.tensor(label_16, dtype=torch.float32)
        
        # 将16维标签转换为18维标签
        label_18 = self._convert_label_16_to_18(label_16)
        
        return ct_img, label_18, sample_key
    
    def __len__(self):
        return len(self.subset_indices)
    
    @property
    def disease_list(self):
        """
        返回疾病名称列表
        """
        return self.disease_names
    
    def get_label_info(self):
        """
        获取标签信息，用于调试
        """
        print("\n" + "=" * 80)
        print("标签信息")
        print("=" * 80)
        print("18种疾病列表（带占位符）:")
        for idx, name in enumerate(self.disease_names):
            marker = " (占位，无数据)" if idx in [4, 13] else ""
            print(f"  {idx:2d}. {name}{marker}")
        print("=" * 80)


class NW_datasets_18(Dataset):
    def __init__(self, h5_path, train=False, val=False, test=False, seed=42, 
                 visualize_samples=True, num_viz_samples=10, viz_save_dir='./visualizations'):
        """
        数据集类，直接使用18种疾病标签（label_18）
        
        Args:
            h5_path: H5文件路径
            train: 是否为训练集
            val: 是否为验证集
            test: 是否为测试集
            seed: 随机种子
            visualize_samples: 是否可视化前几个样本
            num_viz_samples: 可视化样本的数量
            viz_save_dir: 可视化结果保存目录
        """
        super(NW_datasets_18, self).__init__()
        self.h5_path = h5_path
        self.visualize_samples = visualize_samples
        self.num_viz_samples = num_viz_samples
        self.viz_save_dir = viz_save_dir
        
        # 如果需要可视化，创建保存目录
        if self.visualize_samples and not os.path.exists(self.viz_save_dir):
            os.makedirs(self.viz_save_dir)
            print(f"创建可视化保存目录: {self.viz_save_dir}")
        
        # 定义18种疾病名称
        self.disease_names = [
            'Medical material',                      # 0
            'Arterial wall calcification',           # 1
            'Cardiomegaly',                          # 2
            'Pericardial effusion',                  # 3
            'Coronary artery wall calcification',    # 4
            'Hiatal hernia',                         # 5
            'Lymphadenopathy',                       # 6
            'Emphysema',                             # 7
            'Atelectasis',                           # 8
            'Lung nodule',                           # 9
            'Lung opacity',                          # 10
            'Pulmonary fibrotic sequela',            # 11
            'Pleural effusion',                      # 12
            'Mosaic attenuation pattern',            # 13
            'Peribronchial thickening',              # 14
            'Consolidation',                         # 15
            'Bronchiectasis',                        # 16
            'Interlobular septal thickening'         # 17
        ]
        
        # 验证 HDF5 文件是否存在
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"HDF5 文件未找到: {self.h5_path}")
        
        print("=" * 80)
        print(f"初始化数据集: {h5_path}")
        print("=" * 80)
        
        # 读取 HDF5 的所有键
        with h5py.File(self.h5_path, 'r') as f:
            all_keys = sorted([k.strip() for k in f.keys()])
            print(f"HDF5 键总数: {len(all_keys)}")
            
            # 检查是否有label_18数据集
            sample_key = all_keys[0] if all_keys else None
            if sample_key:
                if 'label_18' not in f[sample_key]:
                    raise KeyError(f"样本 {sample_key} 中未找到 'label_18' 数据集")
                else:
                    print(f"✓ 验证成功：样本包含 label_18 数据集")
                    label_18_shape = f[sample_key]['label_18'].shape
                    print(f"✓ label_18 形状: {label_18_shape}")
                    
                # 打印CT图像的形状信息
                ct_shape = f[sample_key]['ct'].shape
                print(f"✓ CT图像形状: {ct_shape}")
        
        # 根据test参数筛选键（如果只要测试集）
        if test:
            self.data_keys = all_keys
            print(f"测试模式：使用所有 {len(self.data_keys)} 个样本")
        else:
            self.data_keys = all_keys
        
        if len(self.data_keys) == 0:
            raise RuntimeError("没有可用样本，请检查 HDF5 文件")
        
        # 设置随机种子并随机打乱索引
        torch.manual_seed(seed)
        indices = torch.randperm(len(self.data_keys)).tolist()
        
        # 根据参数选择当前子集
        if train or val:
            total_size = len(self.data_keys)
            train_size = int(total_size * 15 / 16)
            self.train_indices = indices[:train_size]
            self.val_indices = indices[train_size:]
            self.subset_indices = self.train_indices if train else self.val_indices
            print(f"{'训练集' if train else '验证集'}样本数: {len(self.subset_indices)}")
        elif test:
            self.subset_indices = indices
            print(f"测试集样本数: {len(self.subset_indices)}")
        else:
            raise ValueError("必须指定 train, val 或 test 中的一个为 True")
        
        print("=" * 80)
        print(f"数据集初始化完成")
        print(f"  - 总样本数: {len(self.data_keys)}")
        print(f"  - 当前子集样本数: {len(self.subset_indices)}")
        print(f"  - 疾病类别数: {len(self.disease_names)} (18种)")
        print(f"  - 标签来源: label_18 (直接读取)")
        print("=" * 80)
        
        # 如果需要可视化，执行可视化
        if self.visualize_samples:
            self._visualize_initial_samples()
    
    def visualize_ct_slices(self, ct_img, sample_key, save_path=None):
        """
        可视化CT图像的三个正交平面的中心切片
        
        Args:
            ct_img: CT图像张量 [C, D, H, W] 或 [D, H, W]
            sample_key: 样本ID
            save_path: 保存路径，如果为None则不保存
        """
        # 转换为numpy数组
        if isinstance(ct_img, torch.Tensor):
            ct_img = ct_img.numpy()
        
        # 处理通道维度
        if ct_img.ndim == 4:  # [C, D, H, W]
            ct_img = ct_img[0]  # 取第一个通道 [D, H, W]
        
        D, H, W = ct_img.shape
        
        # 获取三个维度的中心切片
        axial_slice = ct_img[D // 2, :, :]      # 轴向切片 (Axial) [H, W]
        coronal_slice = ct_img[:, H // 2, :]    # 冠状切片 (Coronal) [D, W]
        sagittal_slice = ct_img[:, :, W // 2]   # 矢状切片 (Sagittal) [D, H]
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'CT Slices - Sample: {sample_key}\nShape: [D={D}, H={H}, W={W}]', 
                     fontsize=14, fontweight='bold')
        
        # 轴向切片 (从上往下看)
        im0 = axes[0].imshow(axial_slice, cmap='gray', aspect='auto')
        axes[0].set_title(f'Axial (横断面)\nSlice {D // 2}/{D}')
        axes[0].set_xlabel('Width')
        axes[0].set_ylabel('Height')
        axes[0].axis('on')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        # 冠状切片 (从前往后看)
        im1 = axes[1].imshow(coronal_slice, cmap='gray', aspect='auto')
        axes[1].set_title(f'Coronal (冠状面)\nSlice {H // 2}/{H}')
        axes[1].set_xlabel('Width')
        axes[1].set_ylabel('Depth')
        axes[1].axis('on')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 矢状切片 (从左往右看)
        im2 = axes[2].imshow(sagittal_slice, cmap='gray', aspect='auto')
        axes[2].set_title(f'Sagittal (矢状面)\nSlice {W // 2}/{W}')
        axes[2].set_xlabel('Height')
        axes[2].set_ylabel('Depth')
        axes[2].axis('on')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 保存可视化图像: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def _visualize_initial_samples(self):
        """
        可视化数据集的前几个样本
        """
        print("\n" + "=" * 80)
        print(f"开始可视化前 {self.num_viz_samples} 个样本...")
        print("=" * 80)
        
        num_samples = min(self.num_viz_samples, len(self.subset_indices))
        
        for i in range(num_samples):
            # 获取样本
            ct_img, label_18, sample_key = self.__getitem__(i)
            
            # 显示标签信息
            positive_labels = []
            for idx, val in enumerate(label_18):
                if val > 0:
                    positive_labels.append(f"{self.disease_names[idx]} ({val:.2f})")
            
            print(f"\n样本 {i+1}/{num_samples} - ID: {sample_key}")
            print(f"  - CT形状: {ct_img.shape}")
            print(f"  - 阳性标签: {', '.join(positive_labels) if positive_labels else '无阳性标签'}")
            
            # 生成保存路径
            save_path = os.path.join(self.viz_save_dir, f'sample_{i+1}_{sample_key}.png')
            
            # 可视化
            self.visualize_ct_slices(ct_img, sample_key, save_path)
        
        print("\n" + "=" * 80)
        print(f"可视化完成！图像已保存至: {self.viz_save_dir}")
        print("=" * 80 + "\n")
    
    def __getitem__(self, index):
        """
        获取单个样本
        
        Returns:
            ct_img: CT图像 [C, D, H, W]
            label_18: 18维疾病标签 [18]
            sample_key: 样本ID
        """
        data_index = self.subset_indices[index]
        sample_key = self.data_keys[data_index]
        
        # 从H5文件读取数据
        with h5py.File(self.h5_path, 'r') as f:
            ct_img = f[sample_key]['ct'][:]
            label_18 = f[sample_key]['label_18'][:]  # 直接读取label_18
        
        # 转换为tensor
        ct_img = torch.tensor(ct_img, dtype=torch.float32)
        label_18 = torch.tensor(label_18, dtype=torch.float32)
        
        return ct_img, label_18, sample_key
    
    def __len__(self):
        return len(self.subset_indices)
    
    @property
    def disease_list(self):
        """
        返回疾病名称列表
        """
        return self.disease_names
    
    def get_label_info(self):
        """
        获取标签信息，用于调试
        """
        print("\n" + "=" * 80)
        print("标签信息")
        print("=" * 80)
        print("18种疾病列表:")
        for idx, name in enumerate(self.disease_names):
            print(f"  {idx:2d}. {name}")
        print("=" * 80)    