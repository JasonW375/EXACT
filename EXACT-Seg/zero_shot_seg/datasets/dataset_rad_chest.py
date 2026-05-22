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


# ...existing code...
import os
import h5py
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import numpy as np


class My_datasets(Dataset):
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
        super(My_datasets, self).__init__()
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


class My_datasets_18(Dataset):
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
        super(My_datasets_18, self).__init__()
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