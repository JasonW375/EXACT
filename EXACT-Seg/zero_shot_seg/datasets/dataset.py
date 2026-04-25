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

class NW_datasets_old(Dataset):  
    def __init__(self, h5_path, train=False, val=False, test=False, seed=42):   
        """  
        初始化数据集  
        :param h5_path: HDF5 文件路径  
        :param train: 是否为训练集  
        :param val: 是否为验证集  
        :param test: 是否为测试集  
        :param seed: 随机种子，用于划分一致性  
        """  
        super(NW_datasets_old, self).__init__()  
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
            raise ValueError("必须指定 train, val 或 test 中的一个为 True")  

    def __getitem__(self, index):  
        """  
        根据索引获取数据项  
        :param index: 数据索引  
        :return: CT 图像、14种疾病标签和样本键  
        """  
        # 获取样本的键  
        data_index = self.subset_indices[index]  
        sample_key = self.data_keys[data_index]  

        # 从 HDF5 文件加载数据  
        with h5py.File(self.h5_path, 'r') as f:  
            ct_img = f[sample_key]['ct'][:]  
            label_14 = f[sample_key]['label_14'][:]  # 14种疾病的标签  
            # label_18 = f[sample_key]['label_18'][:]  # 14种疾病的标签  

        # 转换为 PyTorch 张量  
        ct_img = torch.tensor(ct_img, dtype=torch.float32)  
        label_14 = torch.tensor(label_14, dtype=torch.float)  
        # label_18 = torch.tensor(label_18, dtype=torch.float)  

        return ct_img, label_14, sample_key  

    def __len__(self):  
        """  
        返回数据集大小  
        """  
        return len(self.subset_indices)

# ...existing code...
class NW_datasets(Dataset):  
    def __init__(self, h5_path, train=False, val=False, test=False, seed=42,
                 segmentations_dir="/path/to/bxg/storage/ReXGroundingCT/lesion_mask"):   
        # ...existing code...
        # 读取 HDF5 的所有键
        super(NW_datasets, self).__init__()  
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
        # 读取 segmentations 目录：先记录存在分割的样本键
        # if "total_processed_data" in self.h5_path:
        #     if not os.path.isdir(segmentations_dir):
        #         raise FileNotFoundError(f"segmentations 目录未找到: {segmentations_dir}")
        #     seg_files = [fn for fn in os.listdir(segmentations_dir) if fn.endswith(".nii.gz")]
        #     print(f"[NW_datasets] 在分割目录中找到 {len(seg_files)} 个 .nii.gz 文件")
            
        #     seg_keys = {
        #         os.path.splitext(os.path.splitext(fn)[0])[0].strip()  # 去掉 .nii.gz 并 strip
        #         for fn in seg_files
        #     }

            # json_path = "/path/to/bxg/storage/ReXGroundingCT/dataset.json"
            # if not os.path.exists(json_path):
            #     raise FileNotFoundError(f"dataset.json 未找到: {json_path}")
            # try:
            #     with open(json_path, "r", encoding="utf-8") as jf:
            #         jd = json.load(jf)
            #     for entry in _iter_entries(jd):
            #         name = entry.get("name")
            #         if not name:
            #             continue
            #         base = _strip_nii_suffix(name)  # 修复：正确去掉 .nii.gz/.nii
            #         px = entry.get("pixels", {})
            #         if isinstance(px, dict):
            #             s = sum(v for v in px.values() if isinstance(v, (int, float)))
            #         else:
            #             s = 0
            #         pixel_sums[base] = s
            # except Exception as e:
            #     raise RuntimeError(f"解析 dataset.json 失败: {e}")


            # self.allowed_keys=seg_keys  
            # self.data_keys=[k for k in self.data_keys if k in self.allowed_keys]
        # elif "covid" in self.h5_path.lower():
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
    
            # limit = min(100000, len(self.subset_indices))
            # self.subset_indices = self.subset_indices[:limit]
            # print(f"[NW_datasets] 测试集限制为 {limit} 个样本（从筛选后的 {len(indices)} 个中抽取）")
        else:  
            raise ValueError("必须指定 train, val 或 test 中的一个为 True")  

        print("==========================================")
        print("样本数:", len(self.subset_indices))
        print("==========================================")
        # assert False
    def __getitem__(self, index):  
        data_index = self.subset_indices[index]  
        sample_key = self.data_keys[data_index]  

        # 强校验：确保仅从允许列表读取
        # if sample_key not in self.allowed_keys:
        #     raise RuntimeError(f"运行期检测到未经允许的样本键: {sample_key}")

        with h5py.File(self.h5_path, 'r') as f:  
            ct_img = f[sample_key]['ct'][:]  
            if 'label_18' not in f[sample_key] and 'label_16' in f[sample_key]:
                # print(f"[NW_datasets] 警告: 样本 {sample_key} 缺少 label_18，使用 label_16 填充")
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
        
    