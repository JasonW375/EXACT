import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
import torchio as tio

def normalize_stem(p) -> str:
    """从路径中提取文件名（不含扩展名）"""
    return Path(p).stem.replace('.nii', '')  # 处理.nii.gz的情况

def read_mha_files(directory):
    """读取目录中所有的.mha文件"""
    mha_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".mha"):
                mha_files.append(os.path.join(root, f))
    return mha_files

def resize_array(array, current_spacing, target_spacing):
    """使用最近邻插值调整数组大小"""
    original_shape = array.shape[2:]  # (D,H,W)
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    # 使用最近邻插值而不是三线性插值
    resized_array = F.interpolate(array, size=new_shape, mode='nearest').cpu().numpy()
    return resized_array

def _load_mha_as_tensor(file_path):
    """使用MONAI加载MHA文件为张量"""
    monitor_pid("_load_mha_as_tensor开始")
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
    monitor_pid("_load_mha_as_tensor结束")
    return dictionary['image']  # (C,D,H,W)

def _get_spacing_from_itk(file_path):
    """从ITK图像获取体素间距"""
    monitor_pid("_get_spacing_from_itk开始")
    image = sitk.ReadImage(str(file_path))
    spacing = image.GetSpacing()  # (x, y, z)
    monitor_pid("_get_spacing_from_itk结束")
    return spacing[2], spacing[1], spacing[0]  # → return (z, y, x)

def first_stage_preprocess(file_path):
    """第一阶段预处理 - 处理多通道mask文件并合并通道"""
    monitor_pid("第一阶段预处理开始")
    
    file_path_str = str(file_path)
    key = normalize_stem(file_path_str)
    
    try:
        print(f"使用第一阶段预处理: MHA文件转换 - {file_path_str}")
        
        # 加载MHA文件
        img_data = _load_mha_as_tensor(file_path_str)  # (C,D,H,W) 即 (C,x,y,z)
        print(f"原始数据形状: {img_data.shape}")
        
        # 获取体素间距
        current = _get_spacing_from_itk(file_path_str)  # (z, y, x)
        target = (3.0, 1.0, 1.0)
        print(f"原始体素间距: {current}")
        
        # 对所有通道取并集（逻辑或操作）
        # 将多个病变mask合并为单个mask
        mask_union = torch.max(img_data, dim=0, keepdim=False)[0]  # (D,H,W) 即 (x,y,z)
        
        # 确保是二值mask (0或1)
        mask_union = (mask_union > 0).float()
        
        # 转换为numpy并换维度
        img_np = mask_union.cpu().numpy()     # (D,H,W) 即 (x,y,z)
        img_np = img_np.transpose(2, 0, 1)     # (W,D,H) 即 (z,x,y)
        tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)  # (1,1,W,D,H) 即 (1,1,z,x,y)
        
        # 重采样到目标间距
        resized_array = resize_array(tensor, current, target)    # (1,1,W',D',H') 即 (1,1,z',x',y')
        resized_array = resized_array[0][0]                      # (W',D',H') 即 (z',x',y')
        
        # 翻转y轴 - 保持(z,x,y)顺序
        resized_array = np.flip(resized_array, axis=2)  # 在y维度上翻转

        # 翻转x轴
        resized_array = np.flip(resized_array, axis=1)
        
        print(f"第一阶段预处理完成: 形状={resized_array.shape}")
        
        # 返回处理后的数组 - 已经是(z,x,y)顺序
        monitor_pid("第一阶段预处理结束")
        return {
            'data': resized_array.astype(np.float32),  # (z,x,y)格式
            'spacing': (np.float32(1.0), np.float32(1.0), np.float32(1.0)),  # 标准化的体素间距 (z,x,y)
        }
        
    except Exception as e:
        print(f"第一阶段预处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def preprocess_nii_file(file_path):
    """处理NIfTI格式文件（支持多通道mask）"""
    monitor_pid("NIfTI文件加载前")
    print(f"使用nibabel读取NIfTI格式文件: {file_path}")
    
    nii_img = nib.load(file_path)
    mask_data = nii_img.get_fdata()
    
    # 检查是否为多通道数据
    if len(mask_data.shape) == 4:
        # 4D数据 (c,x,y,z) - 多通道mask
        print(f"检测到多通道mask: {mask_data.shape}")
        # 对所有通道取并集
        mask_union = np.max(mask_data, axis=0)  # (x,y,z)
    else:
        # 3D数据 (x,y,z)
        mask_union = mask_data

    
    # 确保是二值mask
    mask_union = (mask_union > 0).astype(np.float32)
    
    # NIfTI格式通常为(x,y,z)，需要转置为(z,x,y)
    mask_union = np.transpose(mask_union, (2, 0, 1))
    
    # 获取体素间距
    spacing = nii_img.header.get_zooms()
    spacing = (spacing[2], spacing[0], spacing[1])  # (x,y,z) → (z,x,y)
    
    monitor_pid("NIfTI文件加载后")
    print(f"NIfTI数据加载完成: 形状={mask_union.shape}, 体素间距={spacing}")
    
    return {
        'data': mask_union.astype(np.float32),
        'spacing': spacing,
    }

def resize_data(image, target_shape=(64, 128, 128)):
    """
    使用torchio重新调整图像大小（最近邻插值）
    输入: torchio.ScalarImage对象
    输出: 调整大小后的torchio.ScalarImage对象
    """
    monitor_pid("resize操作前")
    # 对mask使用最近邻插值
    resize_transform = tio.Resize(target_shape, image_interpolation='nearest')
    # resize_transform = tio.Resize(target_shape, image_interpolation='bspline')
    result = resize_transform(image)
    monitor_pid("resize操作后")
    return result

def resample_data(image, target_spacing=(1, 1, 1)):
    """
    使用torchio重采样图像（最近邻插值）
    输入: torchio.ScalarImage对象
    输出: 重采样后的torchio.ScalarImage对象
    """
    monitor_pid("resample操作前")
    # 对mask使用最近邻插值
    resample_transform = tio.Resample(target_spacing, image_interpolation='nearest')
    # resample_transform = tio.Resample(target_spacing, image_interpolation='bspline')
    result = resample_transform(image)
    monitor_pid("resample操作后")
    return result

def preprocess_volume(file_path):
    """处理单个mask体积文件，包含通道合并和尺寸调整"""
    try:
        file_path_str = str(file_path)
        print(f"正在处理文件: {file_path_str}")
        monitor_pid("预处理开始")
        
        # 判断文件类型并进行相应预处理
        if file_path_str.lower().endswith('.mha'):
            # MHA文件 - 使用第一阶段预处理
            processed_data = first_stage_preprocess(file_path)
        elif file_path_str.endswith('.nii.gz') or file_path_str.endswith('.nii'):
            # NIfTI文件 - 使用NIfTI加载处理
            processed_data = preprocess_nii_file(file_path_str)
        else:
            raise ValueError(f"不支持的文件格式: {file_path_str}。仅支持.nii、.nii.gz和.mha格式。")
        
        # 获取处理后的数据
        mask_data = processed_data['data']        # 此时应为(z,x,y)格式
        spacing = processed_data['spacing']       # 此时应为(z,x,y)格式
        
        print(f"预处理后数据形状: {mask_data.shape} - 应为(z,x,y)格式")
        print(f"预处理后体素间距: {spacing} mm - 应为(z,x,y)格式")
        
        # 扩展维度以匹配预期格式
        mask_data = np.expand_dims(mask_data, axis=0)  # (1,z,x,y)
        
        # 创建torchio对象，并传入正确的空间信息
        monitor_pid("创建TorchIO对象前")
        mask_tensor = torch.from_numpy(mask_data)
        mask_subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=mask_tensor,
                spacing=spacing  # 提供正确的空间信息
            )
        )
        monitor_pid("创建TorchIO对象后")
        
        # 应用重采样和调整大小
        print("应用重采样处理...")
        mask_resampled_subject = resample_data(mask_subject.image)
        print(f"重采样后形状: {mask_resampled_subject.shape}")
        
        print("调整图像大小至64x128x128...")
        mask_resized_subject = resize_data(mask_resampled_subject, target_shape=(64, 128, 128))
        print(f"调整大小后形状: {mask_resized_subject.shape}")
        
        # 获取处理后的张量
        mask_tensor = mask_resized_subject.data
        
        # 确保输出仍然是二值mask
        mask_tensor = (mask_tensor > 0.5).float()
        
        monitor_pid("预处理完成后")
        
        return mask_tensor
        
    except Exception as e:
        print(f"预处理 {file_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def monitor_pid(checkpoint_name=""):
    """监控进程ID和线程数（辅助函数）"""
    import psutil
    process = psutil.Process()
    thread_count = process.num_threads()
    print(f"[{checkpoint_name}] PID: {process.pid}, 线程数: {thread_count}")
    return thread_count

def strip_known_suffixes(filename):
    """移除已知的文件后缀（辅助函数）
    说明：原实现先剥离 .gz 导致 .nii.gz 变成 .nii，再拼接 .nii.gz 产生 .nii.nii.gz。
    修复：优先剥离复合后缀 .nii.gz，并在可能的情况下循环剥离一次，直到不再匹配。
    """
    name = filename
    lowered = name.lower()
    # 循环最多两次，处理诸如 .nii.gz / .mha / .nii 等情况
    for _ in range(2):
        if lowered.endswith('.nii.gz'):
            name = name[: -len('.nii.gz')]
        elif lowered.endswith('.mha'):
            name = name[: -len('.mha')]
        elif lowered.endswith('.nii'):
            name = name[: -len('.nii')]
        else:
            break
        lowered = name.lower()
    return name

# 主函数示例
def process_mask_files(input_dir, output_dir):
    """处理目录中的所有mask文件"""
    import os
    from pathlib import Path
    
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    # 仅选择医学影像相关的掩码文件，避免把任意 .gz 误当作输入
    mask_files = sorted([
        p for p in input_path.iterdir()
        if p.name.lower().endswith(('.mha', '.nii', '.nii.gz'))
    ])

    mask_files=[f for f in mask_files if "valid" in f.name]
    # mask_files=[Path("/FM_data/bxg/CT-RATE/CT-RATE-valid-fixed/dataset/valid_fixed/valid_1250/valid_1250_b/valid_1250_b_1.nii.gz")]
    # sample_name=os.listdir("/FM_data/bxg/CT_Report/CT_Report9_test/results/segmamba__Sunday_12_October_2025_15h_32m_35s/test_results/prediction_heatmaps/epoch_10")
    # sample_name=[f"{sample}.nii.gz" for sample in sample_name]
    # mask_files=[Path(f"/FM_data/bxg/CT_Report/CT_Report9_test/ReXGroundingCT/segmentations/{name}") for name in sample_name]
    if not mask_files:
        print(f"未在 {input_dir} 中找到mask文件")
        return
    
    print(f"找到 {len(mask_files)} 个mask文件")
    
    for file_path in mask_files:
        try:
            print(f"\n处理: {file_path.name}")
            
            # 预处理mask文件（包含通道合并和尺寸调整）
            processed_mask = preprocess_volume(file_path)
            
            print(f"最终mask形状: {processed_mask.shape}")
            print(f"Mask值范围: [{processed_mask.min():.2f}, {processed_mask.max():.2f}]")
            print(f"非零元素比例: {(processed_mask > 0).sum() / processed_mask.numel():.4f}")
            
            # 规范化输出基础名，确保不会出现 .nii.nii.gz
            base_name = strip_known_suffixes(file_path.name)
            output_file = Path(output_dir) / f"{base_name}.nii.gz"
            if output_file.exists():
                print(f"目标文件已存在，跳过: {output_file}")
                continue
            # 将tensor转换为numpy数组，并调整维度以匹配NIfTI格式 (x, y, z)
            # 当前格式: (1, z, x, y) -> 目标格式: (x, y, z)
            mask_numpy = processed_mask.squeeze(0).cpu().numpy()  # (z, x, y)
            mask_numpy=mask_numpy[...,::-1,::-1]
            # mask_numpy = np.transpose(mask_numpy, (1, 2, 0))      # (x, y, z)

            # 创建一个NIfTI图像对象
            # 仿射矩阵设为单位矩阵，因为已经重采样到1mm间距
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(mask_numpy.astype(np.uint8), affine)

            # 保存NIfTI文件
            nib.save(nifti_img, output_file)
            print(f"已保存至: {output_file}")
            
        except Exception as e:
            print(f"处理 {file_path} 时出错: {str(e)}")
            continue
    
    print("\n所有文件处理完成")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分割Mask预处理")
    parser.add_argument("--input", type=str, default="/path/to/bxg/storage/ReXGroundingCT/segmentations", help="输入mask文件目录")
    parser.add_argument("--output", type=str, default="/path/to/bxg/storage/ReXGroundingCT/lesion_mask", help="输出目录")
    
    args = parser.parse_args()
    process_mask_files(args.input, args.output)