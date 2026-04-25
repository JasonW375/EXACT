import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import wandb
import time
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
import math
import nibabel as nib 
import os
import seaborn as sns

from sklearn.metrics import roc_auc_score  
import numpy as np  
import torch  
from tqdm import tqdm  
from datetime import datetime  
from pathlib import Path
# 添加混淆矩阵的计算和可视化
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve  
import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np  
import torch  
from tqdm import tqdm  
import wandb  
from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt  



import matplotlib.patches as mpatches 
import csv  # 添加这一行导入csv模块

def calculate_dice(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2 * intersection + 1e-7) / (union + 1e-7)  # 防止除零
    return dice.mean().item()
# 双向权重
import torch
import torch.nn.functional as F


#一次/根号
def weighted_binary_cross_entropy(pred, target, weight_pos, weight_neg):
    """
    计算加权的二元交叉熵损失。
    pred: 预测值，形状为 (B,)
    target: 真实标签，形状为 (B,)
    weight_pos: 正样本权重
    weight_neg: 负样本权重
    """
    # 计算标准的二元交叉熵损失
    loss_pos = weight_pos * target * torch.log(torch.sigmoid(pred) + 1e-6)
    loss_neg = weight_neg * (1 - target) * torch.log(1 - torch.sigmoid(pred) + 1e-6)
    loss = -(loss_pos + loss_neg)
    return loss

def get_organ_disease_mapping():  
    """返回疾病到器官的映射关系  
    现在的器官顺序是:  
    0: lung  
    1: trachea and bronchie  
    2: pleura  
    3: mediastinum  
    4: heart  
    5: esophagus  
    """  
    return {  
        0: 4,    # Cardiomegaly -> heart (4)  
        1: 4,    # Pericardial effusion -> heart (4)  
        2: 4,    # Coronary artery wall calcification -> heart (4)  
        3: 5,    # Hiatal hernia -> esophagus (5)  
        4: 3,    # Lymphadenopathy -> mediastinum (3)  
        5: 0,    # Emphysema -> lung (0)      
        6: 0,    # Atelectasis -> lung (0)  
        7: 0,    # Lung nodule -> lung (0)  
        8: 0,    # Lung opacity -> lung (0)  
        9: 0,    # Pulmonary fibrotic sequela -> lung (0)  
        10: 2,   # Pleural effusion -> pleura (2)  
        11: 0,   # Mosaic attenuation pattern -> lung (0)  
        12: 1,   # Peribronchial thickening -> trachea and bronchie (1)  
        13: 0,   # Consolidation -> lung (0)  
        14: 1,   # Bronchiectasis -> trachea and bronchie (1)  
        15: 0,   # Interlobular septal thickening -> lung (0)  
    }

def Abnormal_loss_multiscale(seg_pred, abnormal_preds, abnormal_targets, disease_frequencies, k=3, epsilon=1e-6, seg_threshold=0.5):  
    """  
    计算多尺度异常检测损失  
    Args:  
        seg_pred: 分割预测 (B, 6, D, H, W)  
        abnormal_preds: 疾病预测列表 [scale1_pred, scale2_pred]，每个元素形状为(B, 16, D, H, W)  
        abnormal_targets: 疾病标签 (B, 16)  
        disease_frequencies: 16种疾病的阳性样本频率列表  
        k: top-k取值  
        epsilon: 数值稳定性常数  
        seg_threshold: 分割掩码阈值  
    """  
    B, _, D, H, W = seg_pred.shape  
    num_diseases = abnormal_preds[0].shape[1]  # 16个疾病通道  
    num_scales = len(abnormal_preds)  # 尺度数量  
    
    # 获取疾病到器官的映射  
    disease_to_organ = get_organ_disease_mapping()  
    
    # 对分割预测进行二值化  
    seg_mask = (seg_pred > seg_threshold).float()  
    
    # 为每个尺度准备列表  
    disease_losses = [[] for _ in range(num_scales)]  
    disease_predictions = [[] for _ in range(num_scales)]  
    
    for scale_idx, abnormal_pred in enumerate(abnormal_preds):  
        current_k = k * (8 ** scale_idx)  # 每进入下一个尺度，k 乘以 8
        # print(f"Scale {scale_idx}: current_k = {current_k}")  # 打印当前尺度和对应的 k 值 
        # 将分割掩码调整到当前尺度  
        current_size = abnormal_pred.shape[2:]  
        scaled_seg_mask = F.interpolate(seg_mask, size=current_size, mode='trilinear', align_corners=False)  
        
        for disease_idx in range(num_diseases):  
            # 获取对应的器官索引  
            organ_idx = disease_to_organ[disease_idx]  
            
            # 将对应器官的分割掩码与疾病预测相乘  
            organ_mask = scaled_seg_mask[:, organ_idx:organ_idx+1]  
            disease_pred = abnormal_pred[:, disease_idx:disease_idx+1]  
            
            final_pred = organ_mask * disease_pred  
            
            # 取top-k值的平均  
            top_k_values, _ = torch.topk(final_pred.view(B, -1), current_k, dim=1)  
            avg_top_k = top_k_values.mean(dim=1)  
            
            # 使用疾病的频率计算权重  
            freq = disease_frequencies[disease_idx]  
            weight_pos = (1 - freq + epsilon) / (freq + epsilon)  
            weight_neg = 1.0  
            
            # 计算损失  
            target = abnormal_targets[:, disease_idx]  
            loss = weighted_binary_cross_entropy(avg_top_k, target, weight_pos, weight_neg)  
            loss = loss.mean()  
            
            disease_losses[scale_idx].append(loss)  
            disease_predictions[scale_idx].append(avg_top_k)  
    
    # 返回每个尺度的损失和预测  
    scale_losses = [torch.stack(losses) for losses in disease_losses]  
    scale_predictions = [torch.stack(preds, dim=1) for preds in disease_predictions]  
    
    return scale_losses, scale_predictions



# 添加动态权重计算函数  
def calculate_dynamic_weight(epoch, initial_weight, total_epochs, decay_rate=9.0):  
    """  
    计算动态权重  
    :param epoch: 当前epoch  
    :param initial_weight: 初始权重  
    :param total_epochs: 总epoch数  
    :param decay_rate: 衰减率  
    :return: 当前权重  
    """  
    k = decay_rate / total_epochs  
    weight =initial_weight * math.exp(-k * epoch) 
    weight = max(weight, 0.5)
    return weight


def upsample_3d(tensor, target_size):  
    """  
    3D上采样函数，将输入张量调整到目标大小  
    
    Args:  
        tensor (torch.Tensor): 输入张量，形状为 [B, C, D, H, W]  
        target_size (tuple): 目标大小 (D, H, W)  
    
    Returns:  
        torch.Tensor: 上采样后的张量  
    """  
    # 使用三线性插值进行上采样  
    return F.interpolate(  
        tensor,   
        size=target_size,   
        mode='trilinear',   
        align_corners=False  
    )  

# def save_prediction_heatmaps(  
#     predictions,   
#     targets,   
#     images,   
#     epoch,   
#     organ_names,   
#     sample_idx,  
#     base_dir=None,   
#     topk=3,   
#     abnormal_threshold=None  
# ):  
#     """  
#     保存预测热图  
#     Args:  
#         predictions: 多尺度预测结果的元组 (low_res_preds, high_res_preds)  
#         targets: 目标标签 (B, 14)  
#         images: 输入图像 (B, 1, D, H, W)  
#         epoch: 当前轮次  
#         organ_names: 器官名称列表  
#         sample_idx: 样本索引  
#         base_dir: 保存目录  
#         topk: top-k取值  
#         abnormal_threshold: 每个疾病的阈值列表  
#     """  
#     # 解包多尺度预测结果  
#     low_res_preds, high_res_preds = predictions  

#     # # 打印原始预测结果的形状  
#     # print("Original Predictions Shapes:")  
#     # print(f"Low-resolution predictions shape: {low_res_preds.shape}")  
#     # print(f"High-resolution predictions shape: {high_res_preds.shape}")  
#     # print(f"Targets shape: {targets.shape}")  
#     # print(f"Images shape: {images.shape}")  
    
#     # 获取目标大小（高分辨率）  
#     _, _, target_depth, target_height, target_width = high_res_preds.shape  
    
#     # 对低分辨率预测进行上采样  
#     low_res_upsampled = upsample_3d(low_res_preds, (target_depth, target_height, target_width))  
    
#     # 选择使用高分辨率预测  
#     high_res_pred = high_res_preds[0].cpu().numpy()  # [16, D, H, W]  
#     low_res_pred = low_res_upsampled[0].cpu().numpy()  # [16, D, H, W]  
    
#     if base_dir is None:  
#         base_dir = Path.cwd()  
#     else:  
#         base_dir = Path(base_dir)  
    
#     save_dir = base_dir / "prediction_heatmaps"  
#     epoch_dir = save_dir / f"epoch_{epoch}"  
#     sample_dir = epoch_dir / str(sample_idx)  
#     sample_dir.mkdir(parents=True, exist_ok=True)  
    
#     target = targets[0].cpu().numpy()  # [14]  
#     original_image = images[0, 0].cpu().numpy()  # [D, H, W]  
    
#     # 保存原始图像  
#     affine = np.eye(4)  
#     original_nifti = nib.Nifti1Image(original_image, affine)  
#     original_nifti.header['descrip'] = f'Original 3D Image, Epoch: {epoch}'  
#     original_save_path = sample_dir / f"original_image.nii.gz"  
#     nib.save(original_nifti, original_save_path)  
    
#     # 16种疾病名称列表  
#     disease_names_16 = [  
#         "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
#         "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
#         "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
#         "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
#         "Bronchiectasis", "Interlobular septal thickening"  
#     ]  

#     # 14种疾病名称列表  
#     disease_names_14 = [  
#         "Cardiomegaly", "Pericardial effusion", "Hiatal hernia",   
#         "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
#         "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
#         "Peribronchial thickening", "Consolidation", "Bronchiectasis",   
#         "Interlobular septal thickening"  
#     ]  

#     # 创建16到14的映射  
#     mapping_16_to_14 = []  
#     for disease in disease_names_16:  
#         try:  
#             idx = disease_names_14.index(disease)  
#             mapping_16_to_14.append(idx)  
#         except ValueError:  
#             mapping_16_to_14.append(-1)  
    
#     # 疾病-器官映射关系  
#     disease_organ_mapping = {  
#         "Cardiomegaly": ["heart"],  
#         "Pericardial effusion": ["heart"],  
#         "Hiatal hernia": ["esophagus"],  
#         "Lymphadenopathy": ["mediastinum"],  
#         "Emphysema": ["lung"],  
#         "Atelectasis": ["lung"],  
#         "Lung nodule": ["lung"],  
#         "Lung opacity": ["lung"],  
#         "Pulmonary fibrotic sequela": ["lung"],  
#         "Pleural effusion": ["pleura"],  
#         "Peribronchial thickening": ["trachea and bronchie"],  
#         "Consolidation": ["lung"],  
#         "Bronchiectasis": ["trachea and bronchie"],  
#         "Interlobular septal thickening": ["lung"]  
#     }  
    
#     # 创建预测信息文件  
#     info_file = sample_dir / "prediction_info.txt"  
#     with open(info_file, "w") as f:  
#         f.write(f"Epoch: {epoch}\n")  
#         f.write(f"Sample: {sample_idx}\n")  
#         f.write(f"Abnormal Detection Parameters: top-{topk}\n")  
#         f.write("Disease Predictions:\n")  
        
#         # 存储每个尺度的预测结果  
#         prediction_results = {  
#             "High-Res": high_res_pred,  
#             "Low-Res": low_res_pred  
#         }  
        
#         # 遍历不同分辨率的预测  
#         for res_name, pred in prediction_results.items():  
#             f.write(f"\n{res_name} Predictions:\n")  
            
#             # 对每种疾病进行处理  
#             for disease_idx, disease_name in enumerate(disease_names_16):  
#                 # 只处理14种疾病中的疾病  
#                 mapping_idx = mapping_16_to_14[disease_idx]  
#                 if mapping_idx == -1:  
#                     continue  

#                 # 3D预测图处理  
#                 disease_pred = pred[disease_idx]  # 当前疾病的3D预测图 [D, H, W]  
#                 disease_label = int(target[mapping_idx])  # 当前疾病的真实标签  
                
#                 # 计算前k个最大值并计算平均值  
#                 topk_mean = np.mean(np.sort(disease_pred.flatten())[-topk:])  
                
#                 # 使用对应疾病的最佳阈值  
#                 current_threshold = abnormal_threshold[mapping_idx]  
#                 pred_abnormal = int(topk_mean > current_threshold)  
                
#                 # 获取相关器官  
#                 related_organs = disease_organ_mapping[disease_name]  
                
#                 # 写入预测信息  
#                 f.write(f"\n  {disease_name}:\n")  
#                 f.write(f"    Ground Truth: {disease_label}\n")  
#                 f.write(f"    Prediction: {pred_abnormal}\n")  
#                 f.write(f"    Top-{topk} Mean: {topk_mean:.4f}\n")  
#                 f.write(f"    Threshold: {current_threshold:.4f}\n")  
#                 f.write(f"    Related Organs: {', '.join(related_organs)}\n")  
                
#                 # 创建包含更多信息的文件名  
#                 result_str = f"GT{disease_label}_PD{pred_abnormal}"  
                
#                 # 保存3D预测结果为NIfTI格式   
#                 nifti_img = nib.Nifti1Image(disease_pred, affine)  
#                 nifti_img.header['descrip'] = (f'{res_name} 3D Disease: {disease_name}, Epoch: {epoch}, '  
#                                              f'Sample: {sample_idx}, '  
#                                              f'GT Label: {disease_label}, Pred: {pred_abnormal}, '  
#                                              f'Top-{topk} Mean: {topk_mean:.4f}, '  
#                                              f'Threshold: {current_threshold:.4f}, '  
#                                              f'Related Organs: {", ".join(related_organs)}')  
#                 save_path = sample_dir / f"{disease_name}_{result_str}_{res_name.lower()}_pred.nii.gz"  
#                 nib.save(nifti_img, save_path)  

#     return save_dir
def save_prediction_heatmaps(  
    predictions,   
    segmentation_preds,  # 新增参数：分割预测结果
    targets,   
    images,   
    epoch,   
    organ_names,   
    sample_idx,  
    base_dir=None,   
    seg_threshold=0.5,  # 新增参数：分割阈值
    topk=3,   
    abnormal_threshold=None  
):  
    """  
    保存预测热图  
    Args:  
        predictions: 多尺度预测结果的元组 (low_res_preds, high_res_preds)  
        segmentation_preds: 器官分割预测结果 (B, 6, D, H, W)
        targets: 目标标签 (B, 14)  
        images: 输入图像 (B, 1, D, H, W)  
        epoch: 当前轮次  
        organ_names: 器官名称列表  
        sample_idx: 样本索引  
        base_dir: 保存目录  
        seg_threshold: 分割掩码阈值
        topk: top-k取值  
        abnormal_threshold: 每个疾病的阈值列表  
    """  
    # 解包多尺度预测结果  
    low_res_preds, high_res_preds = predictions  

    # # 打印原始预测结果的形状  
    # print("Original Predictions Shapes:")  
    # print(f"Low-resolution predictions shape: {low_res_preds.shape}")  
    # print(f"High-resolution predictions shape: {high_res_preds.shape}")  
    # print(f"Segmentation predictions shape: {segmentation_preds.shape}")  
    # print(f"Targets shape: {targets.shape}")  
    # print(f"Images shape: {images.shape}")  
    
    # 获取目标大小（高分辨率）  
    _, _, target_depth, target_height, target_width = high_res_preds.shape  
    
    # 对低分辨率预测进行上采样  
    low_res_upsampled = upsample_3d(low_res_preds, (target_depth, target_height, target_width))  
    
    # 选择使用高分辨率预测  
    high_res_pred = high_res_preds[0].cpu().numpy()  # [16, D, H, W]  
    low_res_pred = low_res_upsampled[0].cpu().numpy()  # [16, D, H, W]  
    
    if base_dir is None:  
        base_dir = Path.cwd()  
    else:  
        base_dir = Path(base_dir)  
    
    save_dir = base_dir / "prediction_heatmaps"  
    epoch_dir = save_dir / f"epoch_{epoch}"  
    sample_dir = epoch_dir / str(sample_idx)  
    sample_dir.mkdir(parents=True, exist_ok=True)  
    
    # 处理分割预测结果
    seg_pred = segmentation_preds[0].cpu().numpy()  # [6, D, H, W]  
    
    # 生成分割掩码
    seg_mask = (seg_pred > seg_threshold).astype(np.float32)
    
    target = targets[0].cpu().numpy()  # [14]  
    original_image = images[0, 0].cpu().numpy()  # [D, H, W]  
    
    # 保存原始图像  
    affine = np.eye(4)  
    original_nifti = nib.Nifti1Image(original_image, affine)  
    original_nifti.header['descrip'] = f'Original 3D Image, Epoch: {epoch}'  
    original_save_path = sample_dir / f"original_image.nii.gz"  
    nib.save(original_nifti, original_save_path)  
    
    # 16种疾病名称列表  
    disease_names_16 = [  
        "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
        "Bronchiectasis", "Interlobular septal thickening"  
    ]  

    # 14种疾病名称列表  
    disease_names_14 = [  
        "Cardiomegaly", "Pericardial effusion", "Hiatal hernia",   
        "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Peribronchial thickening", "Consolidation", "Bronchiectasis",   
        "Interlobular septal thickening"  
    ]  

    # 创建16到14的映射  
    mapping_16_to_14 = []  
    for disease in disease_names_16:  
        try:  
            idx = disease_names_14.index(disease)  
            mapping_16_to_14.append(idx)  
        except ValueError:  
            mapping_16_to_14.append(-1)  
    
    # 疾病-器官映射关系  
    disease_organ_mapping = {  
        "Cardiomegaly": ["heart"],  
        "Pericardial effusion": ["heart"],  
        "Coronary artery wall calcification": ["heart"],  
        "Hiatal hernia": ["esophagus"],  
        "Lymphadenopathy": ["mediastinum"],  
        "Emphysema": ["lung"],  
        "Atelectasis": ["lung"],  
        "Lung nodule": ["lung"],  
        "Lung opacity": ["lung"],  
        "Pulmonary fibrotic sequela": ["lung"],  
        "Pleural effusion": ["pleura"],  
        "Mosaic attenuation pattern": ["lung"],  
        "Peribronchial thickening": ["trachea and bronchie"],  
        "Consolidation": ["lung"],  
        "Bronchiectasis": ["trachea and bronchie"],  
        "Interlobular septal thickening": ["lung"]  
    }  
    
    # 创建预测信息文件  
    info_file = sample_dir / "prediction_info.txt"  
    with open(info_file, "w") as f:  
        f.write(f"Epoch: {epoch}\n")  
        f.write(f"Sample: {sample_idx}\n")  
        f.write(f"Abnormal Detection Parameters: top-{topk}\n")  
        f.write("Disease Predictions:\n")  
        
        # 存储每个尺度的预测结果  
        prediction_results = {  
            "High-Res": high_res_pred,  
            "Low-Res": low_res_pred  
        }  
        
        # 遍历不同分辨率的预测  
        for res_name, pred in prediction_results.items():  
            f.write(f"\n{res_name} Predictions:\n")  
            
            # 对每种疾病进行处理  
            for disease_idx, disease_name in enumerate(disease_names_16):  
                # 只处理14种疾病中的疾病  
                mapping_idx = mapping_16_to_14[disease_idx]  
                if mapping_idx == -1:  
                    continue  

                # 3D预测图处理  
                disease_pred = pred[disease_idx]  # 当前疾病的3D预测图 [D, H, W]  
                disease_label = int(target[mapping_idx])  # 当前疾病的真实标签  
                
                # 获取相关器官的分割掩码
                related_organs = disease_organ_mapping[disease_name]
                combined_mask = np.zeros_like(seg_mask[0])
                for organ_name in related_organs:
                    organ_idx = organ_names.index(organ_name)
                    combined_mask = np.maximum(combined_mask, seg_mask[organ_idx])
                
                # 计算combined_pred（将预测与器官掩码相乘）
                combined_pred = disease_pred * combined_mask
                
                # 计算前k个最大值并计算平均值（使用组合预测结果）
                topk_mean = np.mean(np.sort(combined_pred.flatten())[-topk:])  
                
                # 使用对应疾病的最佳阈值  
                current_threshold = abnormal_threshold[mapping_idx]  
                pred_abnormal = int(topk_mean > current_threshold)  
                
                # 写入预测信息  
                f.write(f"\n  {disease_name}:\n")  
                f.write(f"    Ground Truth: {disease_label}\n")  
                f.write(f"    Prediction: {pred_abnormal}\n")  
                f.write(f"    Top-{topk} Mean: {topk_mean:.4f}\n")  
                f.write(f"    Threshold: {current_threshold:.4f}\n")  
                f.write(f"    Related Organs: {', '.join(related_organs)}\n")  
                
                # 创建包含更多信息的文件名  
                result_str = f"GT{disease_label}_PD{pred_abnormal}"  
                
                # 保存原始疾病预测热图
                nifti_img = nib.Nifti1Image(disease_pred, affine)  
                nifti_img.header['descrip'] = (f'{res_name} 3D Disease: {disease_name}, Epoch: {epoch}, '  
                                             f'Sample: {sample_idx}, '  
                                             f'GT Label: {disease_label}, Pred: {pred_abnormal}, '  
                                             f'Top-{topk} Mean: {topk_mean:.4f}, '  
                                             f'Threshold: {current_threshold:.4f}, '  
                                             f'Related Organs: {", ".join(related_organs)}')  
                save_path = sample_dir / f"{disease_name}_{result_str}_{res_name.lower()}_pred.nii.gz"  
                nib.save(nifti_img, save_path)
                
                # 保存结合器官分割的疾病预测热图
                combined_nifti = nib.Nifti1Image(combined_pred, affine)
                combined_nifti.header['descrip'] = (f'{res_name} 3D Disease: {disease_name}, Epoch: {epoch}, '  
                                             f'Sample: {sample_idx}, '  
                                             f'GT Label: {disease_label}, Pred: {pred_abnormal}, '  
                                             f'Top-{topk} Mean: {topk_mean:.4f}, '  
                                             f'Threshold: {current_threshold:.4f}, '  
                                             f'Related Organs: {", ".join(related_organs)}')
                combined_save_path = sample_dir / f"{disease_name}_{result_str}_{res_name.lower()}_combined_pred.nii.gz"
                nib.save(combined_nifti, combined_save_path)
                
                # 保存器官掩码
                mask_nifti = nib.Nifti1Image(combined_mask, affine)
                mask_nifti.header['descrip'] = (f'Organ Mask for {disease_name}, '
                                              f'Related Organs: {", ".join(related_organs)}')
                mask_save_path = sample_dir / f"{disease_name}_organ_mask.nii.gz"
                nib.save(mask_nifti, mask_save_path)

    return save_dir
    

def Abnormal_loss(seg_pred, abnormal_pred, abnormal_targets, disease_frequencies, k=3, epsilon=1e-6, seg_threshold=0.5):  
    """  
    计算异常检测损失  
    Args:  
        seg_pred: 分割预测 (B, 6, D, H, W)  
        abnormal_pred: 疾病预测 (B, 16, D, H, W)  
        abnormal_targets: 疾病标签 (B, 14)  # 修改为14种疾病  
        disease_frequencies: 16种疾病的阳性样本频率列表  
        k: top-k取值  
        epsilon: 数值稳定性常数  
        seg_threshold: 分割掩码阈值  
    """  
    B, _, D, H, W = seg_pred.shape  
    num_diseases = abnormal_pred.shape[1]  # 16个疾病通道  
    
    # 定义16种疾病到14种疾病的映射  
    mapping_16_to_14 = []  
    disease_names_16 = [  
        "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
        "Bronchiectasis", "Interlobular septal thickening"  
    ]  
    disease_names_14 = [  
        "Cardiomegaly", "Pericardial effusion", "Hiatal hernia",   
        "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Peribronchial thickening", "Consolidation", "Bronchiectasis",   
        "Interlobular septal thickening"  
    ]  
    
    for disease in disease_names_16:  
        try:  
            idx = disease_names_14.index(disease)  
            mapping_16_to_14.append(idx)  
        except ValueError:  
            mapping_16_to_14.append(-1)  
    
    # 获取疾病到器官的映射  
    disease_to_organ = get_organ_disease_mapping()  
    
    # 对分割预测进行二值化  
    seg_mask = (seg_pred > seg_threshold).float()  
    
    disease_losses = []  
    disease_predictions = []  
    
    for disease_idx in range(num_diseases):  
        # 跳过不在14种疾病中的疾病  
        if mapping_16_to_14[disease_idx] == -1:  
            disease_losses.append(torch.tensor(0.0, device=abnormal_pred.device))  
            disease_predictions.append(torch.zeros(B, device=abnormal_pred.device))  
            continue  

        # 获取对应的器官索引  
        organ_idx = disease_to_organ[disease_idx]  
        
        # 将对应器官的分割掩码与疾病预测相乘  
        organ_mask = seg_mask[:, organ_idx:organ_idx+1]  
        disease_pred = abnormal_pred[:, disease_idx:disease_idx+1]  
        
        final_pred = organ_mask * disease_pred  
        
        # 取top-k值的平均  
        top_k_values, _ = torch.topk(final_pred.view(B, -1), k, dim=1)  
        avg_top_k = top_k_values.mean(dim=1)  
        
        # 使用疾病的频率计算权重  
        freq = disease_frequencies[disease_idx]  
        if freq > 0:  # 只对存在的疾病计算权重  
            weight_pos = (1 - freq + epsilon) / (freq + epsilon)  
            weight_neg = 1.0  
            
            # 计算损失  
            target = abnormal_targets[:, mapping_16_to_14[disease_idx]]  # 使用映射后的索引  
            loss = weighted_binary_cross_entropy(avg_top_k, target, weight_pos, weight_neg)  
            loss = loss.mean()  
        else:  # 对不存在的疾病，设置损失为0  
            loss = torch.tensor(0.0, device=abnormal_pred.device)  
        
        disease_losses.append(loss)  
        disease_predictions.append(avg_top_k)  
    
    return disease_losses, torch.stack(disease_predictions, dim=1)


def plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, organ_names, epoch):  
    plt.figure(figsize=(10, 8))  
    colors = plt.cm.tab10(np.linspace(0, 1, len(organ_names)))  
    
    for organ, color in zip(organ_names, colors):  
        plt.plot(fpr_dict[organ], tpr_dict[organ], color=color,  
                label=f'{organ} (AUC = {roc_auc_dict[organ]:.3f})')  
    
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlim([0.0, 1.0])  
    plt.ylim([0.0, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title(f'ROC Curves for Different Organs')  
    plt.legend(loc="lower right")  
    
    # 保存图片  
    save_path = f'roc_curves_epoch_{epoch}.png'  
    plt.savefig(save_path)  
    plt.close()  
    return save_path  



def plot_confusion_matrix(cm, classes, title):  
    """绘制混淆矩阵"""  
    fig, ax = plt.subplots()  
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  
    ax.figure.colorbar(im, ax=ax)  
    ax.set(xticks=np.arange(cm.shape[1]),  
           yticks=np.arange(cm.shape[0]),  
           xticklabels=classes, yticklabels=classes,  
           title=title,  
           ylabel='True label',  
           xlabel='Predicted label')  
    
    # 在格子中添加数字  
    fmt = 'd'  
    thresh = cm.max() / 2.  
    for i in range(cm.shape[0]):  
        for j in range(cm.shape[1]):  
            ax.text(j, i, format(cm[i, j], fmt),  
                   ha="center", va="center",  
                   color="white" if cm[i, j] > thresh else "black")  
    fig.tight_layout()  
    return fig  

def calculate_organ_metrics(predictions, targets):  
    """计算器官级别的指标"""  
    predictions = np.array(predictions)  
    targets = np.array(targets)  
    
    auroc = roc_auc_score(targets, predictions)  
    predictions_binary = (predictions > 0.5).astype(int)  
    
    tn, fp, fn, tp = confusion_matrix(targets, predictions_binary).ravel()  
    precision = tp / (tp + fp + 1e-8)  
    recall = tp / (tp + fn + 1e-8)  
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)  
    f1 = 2 * precision * recall / (precision + recall + 1e-8)  
    
    return {  
        'auroc': auroc,  
        'precision': precision,  
        'recall': recall,  
        'accuracy': accuracy,  
        'f1': f1  
    }

def create_test_log_info(epoch, metrics_per_disease, organ_metrics, disease_names, organ_names, best_thresholds):  
    """  
    创建测试日志信息  
    
    参数:  
    epoch : int  
        当前轮次  
    metrics_per_disease : list of dict  
        每种疾病的指标  
    organ_metrics : dict  
        每个器官的指标  
    disease_names : list  
        疾病名称列表  
    organ_names : list  
        器官名称列表  
    best_thresholds : numpy array  
        每种疾病的最佳阈值  
    
    返回:  
    str : 格式化的日志信息  
    """  
    import numpy as np  
    
    # 计算疾病级别的平均指标  
    avg_metrics = {  
        'precision': np.mean([m['precision'] for m in metrics_per_disease]),  
        'recall': np.mean([m['recall'] for m in metrics_per_disease]),  
        'f1': np.mean([m['f1'] for m in metrics_per_disease]),  
        'accuracy': np.mean([m['accuracy'] for m in metrics_per_disease]),  
        'auroc': np.mean([m['auroc'] for m in metrics_per_disease])  
    }  
    
    # 计算器官级别的平均指标  
    avg_organ_metrics = {  
        'precision': np.mean([m['precision'] for m in organ_metrics.values()]),  
        'recall': np.mean([m['recall'] for m in organ_metrics.values()]),  
        'f1': np.mean([m['f1'] for m in organ_metrics.values()]),  
        'accuracy': np.mean([m['accuracy'] for m in organ_metrics.values()]),  
        'auroc': np.mean([m['auroc'] for m in organ_metrics.values()])  
    }  
    
    # 构建日志信息  
    log_info = f"\nTest Epoch {epoch} Summary:\n"  
    log_info += "=" * 50 + "\n"  
    
    # 总体疾病指标  
    log_info += "\nOverall Disease Metrics:\n"  
    log_info += "-" * 30 + "\n"  
    log_info += f"Average Precision: {avg_metrics['precision']:.4f}\n"  
    log_info += f"Average Recall: {avg_metrics['recall']:.4f}\n"  
    log_info += f"Average F1-Score: {avg_metrics['f1']:.4f}\n"  
    log_info += f"Average Accuracy: {avg_metrics['accuracy']:.4f}\n"  
    log_info += f"Average AUROC: {avg_metrics['auroc']:.4f}\n"  
    
    # 总体器官指标  
    log_info += "\nOverall Organ Metrics:\n"  
    log_info += "-" * 30 + "\n"  
    log_info += f"Average Precision: {avg_organ_metrics['precision']:.4f}\n"  
    log_info += f"Average Recall: {avg_organ_metrics['recall']:.4f}\n"  
    log_info += f"Average F1-Score: {avg_organ_metrics['f1']:.4f}\n"  
    log_info += f"Average Accuracy: {avg_organ_metrics['accuracy']:.4f}\n"  
    log_info += f"Average AUROC: {avg_organ_metrics['auroc']:.4f}\n"  
    
    # 每个疾病的详细指标  
    log_info += "\nPer-Disease Metrics:\n"  
    log_info += "-" * 30 + "\n"  
    for i, disease in enumerate(disease_names):  
        log_info += f"\n{disease}:\n"  
        log_info += f"Threshold: {best_thresholds[i]:.3f}\n"  
        log_info += f"Precision: {metrics_per_disease[i]['precision']:.4f}\n"  
        log_info += f"Recall: {metrics_per_disease[i]['recall']:.4f}\n"  
        log_info += f"F1-Score: {metrics_per_disease[i]['f1']:.4f}\n"  
        log_info += f"Accuracy: {metrics_per_disease[i]['accuracy']:.4f}\n"  
        log_info += f"AUROC: {metrics_per_disease[i]['auroc']:.4f}\n"  
    
    # 每个器官的详细指标  
    log_info += "\nPer-Organ Metrics:\n"  
    log_info += "-" * 30 + "\n"  
    for organ in organ_names:  
        if organ in organ_metrics:  
            log_info += f"\n{organ.capitalize()}:\n"  
            log_info += f"Precision: {organ_metrics[organ]['precision']:.4f}\n"  
            log_info += f"Recall: {organ_metrics[organ]['recall']:.4f}\n"  
            log_info += f"F1-Score: {organ_metrics[organ]['f1']:.4f}\n"  
            log_info += f"Accuracy: {organ_metrics[organ]['accuracy']:.4f}\n"  
            log_info += f"AUROC: {organ_metrics[organ]['auroc']:.4f}\n"  
    
    log_info += "\n" + "=" * 50 + "\n"  
    
    return log_info

def plot_confusion_matrix(y_true, y_pred, disease_name, save_dir):  
    """  
    绘制并保存混淆矩阵  
    """  
    cm = confusion_matrix(y_true, y_pred)  
    plt.figure(figsize=(8, 6))  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
    plt.title(f'Confusion Matrix - {disease_name}')  
    plt.ylabel('True Label')  
    plt.xlabel('Predicted Label')  
    
    save_path = os.path.join(save_dir, f'confusion_matrix_{disease_name.replace(" ", "_")}.png')  
    plt.savefig(save_path)  
    plt.close()  
    return save_path, cm 



def valid_one_epoch(valid_loader, model, segmentation_criterion, abnormal_criterion, epoch, logger, config, writer, device, save_heatmap=True):
    model.eval()
    loss_list = []
    seg_loss_list = []
    abnormal_loss_list = [[] for _ in range(16)]
    sample_idx = 0
    np.random.seed(42)
    
    # 添加阈值搜索相关的变量
    threshold_candidates = np.arange(0.1, 0.9, 0.05)  # 从0.1到0.9，步长0.05
    best_thresholds = np.array([0.5] * 16)  # 初始化每个疾病的最佳阈值
    best_f1_scores = np.array([-1] * 16)  # 记录每个疾病的最佳F1分数
    
    dataset_size = len(valid_loader.dataset)
    selected_indices = set(np.random.choice(dataset_size, min(40, dataset_size), replace=False))
    current_dir = config.work_dir

    if save_heatmap:
        save_dir = os.path.join(config.work_dir, "validation_results")
        os.makedirs(save_dir, exist_ok=True)
        
        # 用于存储需要生成热图的数据
        heatmap_data = []

    current_seg_weight = calculate_dynamic_weight(
        epoch=epoch,
        initial_weight=config.initial_segmentation_weight,
        total_epochs=config.epochs,
        decay_rate=config.weight_decay_rate
    )
    
    logger.info(f"Validation epoch {epoch}: Segmentation weight = {current_seg_weight:.4f}, "
                f"Abnormal weight = {config.abnormal_loss_weight}")

    valid_loader = tqdm(valid_loader, desc=f"Epoch {epoch} Validation", leave=True, dynamic_ncols=True)

    dice_scores = []
    tp_sum = np.zeros(16)
    tn_sum = np.zeros(16)
    fp_sum = np.zeros(16)
    fn_sum = np.zeros(16)

    # 修改预测结果的收集列表
    processed_predictions = [[] for _ in range(16)]  # 存储处理后的预测结果
    all_targets = [[] for _ in range(16)]

    organ_names = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", "esophagus"]  

    disease_names = [  
        "Cardiomegaly",                      # 0  
        "Pericardial effusion",              # 1  
        "Coronary artery wall calcification",# 2  
        "Hiatal hernia",                     # 3  
        "Lymphadenopathy",                   # 4  
        "Emphysema",                         # 5  
        "Atelectasis",                       # 6  
        "Lung nodule",                       # 7  
        "Lung opacity",                      # 8  
        "Pulmonary fibrotic sequela",        # 9  
        "Pleural effusion",                  # 10  
        "Mosaic attenuation pattern",        # 11  
        "Peribronchial thickening",          # 12  
        "Consolidation",                     # 13  
        "Bronchiectasis",                    # 14  
        "Interlobular septal thickening"     # 15  
    ]  

    # 修订后的器官-疾病映射  
    organ_disease_mapping = {  
        "lung": [5, 6, 7, 8, 9, 11, 13, 15],  # Emphysema, Atelectasis, Lung nodule, Lung opacity, Pulmonary fibrotic sequela, Mosaic attenuation pattern, Consolidation, Interlobular septal thickening  
        "heart": [0, 1, 2],  # Cardiomegaly, Pericardial effusion, Coronary artery wall calcification  
        "pleura": [10],  # Pleural effusion  
        "mediastinum": [4],  # Lymphadenopathy  
        "esophagus": [3],  # Hiatal hernia  
        "trachea and bronchie": [12, 14]  # Peribronchial thickening, Bronchiectasis  
    }

    with torch.no_grad():
        for iter, data in enumerate(valid_loader):
            images, seg_targets, abnormal_targets, sample_names = data
            images, seg_targets, abnormal_targets = images.to(device), seg_targets.to(device), abnormal_targets.to(device)

            seg_pred, abnormal_preds = model(images)
            abnormal_pred = abnormal_preds[-1]  # 取最后一个尺度的输出 
            
            disease_frequencies = [
                0.1072, 0.0705, 0.2476, 0.1420, 0.2534, 0.1939, 0.2558, 0.4548,
                0.3666, 0.2672, 0.1185, 0.0744, 0.1034, 0.1755, 0.0999, 0.0788
            ]

            seg_loss = segmentation_criterion(seg_pred, seg_targets)
            disease_losses, abnormal_pred_avg = Abnormal_loss(seg_pred, abnormal_pred, abnormal_targets, disease_frequencies)
            
            # 收集处理后的预测结果和真实标签
            abnormal_targets_np = abnormal_targets.cpu().numpy()
            abnormal_pred_avg_np = abnormal_pred_avg.cpu().numpy()
            abnormal_targets_np=1-abnormal_targets_np#将阴性作为1
            abnormal_pred_avg_np=1-abnormal_pred_avg_np


            for i in range(16):
                processed_predictions[i].extend(abnormal_pred_avg_np[:, i])
                all_targets[i].extend(abnormal_targets_np[:, i])

            for i, disease_loss in enumerate(disease_losses):
                abnormal_loss_list[i].append(disease_loss.item())

            abnormal_loss = torch.mean(torch.stack(disease_losses))
            total_loss = current_seg_weight * seg_loss + config.abnormal_loss_weight * abnormal_loss

            loss_list.append(total_loss.item())
            seg_loss_list.append(seg_loss.item())

            dice_score = calculate_dice(seg_pred, seg_targets)
            dice_scores.append(dice_score)

            # 如果需要保存热图，先存储相关数据
            if save_heatmap and iter in selected_indices:
                for batch_idx in range(images.size(0)):
                    heatmap_data.append({
                        'predictions': [  
                            abnormal_preds[0][batch_idx:batch_idx+1].cpu(),  # 低分辨率预测  
                            abnormal_preds[1][batch_idx:batch_idx+1].cpu()   # 高分辨率预测  
                        ], 
                        'segmentation_preds': seg_pred[batch_idx:batch_idx+1].cpu(),
                        'targets': abnormal_targets[batch_idx:batch_idx+1].cpu(),
                        'images': images[batch_idx:batch_idx+1].cpu(),
                        'sample_name': sample_names[batch_idx]
                    })

    # 将原来的阈值搜索部分替换为以下代码  
    logger.info("Finding optimal thresholds for each disease using Youden's index...")  
    for disease_idx in range(16):  
        y_true = np.array(all_targets[disease_idx]) 
        # y_true = 1 - y_true 
        y_pred_proba = np.array(processed_predictions[disease_idx])  
        # y_pred_proba=1-y_pred_proba
        
        # 使用ROC曲线和Youden指数寻找最优阈值  
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)  
        
        # 计算Youden指数 (J = 敏感性 + 特异性 - 1 = TPR - FPR)  
        youden_index = tpr - fpr  
        
        # 找到最大Youden指数对应的阈值  
        optimal_idx = np.argmax(youden_index)  
        best_threshold = thresholds[optimal_idx]  

        # 打印最佳阈值对应的TPR、FPR和阈值  
        print(f"Disease: {disease_names[disease_idx]}")  
        print(f"Optimal Threshold: {best_threshold:.3f}")  
        print(f"TPR (Sensitivity): {tpr[optimal_idx]:.4f}")  
        print(f"FPR: {fpr[optimal_idx]:.4f}")  
        
        # 使用最优阈值计算预测结果  
        y_pred = (y_pred_proba > best_threshold).astype(int)  
        # y_pred = 1 - y_pred
        
        # 计算在最优阈值下的性能指标  
        tp = np.sum((y_true == 1) & (y_pred == 1))  
        fp = np.sum((y_true == 0) & (y_pred == 1))  
        fn = np.sum((y_true == 1) & (y_pred == 0))  
        
        # 计算F1分数  
        precision = tp / (tp + fp + 1e-8)  
        recall = tp / (tp + fn + 1e-8)  
        f1 = 2 * precision * recall / (precision + recall + 1e-8)  
        
        best_thresholds[disease_idx] = best_threshold  
        best_f1_scores[disease_idx] = f1  
        
        # 记录最优阈值和对应的性能指标  
        logger.info(f"{disease_names[disease_idx]}: "  
                    f"Best threshold = {best_threshold:.3f}, "  
                    f"F1 = {f1:.4f}, "  
                    f"Sensitivity = {tpr[optimal_idx]:.4f}, "  
                    f"Specificity = {1-fpr[optimal_idx]:.4f}, "  
                    f"Youden index = {youden_index[optimal_idx]:.4f}")  
    # logger.info("Finding optimal thresholds for each disease using F1 score...")  
    # for disease_idx in range(16):  
    #     y_true = np.array(all_targets[disease_idx])  
    #     y_true = 1 - y_true
    #     y_pred_proba = np.array(processed_predictions[disease_idx])  
        
    #     # 创建一个更精细的阈值范围  
    #     thresholds = np.linspace(0, 1, 1001)  # 0到1之间的1001个等距点  
        
    #     # 存储每个阈值对应的F1分数  
    #     f1_scores = []  
        
    #     # 遍历所有阈值  
    #     for threshold in thresholds:  
    #         # 使用当前阈值进行预测  
    #         y_pred = (y_pred_proba > threshold).astype(int)  
    #         y_pred = 1 - y_pred
            
    #         # 计算性能指标  
    #         tp = np.sum((y_true == 1) & (y_pred == 1))  # 真阳性  
    #         fp = np.sum((y_true == 0) & (y_pred == 1))  # 假阳性  
    #         fn = np.sum((y_true == 1) & (y_pred == 0))  # 假阴性  
            
    #         # 计算F1分数  
    #         precision = tp / (tp + fp + 1e-8)  
    #         recall = tp / (tp + fn + 1e-8)  
    #         f1 = 2 * precision * recall / (precision + recall + 1e-8)  
            
    #         f1_scores.append(f1)  
        
    #     # 找到最大F1分数对应的阈值  
    #     optimal_idx = np.argmax(f1_scores)  
    #     best_threshold = thresholds[optimal_idx]  
    #     best_f1 = f1_scores[optimal_idx]  
        
    #     # 使用最优阈值重新计算混淆矩阵指标  
    #     y_pred = (y_pred_proba > best_threshold).astype(int)  
    #     y_pred = 1 - y_pred
        
    #     tp = np.sum((y_true == 1) & (y_pred == 1))  
    #     fp = np.sum((y_true == 0) & (y_pred == 1))  
    #     fn = np.sum((y_true == 1) & (y_pred == 0))  
    #     tn = np.sum((y_true == 0) & (y_pred == 0))  
        
    #     precision = tp / (tp + fp + 1e-8)  
    #     recall = tp / (tp + fn + 1e-8)  
    #     f1 = 2 * precision * recall / (precision + recall + 1e-8)  
        
    #     # 计算其他性能指标  
    #     sensitivity = tp / (tp + fn + 1e-8)  
    #     specificity = tn / (tn + fp + 1e-8)  
        
    #     # 存储结果  
    #     best_thresholds[disease_idx] = best_threshold  
    #     best_f1_scores[disease_idx] = best_f1  
        
    #     # 记录详细信息  
    #     logger.info(f"{disease_names[disease_idx]}: "  
    #                 f"Best threshold = {best_threshold:.4f}, "  
    #                 f"F1 = {f1:.4f}, "  
    #                 f"Precision = {precision:.4f}, "  
    #                 f"Recall = {recall:.4f}, "  
    #                 f"Sensitivity = {sensitivity:.4f}, "  
    #                 f"Specificity = {specificity:.4f}")

    # 使用最佳阈值重新计算所有指标
    tp_sum = np.zeros(16)
    tn_sum = np.zeros(16)
    fp_sum = np.zeros(16)
    fn_sum = np.zeros(16)
    
    for disease_idx in range(16):
        y_true = np.array(all_targets[disease_idx])
        # y_true = 1 - y_true
        y_pred_proba = np.array(processed_predictions[disease_idx])
        # y_pred_proba=1-y_pred_proba
        y_pred = (y_pred_proba > best_thresholds[disease_idx]).astype(int)
        # y_pred = 1 - y_pred
        
        tp_sum[disease_idx] = np.sum((y_true == 1) & (y_pred == 1))
        tn_sum[disease_idx] = np.sum((y_true == 0) & (y_pred == 0))
        fp_sum[disease_idx] = np.sum((y_true == 0) & (y_pred == 1))
        fn_sum[disease_idx] = np.sum((y_true == 1) & (y_pred == 0))

    # 保存最佳阈值
    threshold_save_path = os.path.join(config.work_dir, f'best_thresholds_epoch_{epoch}.npy')
    np.save(threshold_save_path, best_thresholds)

    # 计算每个疾病的指标
    precision_per_disease = tp_sum / (tp_sum + fp_sum + 1e-8)
    recall_per_disease = tp_sum / (tp_sum + fn_sum + 1e-8)
    accuracy_per_disease = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum + 1e-8)
    # f1_per_disease = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum + 1e-8)
    f1_per_disease = 2 * precision_per_disease * recall_per_disease / (precision_per_disease + recall_per_disease + 1e-8)  

    # 计算每个疾病的 AUROC
    auroc_per_disease = []
    for i in range(16):
        try:
            auroc = roc_auc_score(all_targets[i], processed_predictions[i])
        except ValueError:
            auroc = 0.0
        auroc_per_disease.append(auroc)

    # 计算每个器官的指标
    organ_metrics = {}
    for organ, disease_indices in organ_disease_mapping.items():
        # 收集该器官所有疾病的预测和真实值
        organ_predictions = []
        organ_targets = []
        for disease_idx in disease_indices:
            organ_predictions.extend(processed_predictions[disease_idx])
            organ_targets.extend(all_targets[disease_idx])
        
        # 计算该器官的AUROC
        try:
            organ_auroc = roc_auc_score(organ_targets, organ_predictions)
        except ValueError:
            organ_auroc = 0.0
            
        # 使用最佳阈值计算该器官的其他指标
        organ_tp = sum(tp_sum[i] for i in disease_indices)
        organ_tn = sum(tn_sum[i] for i in disease_indices)
        organ_fp = sum(fp_sum[i] for i in disease_indices)
        organ_fn = sum(fn_sum[i] for i in disease_indices)
        
        # 计算器官级别的指标
        organ_precision = organ_tp / (organ_tp + organ_fp + 1e-8)
        organ_recall = organ_tp / (organ_tp + organ_fn + 1e-8)
        organ_accuracy = (organ_tp + organ_tn) / (organ_tp + organ_tn + organ_fp + organ_fn + 1e-8)
        # organ_f1 = 2 * organ_tp / (2 * organ_tp + organ_fp + organ_fn + 1e-8)
        organ_f1 = 2 * organ_precision * organ_recall / (organ_precision + organ_recall + 1e-8)  
        
        # 存储器官指标
        organ_metrics[organ] = {
            'auroc': organ_auroc,
            'precision': organ_precision,
            'recall': organ_recall,
            'accuracy': organ_accuracy,
            'f1': organ_f1
        }

    # 计算平均指标
    avg_precision = np.mean(precision_per_disease)
    avg_recall = np.mean(recall_per_disease)
    avg_accuracy = np.mean(accuracy_per_disease)
    avg_f1 = np.mean(f1_per_disease)
    avg_abnormal_loss = np.mean([np.mean(losses) if losses else 0.0 for losses in abnormal_loss_list])
    avg_auroc = np.mean(auroc_per_disease)

    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    avg_loss = np.mean(loss_list) if loss_list else 0.0
    avg_seg_loss = np.mean(seg_loss_list) if seg_loss_list else 0.0

    def plot_confusion_matrix(y_true, y_pred, name, save_dir):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        save_path = os.path.join(save_dir, f'confusion_matrix_{name.replace(" ", "_")}.png')
        plt.savefig(save_path)
        plt.close()
        return save_path, cm

    # 创建保存混淆矩阵的目录
    confusion_matrix_dir = os.path.join(config.work_dir, "validation_confusion_matrices", f"epoch_{epoch}")
    os.makedirs(confusion_matrix_dir, exist_ok=True)

    # 计算并保存每个疾病的混淆矩阵
    confusion_matrices = {}
    log_info = f"Validation Epoch {epoch} Summary:\n"
    log_info += "Overall Metrics:\n"
    log_info += f"Total Loss: {avg_loss:.4f}, "
    log_info += f"Seg Loss: {avg_seg_loss:.4f}, "
    log_info += f"Abnormal Loss: {avg_abnormal_loss:.4f}, "
    log_info += f"Seg Dice Score: {avg_dice:.4f}\n"
    log_info += f"Average Disease Metrics: "
    log_info += f"Accuracy: {avg_accuracy:.4f}, "
    log_info += f"Precision: {avg_precision:.4f}, "
    log_info += f"Recall: {avg_recall:.4f}, "
    log_info += f"F1: {avg_f1:.4f}, "
    log_info += f"AUROC: {avg_auroc:.4f}\n"

    log_info += "\nPer-Disease Metrics, Thresholds and Confusion Matrices:"
    for i, disease in enumerate(disease_names):
        y_pred = (np.array(processed_predictions[i]) > best_thresholds[i]).astype(int)  
        # y_pred = 1 - y_pred
        y_true = np.array(all_targets[i])  
        # y_true = 1 - y_true
        
        save_path, cm = plot_confusion_matrix(y_true, y_pred, disease, confusion_matrix_dir)  
        confusion_matrices[disease] = cm  
        
        # 添加到日志信息  
        log_info += f"\n{disease}:\n"  
        log_info += f"Best Threshold: {best_thresholds[i]:.3f}, "  
        log_info += f"Loss: {np.mean(abnormal_loss_list[i]):.4f}, "  
        log_info += f"Accuracy: {accuracy_per_disease[i]:.4f}, "  
        log_info += f"Precision: {precision_per_disease[i]:.4f}, "  
        log_info += f"Recall: {recall_per_disease[i]:.4f}, "  
        log_info += f"F1: {f1_per_disease[i]:.4f}, "  
        log_info += f"AUROC: {auroc_per_disease[i]:.4f}\n"  
        log_info += f"Confusion Matrix:\n"  
        log_info += f"TN: {cm[0,0]}, FP: {cm[0,1]}\n"  
        log_info += f"FN: {cm[1,0]}, TP: {cm[1,1]}\n"  

    # 计算并保存每个器官的混淆矩阵  
    log_info += "\n\nOrgan-level Confusion Matrices:"  
    for organ, disease_indices in organ_disease_mapping.items():  
        organ_predictions = []  
        organ_targets = []  
        for disease_idx in disease_indices:  
            organ_predictions.extend((np.array(processed_predictions[disease_idx]) > best_thresholds[disease_idx]).astype(int))  
            organ_targets.extend(all_targets[disease_idx])  
        
        save_path, cm = plot_confusion_matrix(organ_targets, organ_predictions, organ, confusion_matrix_dir)  
        
        # 添加到日志信息  
        log_info += f"\n{organ}:\n"  
        log_info += f"AUROC: {organ_metrics[organ]['auroc']:.4f}, "  
        log_info += f"Accuracy: {organ_metrics[organ]['accuracy']:.4f}, "  
        log_info += f"Precision: {organ_metrics[organ]['precision']:.4f}, "  
        log_info += f"Recall: {organ_metrics[organ]['recall']:.4f}, "  
        log_info += f"F1: {organ_metrics[organ]['f1']:.4f}\n"  
        log_info += f"Confusion Matrix:\n"  
        log_info += f"TN: {cm[0,0]}, FP: {cm[0,1]}\n"  
        log_info += f"FN: {cm[1,0]}, TP: {cm[1,1]}\n"  

    # 计算总体混淆矩阵统计信息  
    total_tn = sum(cm[0,0] for cm in confusion_matrices.values())  
    total_fp = sum(cm[0,1] for cm in confusion_matrices.values())  
    total_fn = sum(cm[1,0] for cm in confusion_matrices.values())  
    total_tp = sum(cm[1,1] for cm in confusion_matrices.values())  
    
    # 保存总体混淆矩阵  
    total_cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])  
    plt.figure(figsize=(10, 8))  
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues')  
    plt.title('Overall Confusion Matrix')  
    plt.ylabel('True Label')  
    plt.xlabel('Predicted Label')  
    total_cm_path = os.path.join(confusion_matrix_dir, 'total_confusion_matrix.png')  
    plt.savefig(total_cm_path)  
    plt.close()  
    
    log_info += "\n\nOverall Confusion Matrix Statistics:\n"  
    log_info += f"Total True Negative: {total_tn}\n"  
    log_info += f"Total False Positive: {total_fp}\n"  
    log_info += f"Total False Negative: {total_fn}\n"  
    log_info += f"Total True Positive: {total_tp}\n"  
    log_info += f"Total Samples: {total_tn + total_fp + total_fn + total_tp}\n"  

    # 计算总体指标  
    total_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)  
    total_precision = total_tp / (total_tp + total_fp + 1e-8)  
    total_recall = total_tp / (total_tp + total_fn + 1e-8)  
    # total_f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)  
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-8)
    
    log_info += f"\nOverall Metrics from Confusion Matrix:\n"  
    log_info += f"Accuracy: {total_accuracy:.4f}\n"  
    log_info += f"Precision: {total_precision:.4f}\n"  
    log_info += f"Recall: {total_recall:.4f}\n"  
    log_info += f"F1 Score: {total_f1:.4f}\n"  

    # 在找到最佳阈值后，生成和保存热图  
    if save_heatmap and heatmap_data:  
        if epoch % 5 == 0: 
            logger.info("Generating and saving heatmaps with optimal thresholds...")  
            for data in heatmap_data:  
                save_prediction_heatmaps(  
                    predictions=data['predictions'],  
                    segmentation_preds=data['segmentation_preds'],  
                    targets=data['targets'],  
                    images=data['images'],  
                    epoch=epoch,  
                    organ_names=organ_names,  
                    sample_idx=data['sample_name'],  
                    base_dir=save_dir,  
                    seg_threshold=0.5,  
                    topk=3,  
                    abnormal_threshold=best_thresholds  
                )  
                logger.info(f"Successfully saved heatmap for sample {data['sample_name']} at epoch {epoch}")  

    print(log_info)  
    logger.info(log_info)  

    # wandb记录  
    wandb_log_dict = {  
        "Validation Total Loss": avg_loss,  
        "Validation Seg Loss": avg_seg_loss,  
        "Validation Abnormal Loss": avg_abnormal_loss,  
        "Validation Seg_Dice Score": avg_dice,  
        "Validation Avg_Accuracy": avg_accuracy,  
        "Validation Avg_Precision": avg_precision,  
        "Validation Avg_Recall": avg_recall,  
        "Validation Avg_F1": avg_f1,  
        "Validation Avg_AUROC": avg_auroc,  
        "Valid Segmentation Weight": current_seg_weight,  
        "Valid Abnormal Weight": config.abnormal_loss_weight,  
        "Validation Total Confusion Matrix": wandb.Image(total_cm_path),  
        "Validation Total TN": total_tn,  
        "Validation Total FP": total_fp,  
        "Validation Total FN": total_fn,  
        "Validation Total TP": total_tp,  
        "Validation Total Accuracy": total_accuracy,  
        "Validation Total Precision": total_precision,  
        "Validation Total Recall": total_recall,  
        "Validation Total F1": total_f1  
    }  

    # 添加每个疾病的独立指标、最佳阈值和混淆矩阵  
    for i, disease in enumerate(disease_names):  
        disease_metrics = {  
            f"Validation_{disease}_Loss": np.mean(abnormal_loss_list[i]),  
            f"Validation_{disease}_Accuracy": accuracy_per_disease[i],  
            f"Validation_{disease}_Precision": precision_per_disease[i],  
            f"Validation_{disease}_Recall": recall_per_disease[i],  
            f"Validation_{disease}_F1": f1_per_disease[i],  
            f"Validation_{disease}_AUROC": auroc_per_disease[i],  
            f"Validation_{disease}_Best_Threshold": best_thresholds[i],  
            f"Validation_{disease}_Confusion_Matrix": wandb.Image(  
                os.path.join(confusion_matrix_dir, f'confusion_matrix_{disease.replace(" ", "_")}.png')  
            )  
        }  
        # wandb_log_dict.update(disease_metrics)  

    # 添加器官指标和混淆矩阵到wandb记录  
    for organ, metrics in organ_metrics.items():  
        organ_wandb_metrics = {  
            f"Validation_{organ}_AUROC": metrics['auroc'],  
            f"Validation_{organ}_Accuracy": metrics['accuracy'],  
            f"Validation_{organ}_Precision": metrics['precision'],  
            f"Validation_{organ}_Recall": metrics['recall'],  
            f"Validation_{organ}_F1": metrics['f1'],  
            f"Validation_{organ}_Confusion_Matrix": wandb.Image(  
                os.path.join(confusion_matrix_dir, f'confusion_matrix_{organ.replace(" ", "_")}.png')  
            )  
        }  
        # wandb_log_dict.update(organ_wandb_metrics)  

    # 记录到wandb  
    # wandb.log(wandb_log_dict, step=epoch)  

    return avg_loss, avg_auroc, best_thresholds  # 返回最佳阈值供后续使用



def test_one_epoch(test_loader, model, segmentation_criterion,abnormal_criterion, epoch, logger, config, writer, device, best_thresholds, save_heatmap=True):  
    model.eval()  
    loss_list = []  
    abnormal_loss_list = [[] for _ in range(16)]  
    sample_idx = 0  
    np.random.seed(42)  
    
    dataset_size = len(test_loader.dataset)  
    selected_indices = set(np.random.choice(dataset_size, min(40, dataset_size), replace=False))  
    current_dir = config.work_dir  

    if save_heatmap:  
        save_dir = os.path.join(config.work_dir, "test_results")  
        os.makedirs(save_dir, exist_ok=True)  
        heatmap_data = []  

    logger.info(f"Test epoch {epoch}: Abnormal weight = {config.abnormal_loss_weight}")  

    test_loader = tqdm(test_loader, desc=f"Epoch {epoch} Test", leave=True, dynamic_ncols=True)  

    # 初始化评估指标数组（使用14种疾病的数量）  
    processed_predictions = [[] for _ in range(14)]  
    all_targets = [[] for _ in range(14)]  
    tp_sum = np.zeros(14, dtype=int)  
    tn_sum = np.zeros(14, dtype=int)  
    fp_sum = np.zeros(14, dtype=int)  
    fn_sum = np.zeros(14, dtype=int)  

    # 保存样本信息的列表
    all_sample_names = []
    all_sample_predictions = []
    all_sample_targets = []

    # 定义16种疾病名称（模型输出）  
    disease_names_16 = [  
        "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
        "Bronchiectasis", "Interlobular septal thickening"  
    ]  

    # 定义14种疾病名称（实际标签）  
    disease_names_14 = [  
        "Cardiomegaly", "Pericardial effusion", "Hiatal hernia",   
        "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Peribronchial thickening", "Consolidation", "Bronchiectasis",   
        "Interlobular septal thickening"  
    ]  

    # 创建16种疾病到14种疾病的映射索引  
    mapping_16_to_14 = []  
    for disease in disease_names_16:  
        try:  
            idx = disease_names_14.index(disease)  
            mapping_16_to_14.append(idx)  
        except ValueError:  
            mapping_16_to_14.append(-1)  

    # 找到 disease_names_14 在 disease_names_16 中的索引  
    indices_14_in_16 = [disease_names_16.index(name) for name in disease_names_14]  
    print(f'indices_14_in_16:f{indices_14_in_16}')

    # # 根据这些索引提取对应的 best_thresholds  
    # best_thresholds_14 = [best_thresholds[idx] for idx in indices_14_in_16] 
    # print(f'best_thresholds_14:f{best_thresholds_14}')

    organ_names = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", "esophagus"]  
    
    # 更新器官疾病映射以匹配14种疾病  
    organ_disease_mapping = {  
        "lung": [4,5, 6, 7, 8, 11, 13],  # 更新索引以匹配14种疾病  
        "heart": [0, 1],  
        "pleura": [9],  
        "mediastinum": [3],  
        "esophagus": [2],  
        "trachea and bronchie": [10,12]  
    }  

    # 创建保存混淆矩阵的目录  
    confusion_matrix_dir = os.path.join(config.work_dir, "test_results", "test_confusion_matrices", f"epoch_{epoch}")  
    os.makedirs(confusion_matrix_dir, exist_ok=True)   
    # 创建保存 ROC 曲线的目录  
    roc_curve_dir = os.path.join(config.work_dir, "test_results", "roc_curves", f"epoch_{epoch}")  
    os.makedirs(roc_curve_dir, exist_ok=True)   

    with torch.no_grad():  
        for iter, data in enumerate(test_loader):  
            # if iter>100:
            #     break
            images, abnormal_targets, sample_names = data  
            images, abnormal_targets = images.to(device), abnormal_targets.to(device)  

            seg_pred, abnormal_preds = model(images)  
            abnormal_pred = abnormal_preds[-1]  
            
            disease_frequencies = [  
                0.1072, 0.0705, 0.0,  # 将不存在的疾病频率设为0  
                0.1420, 0.2534, 0.1939, 0.2558, 0.4548,  
                0.3666, 0.2672, 0.1185, 0.0,  # 将不存在的疾病频率设为0  
                0.1034, 0.1755, 0.0999, 0.0788  
            ]  

            disease_losses, abnormal_pred_avg = Abnormal_loss(seg_pred, abnormal_pred, abnormal_targets, disease_frequencies)  
            
            abnormal_targets_np = abnormal_targets.cpu().numpy()  
            abnormal_pred_avg_np = abnormal_pred_avg.cpu().numpy()  
            abnormal_targets_np=1-abnormal_targets_np#将阴性作为1
            abnormal_pred_avg_np=1-abnormal_pred_avg_np

            # 保存所有样本信息，用于之后的打印
            for batch_idx in range(images.size(0)):
                sample_name = sample_names[batch_idx]
                # 提取当前样本的所有疾病预测和真实标签
                sample_pred = []
                sample_target = []
                
                for disease_idx in range(14):
                    model_idx = indices_14_in_16[disease_idx]  # 获取在16种疾病中的索引
                    pred_value = abnormal_pred_avg_np[batch_idx, model_idx]
                    gt_value = abnormal_targets_np[batch_idx, disease_idx]
                    sample_pred.append(pred_value)
                    sample_target.append(gt_value)
                
                all_sample_names.append(sample_name)
                all_sample_predictions.append(sample_pred)
                all_sample_targets.append(sample_target)
            
            # 只处理14种疾病的预测结果  
            for i, mapping_idx in enumerate(mapping_16_to_14):  
                if mapping_idx != -1:  # 只处理存在于14种疾病中的疾病  
                    processed_predictions[mapping_idx].extend(abnormal_pred_avg_np[:, i])  
                    all_targets[mapping_idx].extend(abnormal_targets_np[:, mapping_idx])  

            # 记录损失  
            for i, disease_loss in enumerate(disease_losses):  
                if mapping_16_to_14[i] != -1:  # 只记录存在于14种疾病中的损失  
                    abnormal_loss_list[i].append(disease_loss.item())  

            abnormal_loss = torch.mean(torch.stack([loss for i, loss in enumerate(disease_losses)   
                                                  if mapping_16_to_14[i] != -1]))  
            total_loss = config.abnormal_loss_weight * abnormal_loss  
            loss_list.append(total_loss.item())  

            if save_heatmap and iter in selected_indices:  
                for batch_idx in range(images.size(0)):  
                    heatmap_data.append({  
                        'predictions': [  
                            abnormal_preds[0][batch_idx:batch_idx+1].cpu(),  
                            abnormal_preds[1][batch_idx:batch_idx+1].cpu()   
                        ],
                        'seg_pred': seg_pred[batch_idx:batch_idx+1].cpu(),  # 添加分割预测结果
                        'targets': abnormal_targets[batch_idx:batch_idx+1].cpu(),  
                        'images': images[batch_idx:batch_idx+1].cpu(),  
                        'sample_name': sample_names[batch_idx]  
                    })
    # 初始化指标数组  
    precision_per_disease = np.zeros(14)  
    recall_per_disease = np.zeros(14)  
    accuracy_per_disease = np.zeros(14)  
    f1_per_disease = np.zeros(14)  
    auroc_per_disease = np.zeros(14)  

    # 新增列表以收集每个疾病的 y_true 和 y_pred  
    y_true_list = []  
    y_pred_list = []  

    roc_dir = "/path/to/lmx/work/Class_projects/bxg/test/test"  
    os.makedirs(save_dir, exist_ok=True)  # 如果路径不存在，则创建 
    # 创建一个字典保存所有疾病的数据  
    data_dict = {} 

    best_thresholds_14 = np.array([0.5] * 14)  # 初始化每个疾病的最佳阈值 
    # 循环处理每种疾病  
    for disease_idx in range(14):  
        # 获取真实标签和预测概率  
        y_true = np.array(all_targets[disease_idx])  
        y_pred_proba = np.array(processed_predictions[disease_idx])  
        
        # 将数据存入字典  
        disease_name = disease_names_14[disease_idx]  
        data_dict[f"{disease_name}_y_true"] = y_true  
        data_dict[f"{disease_name}_y_pred_proba"] = y_pred_proba  
        # if disease_names_14[disease_idx]=="Lung nodule":
        #     y_true=1-y_true
        #     y_pred_proba=1-y_pred_proba

        # 保存 y_true 和 y_pred_proba，文件名中包含疾病名称  
        y_true_path = os.path.join(roc_curve_dir, f'y_true_{disease_name}.npy')  
        y_pred_proba_path = os.path.join(roc_curve_dir, f'y_pred_proba_{disease_name}.npy')  
        
        np.save(y_true_path, y_true)  
        np.save(y_pred_proba_path, y_pred_proba) 

        
        # 计算ROC曲线  
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)  
        # 计算Youden指数 (J = 敏感性 + 特异性 - 1 = TPR - FPR)  
        youden_index = tpr - fpr  
        
        # 找到最大Youden指数对应的阈值  
        optimal_idx = np.argmax(youden_index)  
        best_threshold = thresholds[optimal_idx]  

        # 打印最佳阈值对应的TPR、FPR和阈值  
        print(f"Disease: {disease_names_14[disease_idx]}")  
        print(f"Optimal Threshold: {best_threshold:.3f}")  
        print(f"TPR (Sensitivity): {tpr[optimal_idx]:.4f}")  
        print(f"FPR: {fpr[optimal_idx]:.4f}")  
        
        # 绘制ROC曲线  
        plt.figure(figsize=(8, 6))  
        plt.plot(fpr, tpr, label="ROC Curve", color="blue", linewidth=2)  
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")  # 对角线  
        plt.xlabel("False Positive Rate (FPR)")  
        plt.ylabel("True Positive Rate (TPR)")  
        plt.title(f"ROC Curve for {disease_names_14[disease_idx]}")  
        plt.legend(loc="lower right")  
        
        # 标注各个阈值对应的点  
        for i, threshold in enumerate(thresholds):  
            if i % 10 == 0:  # 每隔10个点标注一次，避免过于密集  
                plt.scatter(fpr[i], tpr[i], color="red", s=10)  # 标注点  
                plt.text(fpr[i], tpr[i], f"{threshold:.2f}", fontsize=8, color="black")  # 标注阈值  
        
        # 保存图片  
        roc_path = os.path.join(roc_dir, f"roc_curve_{disease_names_14[disease_idx]}.png")  
        plt.savefig(roc_path, dpi=300)  
        plt.close()  # 关闭当前图，避免重叠  
        print(f"ROC curve for {disease_names_14[disease_idx]} saved to {roc_path}")  

    # 保存为 NumPy 压缩文件  
    np_path = os.path.join(roc_dir, "disease_data.npz")  
    np.savez_compressed(np_path, **data_dict)  
    print(f"Saved all disease data to {np_path}")

    # 现在已经计算出了最佳阈值，创建CSV文件并打印样本预测结果
    predictions_csv_path = os.path.join(config.work_dir, "test_results", f"sample_predictions_epoch_{epoch}.csv")
    with open(predictions_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入CSV头部
        header = ['Sample_Name']
        for disease in disease_names_14:
            header.append(f"{disease}_Prob")
            header.append(f"{disease}_Pred")
            header.append(f"{disease}_GT")
        csv_writer.writerow(header)
        
        # 对每个样本应用阈值并打印结果
        for idx, sample_name in enumerate(all_sample_names):
            print(f"\nSample: {sample_name}")
            print("Disease                         | Probability | Prediction | Ground Truth")
            print("--------------------------------|------------|------------|------------")
            
            sample_row = [sample_name]
            sample_pred = all_sample_predictions[idx]
            sample_target = all_sample_targets[idx]
            
            for disease_idx, disease_name in enumerate(disease_names_14):
                # 获取概率值
                prob_value = sample_pred[disease_idx]
                # 应用阈值获取最终预测
                pred_value = 1 if prob_value > best_thresholds_14[disease_idx] else 0
                gt_value = int(sample_target[disease_idx])
                
                print(f"{disease_name:30} | {prob_value:.6f} | {pred_value:10} | {gt_value}")
                
                # 添加到CSV行
                sample_row.append(f"{prob_value:.6f}")
                sample_row.append(f"{pred_value}")
                sample_row.append(f"{gt_value}")
            
            # 写入CSV文件
            csv_writer.writerow(sample_row)
    
    print(f"\n所有样本的预测值已保存到: {predictions_csv_path}")



    # 计算所有指标  
    for disease_idx in range(14):  
        y_true = np.array(all_targets[disease_idx])  
        y_pred_proba = np.array(processed_predictions[disease_idx])  
        y_pred = (y_pred_proba > best_thresholds_14[disease_idx]).astype(int)  


        # 收集 y_true 和 y_pred  
        y_true_list.append(y_true)  
        y_pred_list.append(y_pred)  

        # 计算混淆矩阵并保存  
        cm = confusion_matrix(y_true, y_pred)  
        save_path = plot_confusion_matrix(y_true, y_pred, disease_names_14[disease_idx], confusion_matrix_dir)  

        # # 从混淆矩阵中提取TP、TN、FP、FN  
        # tp = cm[1, 1]  
        # tn = cm[0, 0]  
        # fp = cm[0, 1]  
        # fn = cm[1, 0]  
        # 安全访问混淆矩阵元素
        if cm.shape[0] > 1 and cm.shape[1] > 1:
            tp = cm[1, 1]  # 真阳性
            fp = cm[0, 1]  # 假阳性
            fn = cm[1, 0]  # 假阴性
            tn = cm[0, 0]  # 真阴性
        else:
            # 处理边缘情况
            tp = 0
            fp = 0
            fn = 0
            tn = cm[0, 0] if cm.size > 0 else 0

        # 更新TP、TN、FP、FN的总和  
        tp_sum[disease_idx] = tp  
        tn_sum[disease_idx] = tn  
        fp_sum[disease_idx] = fp  
        fn_sum[disease_idx] = fn  

        # 计算AUROC  
        try:  
            auroc = roc_auc_score(y_true, y_pred_proba)  
        except ValueError:  
            auroc = 0.0  
        auroc_per_disease[disease_idx] = auroc  

        # 计算每个疾病的指标  
        precision_per_disease[disease_idx] = tp / (tp + fp + 1e-8)  
        recall_per_disease[disease_idx] = tp / (tp + fn + 1e-8)  
        accuracy_per_disease[disease_idx] = (tp + tn) / (tp + tn + fp + fn + 1e-8)  
        f1_per_disease[disease_idx] = 2 * precision_per_disease[disease_idx] * recall_per_disease[disease_idx] / (precision_per_disease[disease_idx] + recall_per_disease[disease_idx] + 1e-8)  

        # 计算并绘制每种疾病的 ROC 曲线  
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)  
        roc_auc = auc(fpr, tpr)  

        plt.figure()  
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))  
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  

        # 标注每隔10个阈值的点  
        for i in range(0, len(thresholds), 20):  # 每隔10个点  
            plt.scatter(fpr[i], tpr[i], color='black', s=10)  # 标注点  
            plt.text(fpr[i], tpr[i], f'{thresholds[i]:.2f}', fontsize=8, color='black')  # 标注阈值  

        plt.xlim([0.0, 1.0])  
        plt.ylim([0.0, 1.05])  
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate')  
        plt.title(f'ROC Curve for {disease_names_14[disease_idx]}')  
        plt.legend(loc='lower right')  

        # 使用疾病名字保存ROC曲线  
        disease_name = disease_names_14[disease_idx].replace(" ", "_")  # 替换空格为下划线，避免文件名问题  
        roc_curve_path = os.path.join(roc_curve_dir, f'roc_curve_{disease_name}.png')  
        plt.savefig(roc_curve_path)  
        plt.close()

    # 初始化器官指标数组  
    num_organs = len(organ_names)  
    organ_tp_sum = np.zeros(num_organs)  
    organ_tn_sum = np.zeros(num_organs)  
    organ_fp_sum = np.zeros(num_organs)  
    organ_fn_sum = np.zeros(num_organs)  

    organ_precision = np.zeros(num_organs)  
    organ_recall = np.zeros(num_organs)  
    organ_accuracy = np.zeros(num_organs)  
    organ_f1 = np.zeros(num_organs)  
    organ_auroc = np.zeros(num_organs)  

    # # 计算每个器官的指标  
    # for organ_idx, (organ, disease_indices) in enumerate(organ_disease_mapping.items()):  
    #     organ_predictions = []  
    #     organ_targets = []  
    #     organ_pred_proba = []  
    #     organ_tp = organ_tn = organ_fp = organ_fn = 0   

    #     for disease_idx in disease_indices:  
    #         organ_targets.extend(y_true_list[disease_idx])  
    #         organ_predictions.extend(y_pred_list[disease_idx])  
    #         organ_pred_proba.extend(processed_predictions[disease_idx])  
            
    #         organ_tp += tp_sum[disease_idx]  
    #         organ_tn += tn_sum[disease_idx]  
    #         organ_fp += fp_sum[disease_idx]  
    #         organ_fn += fn_sum[disease_idx]  

    #     organ_tp_sum[organ_idx] = organ_tp  
    #     organ_tn_sum[organ_idx] = organ_tn  
    #     organ_fp_sum[organ_idx] = organ_fp  
    #     organ_fn_sum[organ_idx] = organ_fn   

    #     # 计算该器官的AUROC  
    #     organ_auroc_value = roc_auc_score(organ_targets, organ_pred_proba) if organ_targets else 0.0  
    #     organ_auroc[organ_idx] = organ_auroc_value  

    #     # 计算器官级别的指标  
    #     total = organ_tp + organ_fp + 1e-8  
    #     organ_precision[organ_idx] = organ_tp / total  
    #     organ_recall[organ_idx] = organ_tp / (organ_tp + organ_fn + 1e-8)  
    #     organ_accuracy[organ_idx] = (organ_tp + organ_tn) / (organ_tp + organ_tn + organ_fp + organ_fn + 1e-8)  
    #     organ_f1[organ_idx] = 2 * organ_precision[organ_idx] * organ_recall[organ_idx] / (organ_precision[organ_idx] + organ_recall[organ_idx] + 1e-8)  

    #     # 绘制混淆矩阵并保存  
    #     save_path, cm = plot_confusion_matrix(organ_targets, organ_predictions, organ, confusion_matrix_dir)  

    #     # 计算并绘制该器官的 ROC 曲线  
    #     fpr, tpr, _ = roc_curve(organ_targets, organ_pred_proba)  
    #     organ_roc_auc = auc(fpr, tpr)  

    #     plt.figure()  
    #     plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = {:.2f})'.format(organ_roc_auc))  
    #     plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  
    #     plt.xlim([0.0, 1.0])  
    #     plt.ylim([0.0, 1.05])  
    #     plt.xlabel('False Positive Rate')  
    #     plt.ylabel('True Positive Rate')  
    #     plt.title(f'ROC Curve for {organ}')  
    #     plt.legend(loc='lower right')  
        
    #     organ_roc_curve_path = os.path.join(roc_curve_dir, f'roc_curve_organ_{organ}.png')  
    #     plt.savefig(organ_roc_curve_path)  
    #     plt.close()
    
        # 计算平均指标  
    avg_loss = np.mean(loss_list)  
    avg_abnormal_loss = np.mean([np.mean(losses) for losses in abnormal_loss_list])  
    
    # 计算疾病级别的平均指标  
    avg_accuracy = np.mean(accuracy_per_disease)  
    avg_precision = np.mean(precision_per_disease)  
    avg_recall = np.mean(recall_per_disease)  
    avg_f1 = np.mean(f1_per_disease)  
    avg_auroc = np.mean(auroc_per_disease)  

    # 计算器官级别的平均指标  
    avg_organ_accuracy = np.mean(organ_accuracy)  
    avg_organ_precision = np.mean(organ_precision)  
    avg_organ_recall = np.mean(organ_recall)  
    avg_organ_f1 = np.mean(organ_f1)  
    avg_organ_auroc = np.mean(organ_auroc)  

    # # 准备日志信息  
    # log_info = f"Test Epoch {epoch} Summary:\n\n"  
    
    # # 总体指标  
    # log_info += "Overall Metrics:\n"  
    # log_info += f"Total Loss: {avg_loss:.4f}, Abnormal Loss: {avg_abnormal_loss:.4f}\n"  
    # log_info += f"Average Disease Metrics: Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, "  
    # log_info += f"Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, AUROC: {avg_auroc:.4f}\n"  
    # log_info += f"Average Organ Metrics: Accuracy: {avg_organ_accuracy:.4f}, Precision: {avg_organ_precision:.4f}, "  
    # log_info += f"Recall: {avg_organ_recall:.4f}, F1: {avg_organ_f1:.4f}, AUROC: {avg_organ_auroc:.4f}\n\n"  

    # # 每种疾病的详细指标  
    # log_info += "Disease-wise Metrics:\n"  
    # for i in range(14):  
    #     log_info += f"{disease_names_14[i]}:\n"  
    #     log_info += f"Accuracy: {accuracy_per_disease[i]:.4f}, "  
    #     log_info += f"Precision: {precision_per_disease[i]:.4f}, "  
    #     log_info += f"Recall: {recall_per_disease[i]:.4f}, "  
    #     log_info += f"F1: {f1_per_disease[i]:.4f}, "  
    #     log_info += f"AUROC: {auroc_per_disease[i]:.4f}\n"  
    #     log_info += f"TP: {tp_sum[i]}, TN: {tn_sum[i]}, FP: {fp_sum[i]}, FN: {fn_sum[i]}\n\n"  

    # # 每个器官的详细指标  
    # log_info += "Organ-wise Metrics:\n"  
    # for i, organ in enumerate(organ_names):  
    #     log_info += f"{organ}:\n"  
    #     log_info += f"Accuracy: {organ_accuracy[i]:.4f}, "  
    #     log_info += f"Precision: {organ_precision[i]:.4f}, "  
    #     log_info += f"Recall: {organ_recall[i]:.4f}, "  
    #     log_info += f"F1: {organ_f1[i]:.4f}, "  
    #     log_info += f"AUROC: {organ_auroc[i]:.4f}\n"  
    #     log_info += f"TP: {organ_tp_sum[i]}, TN: {organ_tn_sum[i]}, "  
    #     log_info += f"FP: {organ_fp_sum[i]}, FN: {organ_fn_sum[i]}\n\n"  

    # # 记录到tensorboard  
    # if writer is not None:  
    #     # 记录总体指标  
    #     writer.add_scalar('Test/Total_Loss', avg_loss, epoch)  
    #     writer.add_scalar('Test/Abnormal_Loss', avg_abnormal_loss, epoch)  
        
    #     # 记录疾病级别的平均指标  
    #     writer.add_scalar('Test/Avg_Disease_Accuracy', avg_accuracy, epoch)  
    #     writer.add_scalar('Test/Avg_Disease_Precision', avg_precision, epoch)  
    #     writer.add_scalar('Test/Avg_Disease_Recall', avg_recall, epoch)  
    #     writer.add_scalar('Test/Avg_Disease_F1', avg_f1, epoch)  
    #     writer.add_scalar('Test/Avg_Disease_AUROC', avg_auroc, epoch)  
        
    #     # 记录器官级别的平均指标  
    #     writer.add_scalar('Test/Avg_Organ_Accuracy', avg_organ_accuracy, epoch)  
    #     writer.add_scalar('Test/Avg_Organ_Precision', avg_organ_precision, epoch)  
    #     writer.add_scalar('Test/Avg_Organ_Recall', avg_organ_recall, epoch)  
    #     writer.add_scalar('Test/Avg_Organ_F1', avg_organ_f1, epoch)  
    #     writer.add_scalar('Test/Avg_Organ_AUROC', avg_organ_auroc, epoch)  
        
    #     # 记录每种疾病的指标  
    #     for i in range(14):  
    #         writer.add_scalar(f'Test/Disease_{disease_names_14[i]}/Accuracy', accuracy_per_disease[i], epoch)  
    #         writer.add_scalar(f'Test/Disease_{disease_names_14[i]}/Precision', precision_per_disease[i], epoch)  
    #         writer.add_scalar(f'Test/Disease_{disease_names_14[i]}/Recall', recall_per_disease[i], epoch)  
    #         writer.add_scalar(f'Test/Disease_{disease_names_14[i]}/F1', f1_per_disease[i], epoch)  
    #         writer.add_scalar(f'Test/Disease_{disease_names_14[i]}/AUROC', auroc_per_disease[i], epoch)  
        
    #     # 记录每个器官的指标  
    #     for i, organ in enumerate(organ_names):  
    #         writer.add_scalar(f'Test/Organ_{organ}/Accuracy', organ_accuracy[i], epoch)  
    #         writer.add_scalar(f'Test/Organ_{organ}/Precision', organ_precision[i], epoch)  
    #         writer.add_scalar(f'Test/Organ_{organ}/Recall', organ_recall[i], epoch)  
    #         writer.add_scalar(f'Test/Organ_{organ}/F1', organ_f1[i], epoch)  
    #         writer.add_scalar(f'Test/Organ_{organ}/AUROC', organ_auroc[i], epoch)  

    # 保存热图  
    # if save_heatmap and heatmap_data:  
    #     logger.info("Generating and saving heatmaps with optimal thresholds...")  
    #     for data in heatmap_data:  
    #         save_prediction_heatmaps(  
    #             predictions=data['predictions'],  
    #             segmentation_preds=data['seg_pred'],  # 传入分割预测结果
    #             targets=data['targets'],  
    #             images=data['images'],  
    #             epoch=epoch,  
    #             organ_names=organ_names,  
    #             sample_idx=data['sample_name'],  
    #             base_dir=save_dir,  
    #             seg_threshold=0.5,  # 设置分割阈值
    #             topk=3, 
    #             abnormal_threshold=best_thresholds_14  
    #         )  

    
            # logger.info(f"Successfully saved heatmap for sample {data['sample_name']} at epoch {epoch}")  

    # print(log_info)  
    # logger.info(log_info)  

    return avg_loss