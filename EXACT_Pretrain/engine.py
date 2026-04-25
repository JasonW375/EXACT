import os
import math
from pathlib import Path

# 数据处理与科学计算
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt

# 深度学习相关
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

# 评测指标
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

# 绘图与可视化
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# 进度条与实验记录
from tqdm import tqdm
import wandb



def calculate_nsd(pred, target, threshold=0.5, tau=1.0):
    """
    计算标准化表面Dice（NSD）
    :param pred: 预测的分割掩码，形状 [B, C, D, H, W]
    :param target: 真实的分割掩码，形状 [B, C, D, H, W]
    :param threshold: 二值化阈值
    :param tau: 表面距离容忍阈值（单位：像素）
    :return: NSD值，形状 [B, C]
    """
    pred = (pred > threshold).float()
    target = target.float()
    batch_size, num_organs = pred.shape[0], pred.shape[1]
    nsd_per_organ = torch.zeros((batch_size, num_organs), device=pred.device)
    
    for b in range(batch_size):
        for organ_idx in range(num_organs):
            pred_organ = pred[b, organ_idx].cpu().numpy()
            target_organ = target[b, organ_idx].cpu().numpy()
            
            # 提取表面点
            def get_surface(mask):
                if not np.any(mask):
                    return np.zeros_like(mask, dtype=bool)
                eroded = binary_erosion(mask, structure=np.ones((3,3,3)))
                return mask & ~eroded
            
            surface_pred = get_surface(pred_organ > 0.5)
            surface_target = get_surface(target_organ > 0.5)
            
            surface_pred_coords = np.argwhere(surface_pred)
            surface_target_coords = np.argwhere(surface_target)
            
            # 处理特殊情况
            if len(surface_pred_coords) + len(surface_target_coords) == 0:
                nsd = 1.0
            elif len(surface_pred_coords) == 0 or len(surface_target_coords) == 0:
                nsd = 0.0
            else:
                # 计算距离变换
                dist_map_target = distance_transform_edt(~surface_target)
                dist_map_pred = distance_transform_edt(~surface_pred)
                
                # 计算匹配点数
                pred_dists = dist_map_target[tuple(surface_pred_coords.T)]
                target_dists = dist_map_pred[tuple(surface_target_coords.T)]
                tp_a = np.sum(pred_dists <= tau)
                tp_b = np.sum(target_dists <= tau)
                
                nsd = (tp_a + tp_b) / (len(surface_pred_coords) + len(surface_target_coords))
                
            nsd_per_organ[b, organ_idx] = nsd
            
    return nsd_per_organ

# 双向权重
import torch
import torch.nn.functional as F


#一次/根号
# def weighted_binary_cross_entropy(pred, target, weight_pos, weight_neg):
#     """
#     计算加权的二元交叉熵损失。
#     pred: 预测值，形状为 (B,)
#     target: 真实标签，形状为 (B,)
#     weight_pos: 正样本权重
#     weight_neg: 负样本权重
#     """
#     # 计算标准的二元交叉熵损失
#     loss_pos = weight_pos * target * torch.log(torch.sigmoid(pred) + 1e-6)
#     loss_neg = weight_neg * (1 - target) * torch.log(1 - torch.sigmoid(pred) + 1e-6)
#     loss = -(loss_pos + loss_neg)
#     return loss
#模型输出已经激活不能再次激活！
def weighted_binary_cross_entropy(pred, target, weight_pos, weight_neg):
    """
    计算加权的二元交叉熵损失。
    pred: 预测值，形状为 (B,)
    target: 真实标签，形状为 (B,)
    weight_pos: 正样本权重
    weight_neg: 负样本权重
    """
    # 计算标准的二元交叉熵损失
    loss_pos = weight_pos * target * torch.log(pred + 1e-6)
    loss_neg = weight_neg * (1 - target) * torch.log(1 - pred + 1e-6)
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
    6: global
    
    疾病顺序调整为：
    0: "Medical material"
    1: "Arterial wall calcification"
    2: "Cardiomegaly"
    3: "Pericardial effusion"
    4: "Coronary artery wall calcification"
    5: "Hiatal hernia"
    6: "Lymphadenopathy"
    7: "Emphysema"
    8: "Atelectasis"
    9: "Lung nodule"
    10: "Lung opacity"
    11: "Pulmonary fibrotic sequela"
    12: "Pleural effusion"
    13: "Mosaic attenuation pattern"
    14: "Peribronchial thickening"
    15: "Consolidation"
    16: "Bronchiectasis"
    17: "Interlobular septal thickening"
    """  
    return {  
        0: 6,    # "Medical material" -> global
        1: 6,    # "Arterial wall calcification" -> global
        2: 4,    # "Cardiomegaly" -> heart (4)  
        3: 4,    # "Pericardial effusion" -> heart (4)  
        4: 4,    # "Coronary artery wall calcification" -> heart (4)  
        5: 5,    # "Hiatal hernia" -> esophagus (5)  
        6: 3,    # "Lymphadenopathy" -> mediastinum (3)  
        7: 0,    # "Emphysema" -> lung (0)      
        8: 0,    # "Atelectasis" -> lung (0)  
        9: 0,    # "Lung nodule" -> lung (0)  
        10: 0,   # "Lung opacity" -> lung (0)  
        11: 0,   # "Pulmonary fibrotic sequela" -> lung (0)  
        12: 2,   # "Pleural effusion" -> pleura (2)  
        13: 0,   # "Mosaic attenuation pattern" -> lung (0)  
        14: 1,   # "Peribronchial thickening" -> trachea and bronchie (1)  
        15: 0,   # "Consolidation" -> lung (0)  
        16: 1,   # "Bronchiectasis" -> trachea and bronchie (1)  
        17: 0,   # "Interlobular septal thickening" -> lung (0)  
    }

def Abnormal_loss_multiscale(seg_pred, abnormal_preds, abnormal_targets, disease_frequencies, k=6, epsilon=1e-6, seg_threshold=0.5):  
    """  
    计算多尺度异常检测损失  
    Args:  
        seg_pred: 分割预测 (B, 7, D, H, W)  
        abnormal_preds: 疾病预测列表 [scale1_pred, scale2_pred]，每个元素形状为(B, 18, D, H, W)  
        abnormal_targets: 疾病标签 (B, 18)  
        disease_frequencies: 18种疾病的阳性样本频率列表  
        k: top-k取值  
        epsilon: 数值稳定性常数  
        seg_threshold: 分割掩码阈值  
    """  
    B, _, D, H, W = seg_pred.shape  
    num_diseases = abnormal_preds[0].shape[1]  # 18个疾病通道  
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


def calculate_dice(pred, target, threshold=0.5):
    # 修改 calculate_dice 函数，确保返回每个批次、每个器官的平均 Dice 分数
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2 * intersection + 1e-7) / (union + 1e-7)  # 防止除零
    # 对批次维度取平均，返回每个器官的平均 Dice 分数
    return dice.mean(dim=0)

def train_one_epoch(train_loader, model, segmentation_criterion, abnormal_criterion, optimizer, scheduler, epoch, step, logger, config, writer, device):  
    model.train()  
    loss_list = []  
    seg_loss_list = []  
    # 为每个尺度准备disease loss列表  
    scale_disease_loss_list = [[[] for _ in range(18)] for _ in range(2)]  # 2个尺度，现在是18个疾病  
    
    # 为每个尺度准备organ dice列表  
    scale_organ_dice_scores = [  
        [[] for _ in range(7)] for _ in range(2)  # 2个尺度，现在是7个通道（6个器官+1个全局）  
    ]  

    current_seg_weight = calculate_dynamic_weight(  
        epoch=epoch,  
        initial_weight=config.initial_segmentation_weight,  
        total_epochs=config.epochs,  
        decay_rate=config.weight_decay_rate  
    )  
    
    logger.info(f"Current epoch {epoch}: Segmentation weight = {current_seg_weight:.4f}, "  
                f"Abnormal weight = {config.abnormal_loss_weight}")  

    train_loader = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=True)  
    
    # 更新器官名称，增加全局通道
    organ_names = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", "esophagus", "global"]  
    
    # 更新疾病名称，增加两个新的疾病
    disease_names = [
        "Medical material", "Arterial wall calcification",
        "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
        "Bronchiectasis", "Interlobular septal thickening"  
    ]  

    # 更新疾病频率，增加新疾病的频率
    disease_frequencies = [  
        0.102, 0.2837,  # 新增的两个疾病
        0.1072, 0.0705, 0.2476, 0.1420, 0.2534, 0.1939, 0.2558, 0.4548,  
        0.3666, 0.2672, 0.1185, 0.0744, 0.1034, 0.1755, 0.0999, 0.0788  
    ]  

    for iter, data in enumerate(train_loader):  
        # if iter >= 1500:  
        #     break  
        optimizer.zero_grad()  

        images, seg_targets, abnormal_targets, _ = data  
        images = images.to(device)  
        seg_targets = seg_targets.to(device)  
        abnormal_targets = abnormal_targets.to(device)  
        # print(f"seg_targets shape: {seg_targets.shape}")  # 应该是 (B, 7, D, H, W) 
        # print(f"abnormal_targets: {abnormal_targets.shape}")  # 应该是 (B, 18) 

        seg_pred, abnormal_preds = model(images)  # abnormal_preds是列表[scale1_pred, scale2_pred]  
        # print(f"seg_pred shape: {seg_pred.shape}")  # 应该是 (B, 7, D, H, W) 
        # print(f"abnormal_preds: {abnormal_preds.shape}")  # 应该是 (B, 18) 

        seg_loss = segmentation_criterion(seg_pred, seg_targets)  
        scale_losses, scale_predictions = Abnormal_loss_multiscale(seg_pred, abnormal_preds, abnormal_targets, disease_frequencies)  
        
        # 保存每个尺度每个疾病的loss  
        for scale_idx, scale_loss in enumerate(scale_losses):  
            for disease_idx, disease_loss in enumerate(scale_loss):  
                scale_disease_loss_list[scale_idx][disease_idx].append(disease_loss.item())  

        # 计算总的异常检测损失（所有尺度的平均）  
        abnormal_loss = torch.mean(torch.stack([torch.mean(losses) for losses in scale_losses]))  
        total_loss = current_seg_weight * seg_loss + config.abnormal_loss_weight * abnormal_loss  

        total_loss.backward()  
        optimizer.step()  

        loss_list.append(total_loss.item())  
        seg_loss_list.append(seg_loss.item())  

        # 计算每个尺度的Dice分数  
        for scale_idx, abnormal_pred in enumerate(abnormal_preds):  
            # 将分割预测和目标调整到当前尺度  
            current_size = abnormal_pred.shape[2:]  
            scaled_seg_pred = F.interpolate(seg_pred, size=current_size, mode='trilinear', align_corners=False)  
            scaled_seg_target = F.interpolate(seg_targets, size=current_size, mode='trilinear', align_corners=False)  
            
            # 计算每个器官的Dice分数（现在包括全局通道）  
            for organ_idx, organ_name in enumerate(organ_names):  
                organ_dice = calculate_dice(  
                    scaled_seg_pred[:, organ_idx:organ_idx+1],  
                    scaled_seg_target[:, organ_idx:organ_idx+1]  
                ).item()  
                scale_organ_dice_scores[scale_idx][organ_idx].append(organ_dice)  

        train_loader.set_postfix({  
            'Loss': f'{total_loss.item():.4f}',  
            'Seg_Loss': f'{seg_loss.item():.4f}',  
            'Abn_Loss': f'{abnormal_loss.item():.4f}'  
        })  

    # 计算平均指标  
    avg_loss = np.mean(loss_list)  
    avg_seg_loss = np.mean(seg_loss_list)  
    
    # 计算每个尺度的平均disease losses  
    avg_scale_disease_losses = [  
        [np.mean(losses) for losses in scale_losses]   
        for scale_losses in scale_disease_loss_list  
    ]  
    
    # 计算每个尺度的平均organ dice  
    avg_scale_organ_dice = [  
        [np.mean(scores) for scores in scale_dices]  
        for scale_dices in scale_organ_dice_scores  
    ]  

    # 使用wandb记录训练指标  
    wandb_metrics = {  
        "train/total_loss": avg_loss,  
        "train/seg_loss": avg_seg_loss,  
        "train/abnormal_loss": np.mean([np.mean(losses) for losses in avg_scale_disease_losses]),  
    }  

    # 记录每个尺度每个器官的Dice分数  
    for scale_idx in range(len(avg_scale_organ_dice)):  
        for organ_idx, organ_name in enumerate(organ_names):  
            wandb_metrics[f"train/scale{scale_idx+1}_dice_{organ_name}"] = avg_scale_organ_dice[scale_idx][organ_idx]  

    # 记录每个尺度每个疾病的loss  
    for scale_idx in range(len(avg_scale_disease_losses)):  
        for disease_idx, disease_name in enumerate(disease_names):  
            wandb_metrics[f"train/scale{scale_idx+1}_disease_loss_{disease_name}"] = avg_scale_disease_losses[scale_idx][disease_idx]  

    # 使用wandb记录所有指标  
    wandb.log(wandb_metrics, step=epoch)  

    if scheduler is not None:  
        scheduler.step()  

    step += len(train_loader)  
    return step


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

def save_prediction_heatmaps(  
    predictions,   
    segmentation_preds,   
    targets,   
    images,   
    epoch,   
    organ_names,   
    sample_idx,  
    base_dir=None,   
    seg_threshold=0.5,   
    topk=3,   
    abnormal_threshold=None  
):  
    # 解包多尺度预测结果  
    low_res_preds, high_res_preds = predictions  
    
    # 获取目标大小（高分辨率）  
    _, _, target_depth, target_height, target_width = high_res_preds.shape  
    
    # 对低分辨率预测进行上采样  
    low_res_upsampled = upsample_3d(low_res_preds, (target_depth, target_height, target_width))  

    # 选择使用高分辨率预测  
    high_res_pred = high_res_preds[0].cpu().numpy()  # [18, D, H, W]  
    low_res_pred = low_res_upsampled[0].cpu().numpy()  # [18, D, H, W]  
    
    if base_dir is None:  
        base_dir = Path.cwd()  
    else:  
        base_dir = Path(base_dir)  
    
    save_dir = base_dir / "prediction_heatmaps"  
    epoch_dir = save_dir / f"epoch_{epoch}"  
    sample_dir = epoch_dir / str(sample_idx)  
    sample_dir.mkdir(parents=True, exist_ok=True)  
    
    # 3D数据处理：提取分割预测  
    seg_pred = segmentation_preds[0].cpu().numpy()  # [7, D, H, W]  
    
    # 3D分割掩码生成  
    seg_mask = (seg_pred > seg_threshold).astype(np.float32)  
    target = targets[0].cpu().numpy()  # [18]  
    original_image = images[0, 0].cpu().numpy()  # [D, H, W]  
    
    # 保存原始图像  
    affine = np.eye(4)  
    original_nifti = nib.Nifti1Image(original_image, affine)  
    original_nifti.header['descrip'] = f'Original 3D Image, Epoch: {epoch}'  
    original_save_path = sample_dir / f"original_image.nii.gz"  
    nib.save(original_nifti, original_save_path)  
    
    # 更新疾病名称列表  
    disease_names = [
        "Medical material", "Arterial wall calcification",
        "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
        "Bronchiectasis", "Interlobular septal thickening"  
    ]  
    
    # 更新疾病-器官映射关系  
    disease_organ_mapping = {  
        "Medical material": ["global"],
        "Arterial wall calcification": ["global"],
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
            for disease_idx, disease_name in enumerate(disease_names):  
                # 3D预测图处理  
                disease_pred = pred[disease_idx]  # 当前疾病的3D预测图 [D, H, W]  
                disease_label = int(target[disease_idx])  # 当前疾病的真实标签  
                
                # 获取相关器官的分割掩码  
                related_organs = disease_organ_mapping[disease_name]  
                combined_mask = np.zeros_like(seg_mask[0])  
                for organ_name in related_organs:  
                    organ_idx = organ_names.index(organ_name)  
                    combined_mask = np.maximum(combined_mask, seg_mask[organ_idx])  
                
                # 计算combined_pred（将预测与器官掩码相乘）  
                combined_pred = disease_pred * combined_mask  
                
                # 计算前k个最大值并计算平均值  
                topk_mean = np.mean(np.sort(combined_pred.flatten())[-topk:])  
                
                # 使用对应疾病的最佳阈值  
                current_threshold = abnormal_threshold[disease_idx] if abnormal_threshold is not None else 0.5  
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
                
                # 保存3D预测结果为NIfTI格式   
                nifti_img = nib.Nifti1Image(combined_pred, affine)  
                nifti_img.header['descrip'] = (f'{res_name} 3D Disease: {disease_name}, Epoch: {epoch}, '  
                                             f'Sample: {sample_idx}, '  
                                             f'GT Label: {disease_label}, Pred: {pred_abnormal}, '  
                                             f'Top-{topk} Mean: {topk_mean:.4f}, '  
                                             f'Threshold: {current_threshold:.4f}, '  
                                             f'Related Organs: {", ".join(related_organs)}')  
                save_path = sample_dir / f"{disease_name}_{result_str}_{res_name.lower()}_combined_pred.nii.gz"  
                nib.save(nifti_img, save_path)  

    return save_dir

    

def Abnormal_loss(seg_pred, abnormal_pred, abnormal_targets, disease_frequencies, k=3, epsilon=1e-6, seg_threshold=0.5):  
    """  
    计算异常检测损失  
    Args:  
        seg_pred: 分割预测 (B, 7, D, H, W)  
        abnormal_pred: 疾病预测 (B, 18, D, H, W)  
        abnormal_targets: 疾病标签 (B, 18)  
        disease_frequencies: 16种疾病的阳性样本频率列表  
        k: top-k取值  
        epsilon: 数值稳定性常数  
        seg_threshold: 分割掩码阈值  
    """  
    B, _, D, H, W = seg_pred.shape  
    num_diseases = abnormal_pred.shape[1]  # 18个疾病通道  
    
    # 获取疾病到器官的映射  
    disease_to_organ = get_organ_disease_mapping()  
    
    # 对分割预测进行二值化  
    seg_mask = (seg_pred > seg_threshold).float()  
    
    disease_losses = []  
    disease_predictions = []  
    
    for disease_idx in range(num_diseases):  
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
        weight_pos = 1.0#val 不加权重
        weight_neg = 1.0  
        
        # 计算损失  
        target = abnormal_targets[:, disease_idx]  
        loss = weighted_binary_cross_entropy(avg_top_k, target, weight_pos, weight_neg)  
        loss = loss.mean()  
        
        disease_losses.append(loss)  
        disease_predictions.append(avg_top_k)  
    
    return disease_losses, torch.stack(disease_predictions, dim=1)  

def valid_one_epoch(valid_loader, model, segmentation_criterion, abnormal_criterion, epoch, logger, config, writer, device, save_heatmap=True):
    model.eval()
    loss_list = []
    seg_loss_list = []
    # 将 abnormal_loss_list 改为 18 个通道
    abnormal_loss_list = [[] for _ in range(18)]
    sample_idx = 0
    np.random.seed(42)
    
    # 阈值搜索相关变量也需要扩展到 18 个
    threshold_candidates = np.arange(0.1, 0.9, 0.05)  # 从0.1到0.9，步长0.05
    best_thresholds = np.array([0.5] * 18)  # 初始化每个疾病的最佳阈值
    best_f1_scores = np.array([-1] * 18)  # 记录每个疾病的最佳F1分数
    
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
    dice_scores_organ =[]
    # 扩展到 18 个通道
    tp_sum = np.zeros(18)
    tn_sum = np.zeros(18)
    fp_sum = np.zeros(18)
    fn_sum = np.zeros(18)

    # 修改预测结果的收集列表为 18 个
    processed_predictions = [[] for _ in range(18)]  # 存储处理后的预测结果
    all_targets = [[] for _ in range(18)]

    # 更新器官名称
    organ_names = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", "esophagus", "global"]  

    # 更新疾病名称，增加两个新的疾病
    disease_names = [
        "Medical material", "Arterial wall calcification",
        "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
        "Bronchiectasis", "Interlobular septal thickening"  
    ]

    # 修订后的器官-疾病映射，增加新的疾病映射
    organ_disease_mapping = {  
        "global": [0, 1],  # 新增的两个疾病
        "lung": [7, 8, 9, 10, 11, 13, 15,17],  
        "heart": [2, 3, 4],  
        "pleura": [12],  
        "mediastinum": [6],  
        "esophagus": [5],  
        "trachea and bronchie": [14, 16]  
    }

    # 更新疾病频率，增加新疾病的频率
    disease_frequencies = [  
        0.102, 0.2837,  # 新增的两个疾病
        0.1072, 0.0705, 0.2476, 0.1420, 0.2534, 0.1939, 0.2558, 0.4548,  
        0.3666, 0.2672, 0.1185, 0.0744, 0.1034, 0.1755, 0.0999, 0.0788  
    ]

    with torch.no_grad():
        for iter, data in enumerate(valid_loader):
            # if iter >= 100:  
            #     break  
            images, seg_targets, abnormal_targets, sample_names = data
            images, seg_targets, abnormal_targets = images.to(device), seg_targets.to(device), abnormal_targets.to(device)

            seg_pred, abnormal_preds = model(images)
            abnormal_pred = abnormal_preds[-1]  # 取最后一个尺度的输出 
            
            seg_loss = segmentation_criterion(seg_pred, seg_targets)
            disease_losses, abnormal_pred_avg = Abnormal_loss(seg_pred, abnormal_pred, abnormal_targets, disease_frequencies)
            
            # 收集处理后的预测结果和真实标签
            abnormal_targets_np = abnormal_targets.cpu().numpy()
            abnormal_pred_avg_np = abnormal_pred_avg.cpu().numpy()

            for i in range(18):
                processed_predictions[i].extend(abnormal_pred_avg_np[:, i])
                all_targets[i].extend(abnormal_targets_np[:, i])

            for i, disease_loss in enumerate(disease_losses):
                abnormal_loss_list[i].append(disease_loss.item())

            abnormal_loss = torch.mean(torch.stack(disease_losses))
            total_loss = current_seg_weight * seg_loss + config.abnormal_loss_weight * abnormal_loss

            loss_list.append(total_loss.item())
            seg_loss_list.append(seg_loss.item())

            # 计算Dice分数，注意现在是7个通道
            dice_score_organ = calculate_dice(seg_pred, seg_targets)
            dice_score = dice_score_organ.mean().item()  
            dice_score_organ = dice_score_organ.cpu().numpy()

            dice_scores.append(dice_score)
            dice_scores_organ.append(dice_score_organ)

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

    # 将所有样本的Dice系数转换为NumPy数组  
    dice_scores_organ = np.array(dice_scores_organ)  

    # 保存为.npy文件  
    np.save('dice_scores.npy', dice_scores_organ) 

    # 将原来的阈值搜索部分替换为以下代码  
    logger.info("Finding optimal thresholds for each disease using Youden's index...")  
    # 修改为18个疾病
    for disease_idx in range(18):  
        y_true = np.array(all_targets[disease_idx]) 
        y_pred_proba = np.array(processed_predictions[disease_idx])  
        
        # 使用ROC曲线和Youden指数寻找最优阈值  
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)  
        
        # 计算Youden指数 (J = 敏感性 + 特异性 - 1 = TPR - FPR)  
        youden_index = tpr - fpr  
        
        # 找到最大Youden指数对应的阈值  
        optimal_idx = np.argmax(youden_index)  
        best_threshold = thresholds[optimal_idx]  

        # 使用最优阈值计算预测结果  
        y_pred = (y_pred_proba > best_threshold).astype(int)  
        
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

    # 使用最佳阈值重新计算所有指标
    tp_sum = np.zeros(18)
    tn_sum = np.zeros(18)
    fp_sum = np.zeros(18)
    fn_sum = np.zeros(18)

    for disease_idx in range(18):
        y_true = np.array(all_targets[disease_idx])
        y_pred_proba = np.array(processed_predictions[disease_idx])
        y_pred = (y_pred_proba > best_thresholds[disease_idx]).astype(int)
        
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
    f1_per_disease = 2 * precision_per_disease * recall_per_disease / (precision_per_disease + recall_per_disease + 1e-8)  

    # 计算每个疾病的 AUROC
    auroc_per_disease = []
    for i in range(18):
        try:
            auroc = roc_auc_score(all_targets[i], processed_predictions[i])
        except ValueError:
            auroc = 0.0
        auroc_per_disease.append(auroc)

    # 更新器官指标计算逻辑
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
        y_true = np.array(all_targets[i])  
        
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
        wandb_log_dict.update(disease_metrics)  

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
        wandb_log_dict.update(organ_wandb_metrics)  

    # 记录到wandb  
    wandb.log(wandb_log_dict, step=epoch)  

    return avg_loss, avg_auroc, best_thresholds  # 返回最佳阈值供后续使用


def calculate_dice_per_organ(pred, target, threshold=0.5):  
    # pred shape: [batch_size, num_classes, D, H, W]  
    pred = (pred > threshold).float()  
    # 计算每个器官的dice，对每个batch分别计算  
    dice_scores = []  
    batch_size = pred.size(0)  
    num_classes = pred.size(1)  
    
    for i in range(num_classes):  
        # 取出当前器官的预测和目标  
        pred_organ = pred[:, i:i+1, ...]  # [batch_size, 1, D, H, W]  
        target_organ = target[:, i:i+1, ...]  # [batch_size, 1, D, H, W]  
        
        intersection = (pred_organ * target_organ).sum(dim=(2, 3, 4))  # [batch_size, 1]  
        union = pred_organ.sum(dim=(2, 3, 4)) + target_organ.sum(dim=(2, 3, 4))  # [batch_size, 1]  
        dice = (2 * intersection + 1e-7) / (union + 1e-7)  # [batch_size, 1]  
        
        # 计算这个器官在整个batch上的平均dice  
        dice_scores.append(dice.mean().item())  
    
    return dice_scores  # 返回一个长度为num_classes的列表，每个元素是对应器官的平均dice分数  

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

# def visualize_segmentation(image, target, pred, slice_idx, organ_names, epoch, save_dir='visualization'):  
#     """  
#     可视化分割结果，显示原始图像、目标分割和预测分割的对比。  
    
#     Args:  
#         image: 形状为[C, D, H, W]的输入图像  
#         target: 形状为[C, D, H, W]的目标分割  
#         pred: 形状为[C, D, H, W]的预测分割  
#         slice_idx: 要显示的切片索引  
#         organ_names: 器官名称列表  
#         epoch: 当前训练轮次  
#         save_dir: 保存可视化结果的目录  
#     """  
#     os.makedirs(save_dir, exist_ok=True)  
#     # 根据维度选择切片  
#     if image.dim() == 5:  # 处理 5D 张量  
#         image = image[0, :, :, :, :]  
#         target = target[0, :, :, :, :]  
#         pred = pred[0, :, :, :, :]  
    
#     # 打印张量形状，帮助调试  
#     # print(f"Image shape: {image.shape}")  
#     # print(f"Target shape: {target.shape}")  
#     # print(f"Pred shape: {pred.shape}")  
    
#     # 定义每个器官的颜色  
#     colors = [  
#         [0, 0.5, 1],      # 蓝色 - lung  
#         [1, 0.6, 0],      # 橙色 - trachea_bronchie  
#         [0, 1, 0.6],      # 青色 - pleura  
#         [1, 0, 0],        # 红色 - mediastinum  
#         [0.5, 0, 1],      # 紫色 - heart  
#         [0.6, 0.3, 0],    # 棕色 - esophagus  
#         [1, 0.4, 0.7],    # 粉色 - bone  
#         [0.5, 0.5, 0.5],  # 灰色 - thyroid  
#         [1, 1, 0],        # 黄色 - abdomen  
#     ]  
    
#     # 创建大图，包含三个子图  
#     plt.figure(figsize=(20, 7))  
    
#     # 1. 显示原始输入图像  
#     plt.subplot(131)  
#     img_slice = image[0, slice_idx].cpu().numpy()  
#     # 归一化到[0,1]范围  
#     img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)  
#     plt.imshow(img_slice, cmap='gray')  
#     plt.title(f'Input Image\nSlice {slice_idx}', fontsize=12)  
#     plt.axis('off')  
    
#     # 2. 显示真实标签  
#     plt.subplot(132)  
#     H, W = target.shape[-2:]  
#     combined_target = np.zeros((H, W, 3))  
    
#     # 创建掩码来跟踪已经填充的区域  
#     filled_mask = np.zeros((H, W))  
    
#     for i in range(len(organ_names)):  
#         mask = target[i, slice_idx].cpu().numpy()  
#         # 只在未填充的区域添加新的颜色  
#         unfilled_mask = mask * (1 - filled_mask)  
#         for c in range(3):  
#             combined_target[:, :, c] += unfilled_mask * colors[i][c]  
#         # 更新已填充区域的掩码  
#         filled_mask = np.maximum(filled_mask, mask)  
    
#     # 确保值在[0,1]范围内  
#     combined_target = np.clip(combined_target, 0, 1)  
#     plt.imshow(combined_target)  
#     plt.title('Ground Truth Segmentation', fontsize=12)  
    
#     # 添加图例  
#     patches = [mpatches.Patch(color=colors[i], label=organ_names[i].replace('_', ' ').title())  
#               for i in range(len(organ_names))]  
#     plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)  
#     plt.axis('off')  
    
#     # 3. 显示预测结果  
#     plt.subplot(133)  
#     combined_pred = np.zeros((H, W, 3))  
#     pred_binary = (pred > 0.5).float()  
    
#     # 重置填充掩码  
#     filled_mask = np.zeros((H, W))  
    
#     for i in range(len(organ_names)):  
#         mask = pred_binary[i, slice_idx].cpu().numpy()  
#         # 只在未填充的区域添加新的颜色  
#         unfilled_mask = mask * (1 - filled_mask)  
#         for c in range(3):  
#             combined_pred[:, :, c] += unfilled_mask * colors[i][c]  
#         # 更新已填充区域的掩码  
#         filled_mask = np.maximum(filled_mask, mask)  
    
#     # 确保值在[0,1]范围内  
#     combined_pred = np.clip(combined_pred, 0, 1)  
#     plt.imshow(combined_pred)  
#     plt.title('Model Prediction', fontsize=12)  
    
#     # 添加图例  
#     patches = [mpatches.Patch(color=colors[i], label=organ_names[i].replace('_', ' ').title())  
#               for i in range(len(organ_names))]  
#     plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)  
#     plt.axis('off')  
    
#     # 调整布局并添加总标题  
#     plt.suptitle(f'Segmentation Results - Epoch {epoch}', fontsize=14, y=1.02)  
#     plt.tight_layout()  
    
#     # 构建保存路径  
#     save_path = os.path.join(  
#         save_dir,   
#         f'segmentation_vis_epoch_{epoch}_slice_{slice_idx}.png'  
#     )  
    
#     # 保存高质量图像  
#     plt.savefig(  
#         save_path,  
#         bbox_inches='tight',  
#         dpi=300,  
#         pad_inches=0.2  
#     )  
#     plt.close()  
    
#     return save_path

def save_nifti(data, filename):  
    """保存数据为 NIfTI 格式"""  
    img = nib.Nifti1Image(data, affine=np.eye(4))  # 使用单位矩阵作为仿射变换  
    nib.save(img, filename)  

def visualize_segmentation(image, target, pred, organ_names, epoch, save_dir='visualization'):  
    """  
    可视化分割结果，显示原始图像、目标分割和预测分割的对比，并保存为 NIfTI 格式。  
    
    Args:  
        image: 形状为[C, D, H, W]的输入图像  
        target: 形状为[C, D, H, W]的目标分割  
        pred: 形状为[C, D, H, W]的预测分割  
        organ_names: 器官名称列表  
        epoch: 当前训练轮次  
        save_dir: 保存可视化结果的目录  
    """  
    os.makedirs(save_dir, exist_ok=True)  
    
    # 根据维度选择切片  
    if image.dim() == 5:  # 处理 5D 张量  
        image = image[0, :, :, :, :]  
        target = target[0, :, :, :, :]  
        pred = pred[0, :, :, :, :]  

    image = image[0, :, :, :]  # 选择第一个通道
    # 保存原始图像  
    image_filename = os.path.join(save_dir, f'original_image_epoch_{epoch}.nii.gz')  
    save_nifti(image.cpu().numpy(), image_filename)  
    
    # 保存目标分割和预测分割为 NIfTI 格式  
    for i, organ_name in enumerate(organ_names):  
        target_filename = os.path.join(save_dir, f'target_{organ_name}_epoch_{epoch}.nii.gz')  
        pred_filename = os.path.join(save_dir, f'pred_{organ_name}_epoch_{epoch}.nii.gz')  
        
        save_nifti(target[i].cpu().numpy(), target_filename)  
        save_nifti((pred[i] > 0.5).float().cpu().numpy(), pred_filename)  

    print(f"Saved original image, target, and prediction NIfTI files for epoch {epoch} in '{save_dir}'.")  

    return save_dir


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


def test_one_epoch(test_loader, model, segmentation_criterion, abnormal_criterion, epoch, logger, config, writer, device, best_thresholds, save_heatmap=True,seg_vis=True):  
    model.eval()  
    loss_list = []  
    seg_loss_list = []  
    abnormal_loss_list = [[] for _ in range(16)]  
    sample_idx = 0  
    np.random.seed(42)  
    
    dataset_size = len(test_loader.dataset)  
    selected_indices = set(np.random.choice(dataset_size, min(40, dataset_size), replace=False))  
    current_dir = config.work_dir  

    if save_heatmap:  
        save_dir = os.path.join(config.work_dir, "test_results")  
        os.makedirs(save_dir, exist_ok=True)  
        
        # 用于存储需要生成热图的数据  
        heatmap_data = []

        vis_samples = []  # 变为列表 
    


    current_seg_weight = calculate_dynamic_weight(  
        epoch=epoch,  
        initial_weight=config.initial_segmentation_weight,  
        total_epochs=config.epochs,  
        decay_rate=config.weight_decay_rate  
    )  
    
    logger.info(f"Test epoch {epoch}: Segmentation weight = {current_seg_weight:.4f}, "  
                f"Abnormal weight = {config.abnormal_loss_weight}")  

    test_loader = tqdm(test_loader, desc=f"Epoch {epoch} Test", leave=True, dynamic_ncols=True)  

    dice_scores = []  
    dice_scores_organ = []
    nsd_scores_organ = []
    tp_sum = np.zeros(16, dtype=int)  
    tn_sum = np.zeros(16, dtype=int)  
    fp_sum = np.zeros(16, dtype=int)  
    fn_sum = np.zeros(16, dtype=int)  

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
        for iter, data in enumerate(test_loader):  
            # if iter>100:
            #     break
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
            # abnormal_targets_np=1-abnormal_targets_np#将阴性作为1
            # abnormal_pred_avg_np=1-abnormal_pred_avg_np 

            for i in range(16):  
                processed_predictions[i].extend(abnormal_pred_avg_np[:, i])  
                all_targets[i].extend(abnormal_targets_np[:, i])  

            for i, disease_loss in enumerate(disease_losses):  
                abnormal_loss_list[i].append(disease_loss.item())  

            abnormal_loss = torch.mean(torch.stack(disease_losses))  
            total_loss = current_seg_weight * seg_loss + config.abnormal_loss_weight * abnormal_loss  

            loss_list.append(total_loss.item())  
            seg_loss_list.append(seg_loss.item())  

            dice_score_organ = calculate_dice(seg_pred, seg_targets) 
            dice_score = dice_score_organ.mean().item()  
            dice_score_organ = dice_score_organ.cpu().numpy()

            dice_scores.append(dice_score)
            dice_scores_organ.append(dice_score_organ)

            # 计算NSD
            nsd_per_sample = calculate_nsd(seg_pred, seg_targets)  # [B, C]
            nsd_scores_organ.append(nsd_per_sample.cpu().numpy())
            # print(f'dice_score:{dice_score_organ}')
            # print(f'dice_score:{dice_score}') 
            # dice_scores.append(dice_score)  

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

                    vis_samples.append({  
                        'image': images[batch_idx].cpu(),  
                        'target': seg_targets[batch_idx].cpu(),  
                        'prediction': seg_pred[batch_idx].cpu() ,
                        'sample_name': sample_names[batch_idx]
                    }) 

    # 将所有样本的Dice系数转换为NumPy数组  
    dice_scores_organ = np.array(dice_scores_organ)  

    # 保存为.npy文件  
    np.save('dice_scores.npy', dice_scores_organ) 

    # 创建保存混淆矩阵的目录  
    confusion_matrix_dir = os.path.join(config.work_dir, "test_results","test_confusion_matrices", f"epoch_{epoch}")  
    os.makedirs(confusion_matrix_dir, exist_ok=True) 
    # 创建保存 ROC 曲线的目录  
    roc_curve_dir = os.path.join(config.work_dir, "test_results", "roc_curves", f"epoch_{epoch}")  
    os.makedirs(roc_curve_dir, exist_ok=True) 

    # 初始化指标数组  
    num_diseases = len(disease_names)  
    tp_sum = np.zeros(num_diseases, dtype=int)  
    tn_sum = np.zeros(num_diseases, dtype=int)  
    fp_sum = np.zeros(num_diseases, dtype=int)  
    fn_sum = np.zeros(num_diseases, dtype=int)  

    precision_per_disease = np.zeros(num_diseases)  
    recall_per_disease = np.zeros(num_diseases)  
    accuracy_per_disease = np.zeros(num_diseases)  
    f1_per_disease = np.zeros(num_diseases)  
    auroc_per_disease = np.zeros(num_diseases)  # 使用 NumPy 数组  

    # 新增列表以收集每个疾病的 y_true 和 y_pred  
    y_true_list = []  
    y_pred_list = [] 

    # 计算所有指标  
    for disease_idx in range(num_diseases):  
        y_true = np.array(all_targets[disease_idx])  
        y_pred_proba = np.array(processed_predictions[disease_idx])  
        y_pred = (y_pred_proba > best_thresholds[disease_idx]).astype(int)  

        # 保存 y_true 和 y_pred_proba，文件名中包含疾病名称  
        y_true_path = os.path.join(roc_curve_dir, f'y_true_{disease_names[disease_idx]}.npy')  
        y_pred_proba_path = os.path.join(roc_curve_dir, f'y_pred_proba_{disease_names[disease_idx]}.npy')  
        
        np.save(y_true_path, y_true)  
        np.save(y_pred_proba_path, y_pred_proba) 

        # 收集 y_true 和 y_pred  
        y_true_list.append(y_true)  
        y_pred_list.append(y_pred)

        # 计算混淆矩阵并保存  
        cm = confusion_matrix(y_true, y_pred)  
        save_path = plot_confusion_matrix(y_true, y_pred, disease_names[disease_idx], confusion_matrix_dir)  

        # 从混淆矩阵中提取TP、TN、FP、FN  
        tp = cm[1, 1]  # True Positive  
        tn = cm[0, 0]  # True Negative  
        fp = cm[0, 1]  # False Positive  
        fn = cm[1, 0]  # False Negative  

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
        auroc_per_disease[disease_idx] = auroc  # 将AUROC值存储到NumPy数组中  

        # 计算每个疾病的指标  
        precision_per_disease[disease_idx] = tp / (tp + fp + 1e-8)  
        recall_per_disease[disease_idx] = tp / (tp + fn + 1e-8)  
        accuracy_per_disease[disease_idx] = (tp + tn) / (tp + tn + fp + fn + 1e-8)  
        f1_per_disease[disease_idx] = 2 * precision_per_disease[disease_idx] * recall_per_disease[disease_idx] / (precision_per_disease[disease_idx] + recall_per_disease[disease_idx] + 1e-8)

        # 计算并绘制每种疾病的 ROC 曲线  
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)  
        roc_auc = auc(fpr, tpr)  

        plt.figure()  
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))  
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # 参考线  
        plt.xlim([0.0, 1.0])  
        plt.ylim([0.0, 1.05])  
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate')  
        plt.title(f'ROC Curve for {disease_names[disease_idx]}')  
        plt.legend(loc='lower right')  
        
        # 保存 ROC 曲线图  
        roc_curve_path = os.path.join(roc_curve_dir, f'roc_curve_disease_{disease_idx}.png')  
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

    # 计算每个器官的指标  
    for organ_idx, (organ, disease_indices) in enumerate(organ_disease_mapping.items()):  
        # 收集该器官所有疾病的预测和真实值  
        organ_predictions = []  
        organ_targets = []  
        organ_pred_proba = []  # 新增用于存储预测概率  
        # 初始化TP、TN、FP、FN  
        organ_tp = organ_tn = organ_fp = organ_fn = 0   

        # 收集该器官所有疾病的预测和真实值，并计算TP、TN、FP、FN  
        for disease_idx in disease_indices:  
            organ_targets.extend(y_true_list[disease_idx])  
            organ_predictions.extend(y_pred_list[disease_idx])  # 经过阈值处理的预测值  
            organ_pred_proba.extend(processed_predictions[disease_idx])  # 原始预测概率  
            
            
            organ_tp += tp_sum[disease_idx]  
            organ_tn += tn_sum[disease_idx]  
            organ_fp += fp_sum[disease_idx]  
            organ_fn += fn_sum[disease_idx]  

        # 更新器官的总和  
        organ_tp_sum[organ_idx] = organ_tp  
        organ_tn_sum[organ_idx] = organ_tn  
        organ_fp_sum[organ_idx] = organ_fp  
        organ_fn_sum[organ_idx] = organ_fn 


        # 计算该器官的AUROC  
        organ_auroc_value = roc_auc_score(organ_targets, organ_pred_proba) if organ_targets else 0.0  
        organ_auroc[organ_idx] = organ_auroc_value  

        # 计算器官级别的指标  
        total = organ_tp + organ_fp + 1e-8  
        organ_precision[organ_idx] = organ_tp / total  
        organ_recall[organ_idx] = organ_tp / (organ_tp + organ_fn + 1e-8)  
        organ_accuracy[organ_idx] = (organ_tp + organ_tn) / (organ_tp + organ_tn + organ_fp + organ_fn + 1e-8)  
        organ_f1[organ_idx] = 2 * organ_precision[organ_idx] * organ_recall[organ_idx] / (organ_precision[organ_idx] + organ_recall[organ_idx] + 1e-8)  

        # 绘制混淆矩阵并保存  
        save_path, cm = plot_confusion_matrix(organ_targets, organ_predictions, organ, confusion_matrix_dir)  

        # 计算并绘制该器官的 ROC 曲线  
        fpr, tpr, _ = roc_curve(organ_targets, organ_pred_proba)  
        organ_roc_auc = auc(fpr, tpr)  

        plt.figure()  
        plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = {:.2f})'.format(organ_roc_auc))  
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # 参考线  
        plt.xlim([0.0, 1.0])  
        plt.ylim([0.0, 1.05])  
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate')  
        plt.title(f'ROC Curve for {organ}')  
        plt.legend(loc='lower right')  
        
        # 保存 ROC 曲线图  
        organ_roc_curve_path = os.path.join(roc_curve_dir, f'roc_curve_organ_{organ}.png')  
        plt.savefig(organ_roc_curve_path)  
        plt.close()  

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
 

    # 记录平均指标  
    confusion_matrices = {}  
    log_info = f"Test Epoch {epoch} Summary:\n"  
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

    log_info += "\nPer-Disease Metrics and Confusion Matrices:"  
    for i, disease in enumerate(disease_names):  
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
        log_info += f"TN: {tn_sum[i]}, FP: {fp_sum[i]}\n"  
        log_info += f"FN: {fn_sum[i]}, TP: {tp_sum[i]}\n"  

    # 计算并保存每个器官的混淆矩阵  
    # 后处理
    dice_scores_organ = np.concatenate(dice_scores_organ, axis=0)  # [N, C]
    nsd_scores_organ = np.concatenate(nsd_scores_organ, axis=0)    # [N, C]
    
    avg_dice_per_organ = dice_scores_organ.mean(axis=0)
    avg_nsd_per_organ = nsd_scores_organ.mean(axis=0)
    # 在日志中添加报告
    log_info += "\nOrgan-wise Segmentation Metrics:\n"
    for idx, name in enumerate(organ_names):
        log_info += (f"{name}: Dice={avg_dice_per_organ[idx]:.4f} ± {dice_scores_organ[:, idx].std():.4f}, "
                     f"NSD={avg_nsd_per_organ[idx]:.4f} ± {nsd_scores_organ[:, idx].std():.4f}\n")
        
    log_info += "\n\nOrgan-level Confusion Matrices:"  
    for organ_idx, (organ, disease_indices) in enumerate(organ_disease_mapping.items()):  
        # 添加到日志信息  
        log_info += f"\n{organ}:\n"  
        log_info += f"AUROC: {organ_auroc[organ_idx]:.4f}, "  
        log_info += f"Accuracy: {organ_accuracy[organ_idx]:.4f}, "  
        log_info += f"Precision: {organ_precision[organ_idx]:.4f}, "  
        log_info += f"Recall: {organ_recall[organ_idx]:.4f}, "  
        log_info += f"F1: {organ_f1[organ_idx]:.4f}\n"  

        # 使用记录的TP、TN、FP、FN构建混淆矩阵  
        tn = organ_tn_sum[organ_idx]  
        fp = organ_fp_sum[organ_idx]  
        fn = organ_fn_sum[organ_idx]  
        tp = organ_tp_sum[organ_idx]  

        log_info += f"Confusion Matrix:\n"  
        log_info += f"TN: {tn}, FP: {fp}\n"  
        log_info += f"FN: {fn}, TP: {tp}\n"

    # 计算整体指标
    total_tn = sum(organ_tn_sum)  
    total_fp = sum(organ_fp_sum)  
    total_fn = sum(organ_fn_sum)  
    total_tp = sum(organ_tp_sum)  
    total_cm = np.array([[total_tn, total_fp], [total_fn, total_tp]], dtype=int)  

    # 保存总体混淆矩阵   
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
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-8)   

    log_info += f"\nOverall Metrics from Confusion Matrix:\n"  
    log_info += f"Accuracy: {total_accuracy:.4f}\n"  
    log_info += f"Precision: {total_precision:.4f}\n"  
    log_info += f"Recall: {total_recall:.4f}\n"  
    log_info += f"F1 Score: {total_f1:.4f}\n" 

    # 在找到最佳阈值后，生成和保存热图  
    if save_heatmap and heatmap_data:  
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

            # 分割可视化  
            depth = data['images'].shape[1]  # 获取深度维度的大小  
            middle_slice = depth // 2  # 计算中间切片的索引  
        
        # 分割可视化  
        for vis_sample in vis_samples:  # 遍历 vis_samples 中的每个样本  
            vis_path = visualize_segmentation(  
                vis_sample['image'],      # [C, D, H, W]  
                vis_sample['target'],     # [C, D, H, W]  
                vis_sample['prediction'], # [C, D, H, W]  
                organ_names,  
                epoch,  
                save_dir=os.path.join(save_dir, f'segmentation_visualizations/sample_{vis_sample["sample_name"]}')  # 保存路径  
            )  
            logger.info(f"Successfully saved segmentation visualization for sample {vis_sample['sample_name']} at epoch {epoch}")  

            # vis_path = visualize_segmentation(  
            #     data['images'],      # [C, D, H, W]  
            #     seg_targets,     # [C, D, H, W]  
            #     data['segmentation_preds'], # [C, D, H, W]  
            #     middle_slice,  
            #     organ_names,  
            #     epoch,  
            #     save_dir=os.path.join(save_dir, f'segmentation_visualizations/sample_{data["sample_name"]}')  # 保存路径  
            # )  
            # logger.info(f"Successfully saved segmentation visualization for sample {data['sample_name']} at epoch {epoch}")  

    print(log_info)  
    logger.info(log_info)  

    return avg_loss  # 返回测试损失