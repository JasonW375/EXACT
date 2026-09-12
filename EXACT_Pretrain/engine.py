import os
import math
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from tqdm import tqdm
import wandb



def calculate_nsd(pred, target, threshold=0.5, tau=1.0):
    """Normalised surface Dice (NSD).

    :param pred: predicted segmentation mask, shape [B, C, D, H, W]
    :param target: ground-truth segmentation mask, shape [B, C, D, H, W]
    :param threshold: binarisation threshold
    :param tau: surface distance tolerance, in voxels
    :return: NSD, shape [B, C]
    """
    pred = (pred > threshold).float()
    target = target.float()
    batch_size, num_organs = pred.shape[0], pred.shape[1]
    nsd_per_organ = torch.zeros((batch_size, num_organs), device=pred.device)
    
    for b in range(batch_size):
        for organ_idx in range(num_organs):
            pred_organ = pred[b, organ_idx].cpu().numpy()
            target_organ = target[b, organ_idx].cpu().numpy()
            
            # Extract the surface voxels.
            def get_surface(mask):
                if not np.any(mask):
                    return np.zeros_like(mask, dtype=bool)
                eroded = binary_erosion(mask, structure=np.ones((3,3,3)))
                return mask & ~eroded
            
            surface_pred = get_surface(pred_organ > 0.5)
            surface_target = get_surface(target_organ > 0.5)
            
            surface_pred_coords = np.argwhere(surface_pred)
            surface_target_coords = np.argwhere(surface_target)
            
            # An empty surface on both sides counts as a perfect match.
            if len(surface_pred_coords) + len(surface_target_coords) == 0:
                nsd = 1.0
            elif len(surface_pred_coords) == 0 or len(surface_target_coords) == 0:
                nsd = 0.0
            else:
                dist_map_target = distance_transform_edt(~surface_target)
                dist_map_pred = distance_transform_edt(~surface_pred)

                # Surface voxels lying within tau of the other surface.
                pred_dists = dist_map_target[tuple(surface_pred_coords.T)]
                target_dists = dist_map_pred[tuple(surface_target_coords.T)]
                tp_a = np.sum(pred_dists <= tau)
                tp_b = np.sum(target_dists <= tau)
                
                nsd = (tp_a + tp_b) / (len(surface_pred_coords) + len(surface_target_coords))
                
            nsd_per_organ[b, organ_idx] = nsd
            
    return nsd_per_organ

def weighted_binary_cross_entropy(pred, target, weight_pos, weight_neg):
    """Class-weighted binary cross entropy.

    `pred` is expected to be already activated, so no sigmoid is applied here.

    pred: predictions, shape (B,)
    target: ground-truth labels, shape (B,)
    weight_pos: weight on positive samples
    weight_neg: weight on negative samples
    """
    loss_pos = weight_pos * target * torch.log(pred + 1e-6)
    loss_neg = weight_neg * (1 - target) * torch.log(1 - pred + 1e-6)
    loss = -(loss_pos + loss_neg)
    return loss

def get_organ_disease_mapping():  
    """Map each disease channel to the organ channel it is scored inside.

    Organ channel order:
    0: lung
    1: trachea and bronchie
    2: pleura
    3: mediastinum
    4: heart
    5: esophagus
    6: global

    Disease channel order:
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
    """Multi-scale abnormality detection loss.

    Args:
        seg_pred: segmentation prediction (B, 7, D, H, W)
        abnormal_preds: per-scale disease predictions, each (B, 18, D, H, W)
        abnormal_targets: disease labels (B, 18)
        disease_frequencies: positive-sample frequency of each of the 18 diseases
        k: top-k size at the coarsest scale
        epsilon: numerical stability constant
        seg_threshold: threshold used to binarise the segmentation mask
    """
    B, _, D, H, W = seg_pred.shape  
    num_diseases = abnormal_preds[0].shape[1]
    num_scales = len(abnormal_preds)
    
    # Map each disease to the organ it is scored inside.
    disease_to_organ = get_organ_disease_mapping()  
    
    # Binarise the segmentation prediction.
    seg_mask = (seg_pred > seg_threshold).float()  
    
    # One list per scale.
    disease_losses = [[] for _ in range(num_scales)]  
    disease_predictions = [[] for _ in range(num_scales)]  
    
    for scale_idx, abnormal_pred in enumerate(abnormal_preds):  
        # k grows by a factor of 8 per scale, tracking the voxel count.
        current_k = k * (8 ** scale_idx)
        # Bring the segmentation mask to the current scale.
        current_size = abnormal_pred.shape[2:]  
        scaled_seg_mask = F.interpolate(seg_mask, size=current_size, mode='trilinear', align_corners=False)  
        
        for disease_idx in range(num_diseases):  
            # Organ channel for this disease.
            organ_idx = disease_to_organ[disease_idx]  
            
            # Restrict the disease prediction to that organ.
            organ_mask = scaled_seg_mask[:, organ_idx:organ_idx+1]  
            disease_pred = abnormal_pred[:, disease_idx:disease_idx+1]  
            
            final_pred = organ_mask * disease_pred  
            
            # Average the top-k responses.
            top_k_values, _ = torch.topk(final_pred.view(B, -1), current_k, dim=1)  
            avg_top_k = top_k_values.mean(dim=1)  
            
            # Weight positives by inverse prevalence.
            freq = disease_frequencies[disease_idx]  
            weight_pos = (1 - freq + epsilon) / (freq + epsilon)  
            weight_neg = 1.0  
            
            # Loss for this disease at this scale.
            target = abnormal_targets[:, disease_idx]  
            loss = weighted_binary_cross_entropy(avg_top_k, target, weight_pos, weight_neg)  
            loss = loss.mean()  
            
            disease_losses[scale_idx].append(loss)  
            disease_predictions[scale_idx].append(avg_top_k)  
    
    # Per-scale losses and predictions.
    scale_losses = [torch.stack(losses) for losses in disease_losses]  
    scale_predictions = [torch.stack(preds, dim=1) for preds in disease_predictions]  
    
    return scale_losses, scale_predictions



def calculate_dynamic_weight(epoch, initial_weight, total_epochs, decay_rate=9.0):  
    """Exponentially decaying loss weight.

    :param epoch: current epoch
    :param initial_weight: weight at epoch 0
    :param total_epochs: total number of epochs
    :param decay_rate: decay rate
    :return: weight for this epoch
    """
    k = decay_rate / total_epochs  
    weight =initial_weight * math.exp(-k * epoch) 
    weight = max(weight, 0.5)
    return weight


def calculate_dice(pred, target, threshold=0.5):
    # Mean Dice per organ, averaged over the batch.
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2 * intersection + 1e-7) / (union + 1e-7)  # guard against divide-by-zero
    # Average over the batch dimension.
    return dice.mean(dim=0)

def train_one_epoch(train_loader, model, segmentation_criterion, abnormal_criterion, optimizer, scheduler, epoch, step, logger, config, writer, device):  
    model.train()  
    loss_list = []  
    seg_loss_list = []  
    # One disease-loss list per scale: 2 scales x 18 diseases.
    scale_disease_loss_list = [[[] for _ in range(18)] for _ in range(2)]

    # One organ-dice list per scale: 2 scales x 7 channels (6 organs + global).
    scale_organ_dice_scores = [
        [[] for _ in range(7)]  for _ in range(2)
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
    
    # Organ names, including the global channel.
    organ_names = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", "esophagus", "global"]  
    
    # Disease names.
    disease_names = [
        "Medical material", "Arterial wall calcification",
        "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
        "Bronchiectasis", "Interlobular septal thickening"  
    ]  

    # Positive-sample frequency of each disease.
    disease_frequencies = [  
        0.102, 0.2837,
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

        seg_pred, abnormal_preds = model(images)  # abnormal_preds is a per-scale list

        seg_loss = segmentation_criterion(seg_pred, seg_targets)  
        scale_losses, scale_predictions = Abnormal_loss_multiscale(seg_pred, abnormal_preds, abnormal_targets, disease_frequencies)  
        
        # Record the loss of every disease at every scale.
        for scale_idx, scale_loss in enumerate(scale_losses):  
            for disease_idx, disease_loss in enumerate(scale_loss):  
                scale_disease_loss_list[scale_idx][disease_idx].append(disease_loss.item())  

        # Total abnormality loss, averaged over scales.
        abnormal_loss = torch.mean(torch.stack([torch.mean(losses) for losses in scale_losses]))  
        total_loss = current_seg_weight * seg_loss + config.abnormal_loss_weight * abnormal_loss  

        total_loss.backward()  
        optimizer.step()  

        loss_list.append(total_loss.item())  
        seg_loss_list.append(seg_loss.item())  

        # Dice at each scale.
        for scale_idx, abnormal_pred in enumerate(abnormal_preds):  
            # Bring prediction and target to the current scale.
            current_size = abnormal_pred.shape[2:]  
            scaled_seg_pred = F.interpolate(seg_pred, size=current_size, mode='trilinear', align_corners=False)  
            scaled_seg_target = F.interpolate(seg_targets, size=current_size, mode='trilinear', align_corners=False)  
            
            # Dice per organ, including the global channel.
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

    # Epoch averages.
    avg_loss = np.mean(loss_list)  
    avg_seg_loss = np.mean(seg_loss_list)  
    
    # Mean disease loss per scale.
    avg_scale_disease_losses = [  
        [np.mean(losses) for losses in scale_losses]   
        for scale_losses in scale_disease_loss_list  
    ]  
    
    # Mean organ dice per scale.
    avg_scale_organ_dice = [  
        [np.mean(scores) for scores in scale_dices]  
        for scale_dices in scale_organ_dice_scores  
    ]  

    # Log training metrics to wandb.
    wandb_metrics = {  
        "train/total_loss": avg_loss,  
        "train/seg_loss": avg_seg_loss,  
        "train/abnormal_loss": np.mean([np.mean(losses) for losses in avg_scale_disease_losses]),  
    }  

    # Per-scale, per-organ Dice.
    for scale_idx in range(len(avg_scale_organ_dice)):  
        for organ_idx, organ_name in enumerate(organ_names):  
            wandb_metrics[f"train/scale{scale_idx+1}_dice_{organ_name}"] = avg_scale_organ_dice[scale_idx][organ_idx]  

    # Per-scale, per-disease loss.
    for scale_idx in range(len(avg_scale_disease_losses)):  
        for disease_idx, disease_name in enumerate(disease_names):  
            wandb_metrics[f"train/scale{scale_idx+1}_disease_loss_{disease_name}"] = avg_scale_disease_losses[scale_idx][disease_idx]  

    # Send everything to wandb.
    wandb.log(wandb_metrics, step=epoch)  

    if scheduler is not None:  
        scheduler.step()  

    step += len(train_loader)  
    return step


def upsample_3d(tensor, target_size):  
    """  
    Upsample a 3D tensor to the target size.

    Args:
        tensor (torch.Tensor): input, shape [B, C, D, H, W]
        target_size (tuple): target size (D, H, W)

    Returns:
        torch.Tensor: the upsampled tensor
    """
    # Trilinear interpolation.
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
    # Unpack the multi-scale predictions.
    low_res_preds, high_res_preds = predictions  
    
    # Target size, i.e. the high-resolution grid.
    _, _, target_depth, target_height, target_width = high_res_preds.shape  
    
    # Upsample the low-resolution prediction.
    low_res_upsampled = upsample_3d(low_res_preds, (target_depth, target_height, target_width))  

    # Use the high-resolution prediction.
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
    
    # Extract the segmentation prediction.
    seg_pred = segmentation_preds[0].cpu().numpy()  # [7, D, H, W]  
    
    # Build the 3D segmentation mask.
    seg_mask = (seg_pred > seg_threshold).astype(np.float32)  
    target = targets[0].cpu().numpy()  # [18]  
    original_image = images[0, 0].cpu().numpy()  # [D, H, W]  
    
    # Save the original image.
    affine = np.eye(4)  
    original_nifti = nib.Nifti1Image(original_image, affine)  
    original_nifti.header['descrip'] = f'Original 3D Image, Epoch: {epoch}'  
    original_save_path = sample_dir / f"original_image.nii.gz"  
    nib.save(original_nifti, original_save_path)  
    
    # Disease names.
    disease_names = [
        "Medical material", "Arterial wall calcification",
        "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
        "Bronchiectasis", "Interlobular septal thickening"  
    ]  
    
    # Disease-to-organ mapping.
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
    
    # Write out the prediction summary.
    info_file = sample_dir / "prediction_info.txt"  
    with open(info_file, "w") as f:  
        f.write(f"Epoch: {epoch}\n")  
        f.write(f"Sample: {sample_idx}\n")  
        f.write(f"Abnormal Detection Parameters: top-{topk}\n")  
        f.write("Disease Predictions:\n")  
        
        # Predictions for each scale.
        prediction_results = {  
            "High-Res": high_res_pred,  
            "Low-Res": low_res_pred  
        }  
        
        # Iterate over the resolutions.
        for res_name, pred in prediction_results.items():  
            f.write(f"\n{res_name} Predictions:\n")  
            
            # One disease at a time.
            for disease_idx, disease_name in enumerate(disease_names):  
                # 3D prediction map for this disease.
                disease_pred = pred[disease_idx]  # [D, H, W]
                disease_label = int(target[disease_idx])
                
                # Segmentation mask of the associated organ.
                related_organs = disease_organ_mapping[disease_name]  
                combined_mask = np.zeros_like(seg_mask[0])  
                for organ_name in related_organs:  
                    organ_idx = organ_names.index(organ_name)  
                    combined_mask = np.maximum(combined_mask, seg_mask[organ_idx])  
                
                # Restrict the prediction to that organ.
                combined_pred = disease_pred * combined_mask  
                
                # Mean of the top-k responses.
                topk_mean = np.mean(np.sort(combined_pred.flatten())[-topk:])  
                
                # Best threshold for this disease.
                current_threshold = abnormal_threshold[disease_idx] if abnormal_threshold is not None else 0.5  
                pred_abnormal = int(topk_mean > current_threshold)  
                
                # Write the prediction line.
                f.write(f"\n  {disease_name}:\n")  
                f.write(f"    Ground Truth: {disease_label}\n")  
                f.write(f"    Prediction: {pred_abnormal}\n")  
                f.write(f"    Top-{topk} Mean: {topk_mean:.4f}\n")  
                f.write(f"    Threshold: {current_threshold:.4f}\n")  
                f.write(f"    Related Organs: {', '.join(related_organs)}\n")  
                
                # Encode the key facts in the file name.
                result_str = f"GT{disease_label}_PD{pred_abnormal}"  
                
                # Save the 3D prediction as NIfTI.
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
    Abnormality detection loss.

    Args:
        seg_pred: segmentation prediction (B, 7, D, H, W)
        abnormal_pred: disease prediction (B, 18, D, H, W)
        abnormal_targets: disease labels (B, 18)
        disease_frequencies: positive-sample frequency of each disease
        k: top-k size
        epsilon: numerical stability constant
        seg_threshold: threshold used to binarise the segmentation mask
    """  
    B, _, D, H, W = seg_pred.shape  
    num_diseases = abnormal_pred.shape[1]
    
    # Map each disease to the organ it is scored inside.
    disease_to_organ = get_organ_disease_mapping()  
    
    # Binarise the segmentation prediction.
    seg_mask = (seg_pred > seg_threshold).float()  
    
    disease_losses = []  
    disease_predictions = []  
    
    for disease_idx in range(num_diseases):  
        # Organ channel for this disease.
        organ_idx = disease_to_organ[disease_idx]  
        
        # Restrict the disease prediction to that organ.
        organ_mask = seg_mask[:, organ_idx:organ_idx+1]  
        disease_pred = abnormal_pred[:, disease_idx:disease_idx+1]  
        
        final_pred = organ_mask * disease_pred  
        
        # Average the top-k responses.
        top_k_values, _ = torch.topk(final_pred.view(B, -1), k, dim=1)  
        avg_top_k = top_k_values.mean(dim=1)  
        
        # Validation runs unweighted.
        freq = disease_frequencies[disease_idx]  
        weight_pos = 1.0
        weight_neg = 1.0  
        
        # Loss for this disease.
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
    # One entry per disease channel.
    abnormal_loss_list = [[] for _ in range(18)]
    sample_idx = 0
    np.random.seed(42)
    
    # Threshold search, one threshold per disease.
    threshold_candidates = np.arange(0.1, 0.9, 0.05)
    best_thresholds = np.array([0.5] * 18)
    best_f1_scores = np.array([-1] * 18)
    
    dataset_size = len(valid_loader.dataset)
    selected_indices = set(np.random.choice(dataset_size, min(40, dataset_size), replace=False))
    current_dir = config.work_dir

    if save_heatmap:
        save_dir = os.path.join(config.work_dir, "validation_results")
        os.makedirs(save_dir, exist_ok=True)
        
        # Data kept aside for heatmap rendering.
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
    tp_sum = np.zeros(18)
    tn_sum = np.zeros(18)
    fp_sum = np.zeros(18)
    fn_sum = np.zeros(18)

    # Post-processed predictions, one list per disease.
    processed_predictions = [[] for _ in range(18)]
    all_targets = [[] for _ in range(18)]

    # Organ names.
    organ_names = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", "esophagus", "global"]  

    # Disease names.
    disease_names = [
        "Medical material", "Arterial wall calcification",
        "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
        "Bronchiectasis", "Interlobular septal thickening"  
    ]

    # Organ-to-disease mapping.
    organ_disease_mapping = {
        "global": [0, 1],
        "lung": [7, 8, 9, 10, 11, 13, 15,17],  
        "heart": [2, 3, 4],  
        "pleura": [12],  
        "mediastinum": [6],  
        "esophagus": [5],  
        "trachea and bronchie": [14, 16]  
    }

    # Positive-sample frequency of each disease.
    disease_frequencies = [
        0.102, 0.2837,
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
            abnormal_pred = abnormal_preds[-1]  # output of the finest scale
            
            seg_loss = segmentation_criterion(seg_pred, seg_targets)
            disease_losses, abnormal_pred_avg = Abnormal_loss(seg_pred, abnormal_pred, abnormal_targets, disease_frequencies)
            
            # Collect post-processed predictions and ground truth.
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

            # Dice over the 7 channels.
            dice_score_organ = calculate_dice(seg_pred, seg_targets)
            dice_score = dice_score_organ.mean().item()  
            dice_score_organ = dice_score_organ.cpu().numpy()

            dice_scores.append(dice_score)
            dice_scores_organ.append(dice_score_organ)

            # Stash what the heatmaps will need.
            if save_heatmap and iter in selected_indices:
                for batch_idx in range(images.size(0)):
                    heatmap_data.append({
                        'predictions': [  
                            abnormal_preds[0][batch_idx:batch_idx+1].cpu(),  # low resolution
                            abnormal_preds[1][batch_idx:batch_idx+1].cpu()   # high resolution
                        ], 
                        'segmentation_preds': seg_pred[batch_idx:batch_idx+1].cpu(),
                        'targets': abnormal_targets[batch_idx:batch_idx+1].cpu(),
                        'images': images[batch_idx:batch_idx+1].cpu(),
                        'sample_name': sample_names[batch_idx]
                    })

    # Stack the per-sample Dice coefficients.
    dice_scores_organ = np.array(dice_scores_organ)  

    # Save as .npy.
    np.save('dice_scores.npy', dice_scores_organ) 

    # Threshold search, one disease at a time.
    logger.info("Finding optimal thresholds for each disease using Youden's index...")  
    for disease_idx in range(18):  
        y_true = np.array(all_targets[disease_idx]) 
        y_pred_proba = np.array(processed_predictions[disease_idx])  
        
        # Pick the threshold maximising the Youden index on the ROC curve.
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)  
        
        # Youden index J = sensitivity + specificity - 1 = TPR - FPR
        youden_index = tpr - fpr  
        
        # Threshold at the maximum.
        optimal_idx = np.argmax(youden_index)  
        best_threshold = thresholds[optimal_idx]  

        # Apply the chosen threshold.
        y_pred = (y_pred_proba > best_threshold).astype(int)  
        
        # Performance at that threshold.
        tp = np.sum((y_true == 1) & (y_pred == 1))  
        fp = np.sum((y_true == 0) & (y_pred == 1))  
        fn = np.sum((y_true == 1) & (y_pred == 0))  
        
        # F1 score.
        precision = tp / (tp + fp + 1e-8)  
        recall = tp / (tp + fn + 1e-8)  
        f1 = 2 * precision * recall / (precision + recall + 1e-8)  
        
        best_thresholds[disease_idx] = best_threshold  
        best_f1_scores[disease_idx] = f1  
        
        # Keep the threshold and its metrics.
        logger.info(f"{disease_names[disease_idx]}: "  
                    f"Best threshold = {best_threshold:.3f}, "  
                    f"F1 = {f1:.4f}, "  
                    f"Sensitivity = {tpr[optimal_idx]:.4f}, "  
                    f"Specificity = {1-fpr[optimal_idx]:.4f}, "  
                    f"Youden index = {youden_index[optimal_idx]:.4f}")  

    # Recompute every metric at the chosen thresholds.
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

    # Save the chosen thresholds.
    threshold_save_path = os.path.join(config.work_dir, f'best_thresholds_epoch_{epoch}.npy')
    np.save(threshold_save_path, best_thresholds)

    # Per-disease metrics.
    precision_per_disease = tp_sum / (tp_sum + fp_sum + 1e-8)
    recall_per_disease = tp_sum / (tp_sum + fn_sum + 1e-8)
    accuracy_per_disease = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum + 1e-8)
    f1_per_disease = 2 * precision_per_disease * recall_per_disease / (precision_per_disease + recall_per_disease + 1e-8)  

    # Per-disease AUROC.
    auroc_per_disease = []
    for i in range(18):
        try:
            auroc = roc_auc_score(all_targets[i], processed_predictions[i])
        except ValueError:
            auroc = 0.0
        auroc_per_disease.append(auroc)

    # Organ-level metrics.
    organ_metrics = {}
    for organ, disease_indices in organ_disease_mapping.items():
        # Gather predictions and labels for every disease of this organ.
        organ_predictions = []
        organ_targets = []
        for disease_idx in disease_indices:
            organ_predictions.extend(processed_predictions[disease_idx])
            organ_targets.extend(all_targets[disease_idx])
        
        # Organ AUROC.
        try:
            organ_auroc = roc_auc_score(organ_targets, organ_predictions)
        except ValueError:
            organ_auroc = 0.0
            
        # Remaining organ metrics, at the chosen thresholds.
        organ_tp = sum(tp_sum[i] for i in disease_indices)
        organ_tn = sum(tn_sum[i] for i in disease_indices)
        organ_fp = sum(fp_sum[i] for i in disease_indices)
        organ_fn = sum(fn_sum[i] for i in disease_indices)
        
        # Organ-level summary.
        organ_precision = organ_tp / (organ_tp + organ_fp + 1e-8)
        organ_recall = organ_tp / (organ_tp + organ_fn + 1e-8)
        organ_accuracy = (organ_tp + organ_tn) / (organ_tp + organ_tn + organ_fp + organ_fn + 1e-8)
        organ_f1 = 2 * organ_precision * organ_recall / (organ_precision + organ_recall + 1e-8)  
        
        # Store the organ metrics.
        organ_metrics[organ] = {
            'auroc': organ_auroc,
            'precision': organ_precision,
            'recall': organ_recall,
            'accuracy': organ_accuracy,
            'f1': organ_f1
        }

    # Averages.
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

    # Directory for the confusion matrices.
    confusion_matrix_dir = os.path.join(config.work_dir, "validation_confusion_matrices", f"epoch_{epoch}")
    os.makedirs(confusion_matrix_dir, exist_ok=True)

    # Per-disease confusion matrices.
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
        
        # Append to the log.
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

    # Per-organ confusion matrices.
    log_info += "\n\nOrgan-level Confusion Matrices:"  
    for organ, disease_indices in organ_disease_mapping.items():  
        organ_predictions = []  
        organ_targets = []  
        for disease_idx in disease_indices:  
            organ_predictions.extend((np.array(processed_predictions[disease_idx]) > best_thresholds[disease_idx]).astype(int))  
            organ_targets.extend(all_targets[disease_idx])  
        
        save_path, cm = plot_confusion_matrix(organ_targets, organ_predictions, organ, confusion_matrix_dir)  
        
        # Append to the log.
        log_info += f"\n{organ}:\n"  
        log_info += f"AUROC: {organ_metrics[organ]['auroc']:.4f}, "  
        log_info += f"Accuracy: {organ_metrics[organ]['accuracy']:.4f}, "  
        log_info += f"Precision: {organ_metrics[organ]['precision']:.4f}, "  
        log_info += f"Recall: {organ_metrics[organ]['recall']:.4f}, "  
        log_info += f"F1: {organ_metrics[organ]['f1']:.4f}\n"  
        log_info += f"Confusion Matrix:\n"  
        log_info += f"TN: {cm[0,0]}, FP: {cm[0,1]}\n"  
        log_info += f"FN: {cm[1,0]}, TP: {cm[1,1]}\n"  

    # Overall confusion-matrix statistics.
    total_tn = sum(cm[0,0] for cm in confusion_matrices.values())  
    total_fp = sum(cm[0,1] for cm in confusion_matrices.values())  
    total_fn = sum(cm[1,0] for cm in confusion_matrices.values())  
    total_tp = sum(cm[1,1] for cm in confusion_matrices.values())  
    
        # Save the overall confusion matrix.
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

    # Overall metrics.
    total_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)  
    total_precision = total_tp / (total_tp + total_fp + 1e-8)  
    total_recall = total_tp / (total_tp + total_fn + 1e-8)  
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-8)
    
    log_info += f"\nOverall Metrics from Confusion Matrix:\n"  
    log_info += f"Accuracy: {total_accuracy:.4f}\n"  
    log_info += f"Precision: {total_precision:.4f}\n"  
    log_info += f"Recall: {total_recall:.4f}\n"  
    log_info += f"F1 Score: {total_f1:.4f}\n"  

    # Render the heatmaps now that the thresholds are known.
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

    # wandb logging.
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

    # Per-disease metrics, thresholds and confusion matrices.
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

    # Organ metrics and confusion matrices.
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

    # Send to wandb.
    wandb.log(wandb_log_dict, step=epoch)  

    return avg_loss, avg_auroc, best_thresholds


def calculate_dice_per_organ(pred, target, threshold=0.5):  
    # pred shape: [batch_size, num_classes, D, H, W]  
    pred = (pred > threshold).float()  
    # Dice per organ, computed per batch element.
    dice_scores = []  
    batch_size = pred.size(0)  
    num_classes = pred.size(1)  
    
    for i in range(num_classes):  
        # Prediction and target for this organ.
        pred_organ = pred[:, i:i+1, ...]  # [batch_size, 1, D, H, W]  
        target_organ = target[:, i:i+1, ...]  # [batch_size, 1, D, H, W]  
        
        intersection = (pred_organ * target_organ).sum(dim=(2, 3, 4))  # [batch_size, 1]  
        union = pred_organ.sum(dim=(2, 3, 4)) + target_organ.sum(dim=(2, 3, 4))  # [batch_size, 1]  
        dice = (2 * intersection + 1e-7) / (union + 1e-7)  # [batch_size, 1]  
        
        # Mean dice for this organ over the batch.
        dice_scores.append(dice.mean().item())  
    
    return dice_scores  # one mean dice per organ, length num_classes

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
    
    # Save the figure.
    save_path = f'roc_curves_epoch_{epoch}.png'  
    plt.savefig(save_path)  
    plt.close()  
    return save_path  


def save_nifti(data, filename):  
    """Save an array as NIfTI."""
    img = nib.Nifti1Image(data, affine=np.eye(4))  # identity affine
    nib.save(img, filename)  

def visualize_segmentation(image, target, pred, organ_names, epoch, save_dir='visualization'):  
    """  
    Render original image, target segmentation and prediction side by side,
    and save them as NIfTI.

    Args:
        image: input image, shape [C, D, H, W]
        target: target segmentation, shape [C, D, H, W]
        pred: predicted segmentation, shape [C, D, H, W]
        organ_names: organ names
        epoch: current epoch
        save_dir: output directory
    """  
    os.makedirs(save_dir, exist_ok=True)  
    
    # Pick a slice according to the tensor rank.
    if image.dim() == 5:  # 5D tensor
        image = image[0, :, :, :, :]  
        target = target[0, :, :, :, :]  
        pred = pred[0, :, :, :, :]  

    image = image[0, :, :, :]  # first channel
    # Save the original image.
    image_filename = os.path.join(save_dir, f'original_image_epoch_{epoch}.nii.gz')  
    save_nifti(image.cpu().numpy(), image_filename)  
    
    # Save target and prediction as NIfTI.
    for i, organ_name in enumerate(organ_names):  
        target_filename = os.path.join(save_dir, f'target_{organ_name}_epoch_{epoch}.nii.gz')  
        pred_filename = os.path.join(save_dir, f'pred_{organ_name}_epoch_{epoch}.nii.gz')  
        
        save_nifti(target[i].cpu().numpy(), target_filename)  
        save_nifti((pred[i] > 0.5).float().cpu().numpy(), pred_filename)  

    print(f"Saved original image, target, and prediction NIfTI files for epoch {epoch} in '{save_dir}'.")  

    return save_dir


def plot_confusion_matrix(cm, classes, title):  
    """Plot a confusion matrix."""
    fig, ax = plt.subplots()  
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  
    ax.figure.colorbar(im, ax=ax)  
    ax.set(xticks=np.arange(cm.shape[1]),  
           yticks=np.arange(cm.shape[0]),  
           xticklabels=classes, yticklabels=classes,  
           title=title,  
           ylabel='True label',  
           xlabel='Predicted label')  
    
    # Annotate each cell with its count.
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
    """Organ-level metrics."""
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
    Build the test log message.

    Args:
    epoch : int
        current epoch
    disease_metrics : dict
        per-disease metrics
    organ_metrics : dict
        per-organ metrics
    disease_names : list
        disease names
    organ_names : list
        organ names
    best_thresholds : array
        per-disease best threshold

    Returns:
    str : the formatted log message
    """  
    import numpy as np  
    
    # Disease-level averages.
    avg_metrics = {  
        'precision': np.mean([m['precision'] for m in metrics_per_disease]),  
        'recall': np.mean([m['recall'] for m in metrics_per_disease]),  
        'f1': np.mean([m['f1'] for m in metrics_per_disease]),  
        'accuracy': np.mean([m['accuracy'] for m in metrics_per_disease]),  
        'auroc': np.mean([m['auroc'] for m in metrics_per_disease])  
    }  
    
    # Organ-level averages.
    avg_organ_metrics = {  
        'precision': np.mean([m['precision'] for m in organ_metrics.values()]),  
        'recall': np.mean([m['recall'] for m in organ_metrics.values()]),  
        'f1': np.mean([m['f1'] for m in organ_metrics.values()]),  
        'accuracy': np.mean([m['accuracy'] for m in organ_metrics.values()]),  
        'auroc': np.mean([m['auroc'] for m in organ_metrics.values()])  
    }  
    
    # Assemble the message.
    log_info = f"\nTest Epoch {epoch} Summary:\n"  
    log_info += "=" * 50 + "\n"  
    
    # Overall disease metrics.
    log_info += "\nOverall Disease Metrics:\n"  
    log_info += "-" * 30 + "\n"  
    log_info += f"Average Precision: {avg_metrics['precision']:.4f}\n"  
    log_info += f"Average Recall: {avg_metrics['recall']:.4f}\n"  
    log_info += f"Average F1-Score: {avg_metrics['f1']:.4f}\n"  
    log_info += f"Average Accuracy: {avg_metrics['accuracy']:.4f}\n"  
    log_info += f"Average AUROC: {avg_metrics['auroc']:.4f}\n"  
    
    # Overall organ metrics.
    log_info += "\nOverall Organ Metrics:\n"  
    log_info += "-" * 30 + "\n"  
    log_info += f"Average Precision: {avg_organ_metrics['precision']:.4f}\n"  
    log_info += f"Average Recall: {avg_organ_metrics['recall']:.4f}\n"  
    log_info += f"Average F1-Score: {avg_organ_metrics['f1']:.4f}\n"  
    log_info += f"Average Accuracy: {avg_organ_metrics['accuracy']:.4f}\n"  
    log_info += f"Average AUROC: {avg_organ_metrics['auroc']:.4f}\n"  
    
    # Per-disease detail.
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
    
    # Per-organ detail.
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
    Plot and save a confusion matrix.
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
        
        # Data kept aside for heatmap rendering.
        heatmap_data = []

        vis_samples = []
    


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

    # Post-processed predictions, one list per disease.
    processed_predictions = [[] for _ in range(16)]
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

    # Organ-to-disease mapping.
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
            abnormal_pred = abnormal_preds[-1]  # output of the finest scale
            
            disease_frequencies = [  
                0.1072, 0.0705, 0.2476, 0.1420, 0.2534, 0.1939, 0.2558, 0.4548,  
                0.3666, 0.2672, 0.1185, 0.0744, 0.1034, 0.1755, 0.0999, 0.0788  
            ]  

            seg_loss = segmentation_criterion(seg_pred, seg_targets)  
            disease_losses, abnormal_pred_avg = Abnormal_loss(seg_pred, abnormal_pred, abnormal_targets, disease_frequencies)  
            
            # Collect post-processed predictions and ground truth.
            abnormal_targets_np = abnormal_targets.cpu().numpy()  
            abnormal_pred_avg_np = abnormal_pred_avg.cpu().numpy() 
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

            # NSD.
            nsd_per_sample = calculate_nsd(seg_pred, seg_targets)  # [B, C]
            nsd_scores_organ.append(nsd_per_sample.cpu().numpy())
            # print(f'dice_score:{dice_score_organ}')
            # print(f'dice_score:{dice_score}') 
            # dice_scores.append(dice_score)  

            # Stash what the heatmaps will need.
            if save_heatmap and iter in selected_indices:
                for batch_idx in range(images.size(0)):
                    heatmap_data.append({
                        'predictions': [  
                            abnormal_preds[0][batch_idx:batch_idx+1].cpu(),  # low resolution
                            abnormal_preds[1][batch_idx:batch_idx+1].cpu()   # high resolution
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

    # Stack the per-sample Dice coefficients.
    dice_scores_organ = np.array(dice_scores_organ)  

    # Save as .npy.
    np.save('dice_scores.npy', dice_scores_organ) 

    # Directory for the confusion matrices.
    confusion_matrix_dir = os.path.join(config.work_dir, "test_results","test_confusion_matrices", f"epoch_{epoch}")  
    os.makedirs(confusion_matrix_dir, exist_ok=True) 
    # Directory for the ROC curves.
    roc_curve_dir = os.path.join(config.work_dir, "test_results", "roc_curves", f"epoch_{epoch}")  
    os.makedirs(roc_curve_dir, exist_ok=True) 

    # Metric accumulators.
    num_diseases = len(disease_names)  
    tp_sum = np.zeros(num_diseases, dtype=int)  
    tn_sum = np.zeros(num_diseases, dtype=int)  
    fp_sum = np.zeros(num_diseases, dtype=int)  
    fn_sum = np.zeros(num_diseases, dtype=int)  

    precision_per_disease = np.zeros(num_diseases)  
    recall_per_disease = np.zeros(num_diseases)  
    accuracy_per_disease = np.zeros(num_diseases)  
    f1_per_disease = np.zeros(num_diseases)  
    auroc_per_disease = np.zeros(num_diseases)

    # Per-disease y_true / y_pred, kept for the ROC curves.
    y_true_list = []  
    y_pred_list = [] 

    # Compute every metric.
    for disease_idx in range(num_diseases):  
        y_true = np.array(all_targets[disease_idx])  
        y_pred_proba = np.array(processed_predictions[disease_idx])  
        y_pred = (y_pred_proba > best_thresholds[disease_idx]).astype(int)  

        # Save y_true and y_pred_proba, one file per disease.
        y_true_path = os.path.join(roc_curve_dir, f'y_true_{disease_names[disease_idx]}.npy')  
        y_pred_proba_path = os.path.join(roc_curve_dir, f'y_pred_proba_{disease_names[disease_idx]}.npy')  
        
        np.save(y_true_path, y_true)  
        np.save(y_pred_proba_path, y_pred_proba) 

        # Collect y_true and y_pred.
        y_true_list.append(y_true)  
        y_pred_list.append(y_pred)

        # Confusion matrix.
        cm = confusion_matrix(y_true, y_pred)  
        save_path = plot_confusion_matrix(y_true, y_pred, disease_names[disease_idx], confusion_matrix_dir)  

        # Extract TP, TN, FP, FN.
        tp = cm[1, 1]  # True Positive  
        tn = cm[0, 0]  # True Negative  
        fp = cm[0, 1]  # False Positive  
        fn = cm[1, 0]  # False Negative  

        # Accumulate TP, TN, FP, FN.
        tp_sum[disease_idx] = tp  
        tn_sum[disease_idx] = tn  
        fp_sum[disease_idx] = fp  
        fn_sum[disease_idx] = fn 

        # AUROC.
        try:  
            auroc = roc_auc_score(y_true, y_pred_proba)  
        except ValueError:  
            auroc = 0.0  
        auroc_per_disease[disease_idx] = auroc

        # Remaining per-disease metrics.
        precision_per_disease[disease_idx] = tp / (tp + fp + 1e-8)  
        recall_per_disease[disease_idx] = tp / (tp + fn + 1e-8)  
        accuracy_per_disease[disease_idx] = (tp + tn) / (tp + tn + fp + fn + 1e-8)  
        f1_per_disease[disease_idx] = 2 * precision_per_disease[disease_idx] * recall_per_disease[disease_idx] / (precision_per_disease[disease_idx] + recall_per_disease[disease_idx] + 1e-8)

        # ROC curve for this disease.
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)  
        roc_auc = auc(fpr, tpr)  

        plt.figure()  
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))  
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # chance line
        plt.xlim([0.0, 1.0])  
        plt.ylim([0.0, 1.05])  
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate')  
        plt.title(f'ROC Curve for {disease_names[disease_idx]}')  
        plt.legend(loc='lower right')  
        
        # Save the ROC curve.
        roc_curve_path = os.path.join(roc_curve_dir, f'roc_curve_disease_{disease_idx}.png')  
        plt.savefig(roc_curve_path)  
        plt.close()  

    # Organ metric accumulators.
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

    # Per-organ metrics.
    for organ_idx, (organ, disease_indices) in enumerate(organ_disease_mapping.items()):  
        # Gather predictions and labels for every disease of this organ.
        organ_predictions = []  
        organ_targets = []  
        organ_pred_proba = []
        # TP, TN, FP, FN for this organ.
        organ_tp = organ_tn = organ_fp = organ_fn = 0   

        # Accumulate predictions, labels and the confusion counts.
        for disease_idx in disease_indices:  
            organ_targets.extend(y_true_list[disease_idx])  
            organ_predictions.extend(y_pred_list[disease_idx])          # thresholded
            organ_pred_proba.extend(processed_predictions[disease_idx])  # raw probabilities
            
            
            organ_tp += tp_sum[disease_idx]  
            organ_tn += tn_sum[disease_idx]  
            organ_fp += fp_sum[disease_idx]  
            organ_fn += fn_sum[disease_idx]  

        # Accumulate the organ totals.
        organ_tp_sum[organ_idx] = organ_tp  
        organ_tn_sum[organ_idx] = organ_tn  
        organ_fp_sum[organ_idx] = organ_fp  
        organ_fn_sum[organ_idx] = organ_fn 


        # Organ AUROC.
        organ_auroc_value = roc_auc_score(organ_targets, organ_pred_proba) if organ_targets else 0.0  
        organ_auroc[organ_idx] = organ_auroc_value  

        # Organ-level metrics.
        total = organ_tp + organ_fp + 1e-8  
        organ_precision[organ_idx] = organ_tp / total  
        organ_recall[organ_idx] = organ_tp / (organ_tp + organ_fn + 1e-8)  
        organ_accuracy[organ_idx] = (organ_tp + organ_tn) / (organ_tp + organ_tn + organ_fp + organ_fn + 1e-8)  
        organ_f1[organ_idx] = 2 * organ_precision[organ_idx] * organ_recall[organ_idx] / (organ_precision[organ_idx] + organ_recall[organ_idx] + 1e-8)  

        # Plot and save the confusion matrix.
        save_path, cm = plot_confusion_matrix(organ_targets, organ_predictions, organ, confusion_matrix_dir)  

        # ROC curve for this organ.
        fpr, tpr, _ = roc_curve(organ_targets, organ_pred_proba)  
        organ_roc_auc = auc(fpr, tpr)  

        plt.figure()  
        plt.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = {:.2f})'.format(organ_roc_auc))  
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # chance line
        plt.xlim([0.0, 1.0])  
        plt.ylim([0.0, 1.05])  
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate')  
        plt.title(f'ROC Curve for {organ}')  
        plt.legend(loc='lower right')  
        
        # Save the ROC curve.
        organ_roc_curve_path = os.path.join(roc_curve_dir, f'roc_curve_organ_{organ}.png')  
        plt.savefig(organ_roc_curve_path)  
        plt.close()  

    # Averages.
    avg_precision = np.mean(precision_per_disease)  
    avg_recall = np.mean(recall_per_disease)  
    avg_accuracy = np.mean(accuracy_per_disease)  
    avg_f1 = np.mean(f1_per_disease)  
    avg_abnormal_loss = np.mean([np.mean(losses) if losses else 0.0 for losses in abnormal_loss_list])  
    avg_auroc = np.mean(auroc_per_disease)  

    avg_dice = np.mean(dice_scores) if dice_scores else 0.0  
    avg_loss = np.mean(loss_list) if loss_list else 0.0  
    avg_seg_loss = np.mean(seg_loss_list) if seg_loss_list else 0.0   
 

    # Log the averages.
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
        # Append to the log.
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

    # Per-organ confusion matrices.
    # Post-processing.
    dice_scores_organ = np.concatenate(dice_scores_organ, axis=0)  # [N, C]
    nsd_scores_organ = np.concatenate(nsd_scores_organ, axis=0)    # [N, C]
    
    avg_dice_per_organ = dice_scores_organ.mean(axis=0)
    avg_nsd_per_organ = nsd_scores_organ.mean(axis=0)
    # Append the report to the log.
    log_info += "\nOrgan-wise Segmentation Metrics:\n"
    for idx, name in enumerate(organ_names):
        log_info += (f"{name}: Dice={avg_dice_per_organ[idx]:.4f} ± {dice_scores_organ[:, idx].std():.4f}, "
                     f"NSD={avg_nsd_per_organ[idx]:.4f} ± {nsd_scores_organ[:, idx].std():.4f}\n")
        
    log_info += "\n\nOrgan-level Confusion Matrices:"  
    for organ_idx, (organ, disease_indices) in enumerate(organ_disease_mapping.items()):  
        # Append to the log.
        log_info += f"\n{organ}:\n"  
        log_info += f"AUROC: {organ_auroc[organ_idx]:.4f}, "  
        log_info += f"Accuracy: {organ_accuracy[organ_idx]:.4f}, "  
        log_info += f"Precision: {organ_precision[organ_idx]:.4f}, "  
        log_info += f"Recall: {organ_recall[organ_idx]:.4f}, "  
        log_info += f"F1: {organ_f1[organ_idx]:.4f}\n"  

        # Rebuild the confusion matrix from the accumulated counts.
        tn = organ_tn_sum[organ_idx]  
        fp = organ_fp_sum[organ_idx]  
        fn = organ_fn_sum[organ_idx]  
        tp = organ_tp_sum[organ_idx]  

        log_info += f"Confusion Matrix:\n"  
        log_info += f"TN: {tn}, FP: {fp}\n"  
        log_info += f"FN: {fn}, TP: {tp}\n"

    # Overall metrics.
    total_tn = sum(organ_tn_sum)  
    total_fp = sum(organ_fp_sum)  
    total_fn = sum(organ_fn_sum)  
    total_tp = sum(organ_tp_sum)  
    total_cm = np.array([[total_tn, total_fp], [total_fn, total_tp]], dtype=int)  

    # Save the overall confusion matrix.
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

    # Overall metrics.
    total_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8)  
    total_precision = total_tp / (total_tp + total_fp + 1e-8)  
    total_recall = total_tp / (total_tp + total_fn + 1e-8)  
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-8)   

    log_info += f"\nOverall Metrics from Confusion Matrix:\n"  
    log_info += f"Accuracy: {total_accuracy:.4f}\n"  
    log_info += f"Precision: {total_precision:.4f}\n"  
    log_info += f"Recall: {total_recall:.4f}\n"  
    log_info += f"F1 Score: {total_f1:.4f}\n" 

    # Render the heatmaps now that the thresholds are known.
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

            # Segmentation visualisation.
            depth = data['images'].shape[1]
            middle_slice = depth // 2
        
        # Segmentation visualisation.
        for vis_sample in vis_samples:
            vis_path = visualize_segmentation(  
                vis_sample['image'],      # [C, D, H, W]  
                vis_sample['target'],     # [C, D, H, W]  
                vis_sample['prediction'], # [C, D, H, W]  
                organ_names,  
                epoch,  
                save_dir=os.path.join(save_dir, f'segmentation_visualizations/sample_{vis_sample["sample_name"]}')
            )  
            logger.info(f"Successfully saved segmentation visualization for sample {vis_sample['sample_name']} at epoch {epoch}")  

            # vis_path = visualize_segmentation(  
            #     data['images'],      # [C, D, H, W]  
            #     seg_targets,     # [C, D, H, W]  
            #     data['segmentation_preds'], # [C, D, H, W]  
            #     middle_slice,  
            #     organ_names,  
            #     epoch,  
            # )  
            # logger.info(f"Successfully saved segmentation visualization for sample {data['sample_name']} at epoch {epoch}")  

    print(log_info)  
    logger.info(log_info)  

    return avg_loss