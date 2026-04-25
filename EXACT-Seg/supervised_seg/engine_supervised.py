import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import wandb
import time
import torch.nn.functional as F
import swanlab
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
# Utilities for confusion matrix computation and visualization.
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
from scipy.ndimage import binary_erosion, distance_transform_edt

import matplotlib.patches as mpatches 

def calculate_nsd(pred, target, threshold=0.5, tau=1.0):
    """
    Compute Normalized Surface Dice (NSD).
    :param pred: Predicted segmentation mask, shape [B, C, D, H, W].
    :param target: Ground-truth segmentation mask, shape [B, C, D, H, W].
    :param threshold: Binarization threshold.
    :param tau: Surface distance tolerance in voxels.
    :return: NSD scores, shape [B, C].
    """
    pred = (pred > threshold).float()
    target = target.float()
    batch_size, num_organs = pred.shape[0], pred.shape[1]
    nsd_per_organ = torch.zeros((batch_size, num_organs), device=pred.device)
    
    for b in range(batch_size):
        for organ_idx in range(num_organs):
            pred_organ = pred[b, organ_idx].cpu().numpy()
            target_organ = target[b, organ_idx].cpu().numpy()
            
            # Extract surface voxels from a binary mask.
            def get_surface(mask):
                if not np.any(mask):
                    return np.zeros_like(mask, dtype=bool)
                eroded = binary_erosion(mask, structure=np.ones((3,3,3)))
                return mask & ~eroded
            
            surface_pred = get_surface(pred_organ > 0.5)
            surface_target = get_surface(target_organ > 0.5)
            
            surface_pred_coords = np.argwhere(surface_pred)
            surface_target_coords = np.argwhere(surface_target)
            
            # Handle edge cases for empty predictions/targets.
            if len(surface_pred_coords) + len(surface_target_coords) == 0:
                nsd = 1.0
            elif len(surface_pred_coords) == 0 or len(surface_target_coords) == 0:
                nsd = 0.0
            else:
                # Compute distance maps to the nearest opposite surface.
                dist_map_target = distance_transform_edt(~surface_target)
                dist_map_pred = distance_transform_edt(~surface_pred)
                
                # Count points whose distance is within tau.
                pred_dists = dist_map_target[tuple(surface_pred_coords.T)]
                target_dists = dist_map_pred[tuple(surface_target_coords.T)]
                tp_a = np.sum(pred_dists <= tau)
                tp_b = np.sum(target_dists <= tau)
                
                nsd = (tp_a + tp_b) / (len(surface_pred_coords) + len(surface_target_coords))
                
            nsd_per_organ[b, organ_idx] = nsd
            
    return nsd_per_organ

# Bidirectional weights.
import torch
import torch.nn.functional as F


# Model outputs are already activated; do not activate again.
def weighted_binary_cross_entropy(pred, target, weight_pos, weight_neg):
    """
    Compute weighted binary cross-entropy loss.
    pred: Predicted values, shape (B,).
    target: Ground-truth labels, shape (B,).
    weight_pos: Positive sample weight.
    weight_neg: Negative sample weight.
    """
    # Standard weighted BCE terms.
    loss_pos = weight_pos * target * torch.log(pred + 1e-6)
    loss_neg = weight_neg * (1 - target) * torch.log(1 - pred + 1e-6)
    loss = -(loss_pos + loss_neg)
    return loss

def calculate_dice(pred, target, threshold=0.5):
    # Compute per-organ Dice and average over the batch dimension.
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2 * intersection + 1e-7) / (union + 1e-7)  # Avoid division by zero.
    # Return mean Dice score for each organ.
    return dice.mean(dim=0)

def train_one_epoch(train_loader, model, segmentation_criterion, optimizer, scheduler, epoch, step, logger, config, writer, device):  
    model.train()  
    loss_list = []  


    
    logger.info(f"Current epoch {epoch}:")  

    train_loader = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=True)  
    

    for iter, data in enumerate(train_loader):  
        # if iter >= 1500:  
        #     break  
        optimizer.zero_grad()  

        images, seg_targets,  _ = data  
        images = images.to(device)  
        seg_targets = seg_targets.to(device)  

        # print("images.shape:",images.shape,"seg_targets.shape:",seg_targets.shape)
        seg_pred= model(images)  # abnormal_preds is a list: [scale1_pred, scale2_pred]
        assert seg_pred.shape == seg_targets.shape, f"seg_pred shape {seg_pred.shape} and seg_targets shape {seg_targets.shape} do not match!"
        # print("seg_pred.shape:",seg_pred.shape,"seg_targets.shape:",seg_targets.shape)
        seg_loss = segmentation_criterion(seg_pred, seg_targets)  

        # Total loss for this iteration.

        total_loss = seg_loss

        total_loss.backward()  
        optimizer.step()  

        loss_list.append(total_loss.item())  

        train_loader.set_postfix({  
            'Loss': f'{total_loss.item():.4f}',  
        })  

    # Compute epoch-level average metrics.
    avg_loss = np.mean(loss_list)  

    # Prepare wandb logging payload.
    wandb_metrics = {  
        "train/total_loss": avg_loss,  
    }  

    # Log all training metrics.
    wandb.log(wandb_metrics, step=epoch)  
    swanlab.log({"train/total_loss": avg_loss}, step=epoch)
    if scheduler is not None:  
        scheduler.step()  

    step += len(train_loader)  
    return step


def upsample_3d(tensor, target_size):  
    """  
    3D upsampling helper that resizes an input tensor to a target size.
    
    Args:  
        tensor (torch.Tensor): Input tensor, shape [B, C, D, H, W].
        target_size (tuple): Target spatial size (D, H, W).
    
    Returns:  
        torch.Tensor: Upsampled tensor.
    """  
    # Use trilinear interpolation for volumetric upsampling.
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
    # Unpack multi-scale predictions.
    low_res_preds, high_res_preds = predictions  
    
    # Get target size from high-resolution prediction.
    _, _, target_depth, target_height, target_width = high_res_preds.shape  
    
    # Upsample low-resolution prediction to high-resolution size.
    low_res_upsampled = upsample_3d(low_res_preds, (target_depth, target_height, target_width))  

    # Use both high-res and upsampled low-res predictions.
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
    
    # 3D data processing: extract segmentation prediction.
    seg_pred = segmentation_preds[0].cpu().numpy()  # [7, D, H, W]  
    
    # Build 3D binary segmentation mask.
    seg_mask = (seg_pred > seg_threshold).astype(np.float32)  
    target = targets[0].cpu().numpy()  # [18]  
    original_image = images[0, 0].cpu().numpy()  # [D, H, W]  
    
    # Save original image.
    affine = np.eye(4)  
    original_nifti = nib.Nifti1Image(original_image, affine)  
    original_nifti.header['descrip'] = f'Original 3D Image, Epoch: {epoch}'  
    original_save_path = sample_dir / f"original_image.nii.gz"  
    nib.save(original_nifti, original_save_path)  
    
    # Disease class names.
    disease_names = [
        "Medical material", "Arterial wall calcification",
        "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification",  
        "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule",  
        "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",  
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",  
        "Bronchiectasis", "Interlobular septal thickening"  
    ]  
    
    # -  
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
    
    #   
    info_file = sample_dir / "prediction_info.txt"  
    with open(info_file, "w") as f:  
        f.write(f"Epoch: {epoch}\n")  
        f.write(f"Sample: {sample_idx}\n")  
        f.write(f"Abnormal Detection Parameters: top-{topk}\n")  
        f.write("Disease Predictions:\n")  
        
        #   
        prediction_results = {  
            "High-Res": high_res_pred,  
            "Low-Res": low_res_pred  
        }  
        
        #   
        for res_name, pred in prediction_results.items():  
            f.write(f"\n{res_name} Predictions:\n")  
            
            #   
            for disease_idx, disease_name in enumerate(disease_names):  
                # 3D  
                disease_pred = pred[disease_idx]  # 3D [D, H, W]  
                disease_label = int(target[disease_idx])  #   
                
                #   
                related_organs = disease_organ_mapping[disease_name]  
                combined_mask = np.zeros_like(seg_mask[0])  
                for organ_name in related_organs:  
                    organ_idx = organ_names.index(organ_name)  
                    combined_mask = np.maximum(combined_mask, seg_mask[organ_idx])  
                
                # combined_pred（）  
                combined_pred = disease_pred * combined_mask  
                
                # k  
                topk_mean = np.mean(np.sort(combined_pred.flatten())[-topk:])  
                
                #   
                current_threshold = abnormal_threshold[disease_idx] if abnormal_threshold is not None else 0.5  
                pred_abnormal = int(topk_mean > current_threshold)  
                
                #   
                f.write(f"\n  {disease_name}:\n")  
                f.write(f"    Ground Truth: {disease_label}\n")  
                f.write(f"    Prediction: {pred_abnormal}\n")  
                f.write(f"    Top-{topk} Mean: {topk_mean:.4f}\n")  
                f.write(f"    Threshold: {current_threshold:.4f}\n")  
                f.write(f"    Related Organs: {', '.join(related_organs)}\n")  
                
                #   
                result_str = f"GT{disease_label}_PD{pred_abnormal}"  
                
                # 3DNIfTI   
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

    

 

def valid_one_epoch(valid_loader, model, segmentation_criterion, epoch, logger, config, writer, device, save_heatmap=True):
    model.eval()
    loss_list = []
    seg_loss_list = []
    #  abnormal_loss_list  18 
    abnormal_loss_list = [[] for _ in range(18)]
    sample_idx = 0
    np.random.seed(42)
    
    dataset_size = len(valid_loader.dataset)
    selected_indices = set(np.random.choice(dataset_size, min(40, dataset_size), replace=False))
    current_dir = config.work_dir

    valid_loader = tqdm(valid_loader, desc=f"Epoch {epoch} Validation", leave=True, dynamic_ncols=True)

    dice_scores = []
    dice_scores_organ =[]

    with torch.no_grad():
        for iter, data in enumerate(valid_loader):
            # if iter >= 100:  
            #     break  
            images, seg_targets,  sample_names = data
            images, seg_targets = images.to(device), seg_targets.to(device)

            seg_pred = model(images)
            
            seg_loss = segmentation_criterion(seg_pred, seg_targets)

            # 

            total_loss =  seg_loss 

            loss_list.append(total_loss.item())
            seg_loss_list.append(seg_loss.item())

            # Dice，7
            dice_score_organ = calculate_dice(seg_pred, seg_targets)
            dice_score = dice_score_organ.mean().item()  

            dice_scores.append(dice_score)


            # ，


    # DiceNumPy  
    dice_scores_organ = np.array(dice_scores_organ)  

    # .npy  
    np.save('dice_scores.npy', dice_scores_organ) 

    #   
    logger.info("Finding optimal thresholds for each disease using Youden's index...")  
    # 18
   

    loss_list = np.array(loss_list)
    avg_loss= np.mean(loss_list)
    avg_dice=np.mean(dice_scores)
    # 
    confusion_matrices = {}
    log_info = f"Validation Epoch {epoch} Summary:\n"
    log_info += "Overall Metrics:\n"
    log_info += f"Total Loss: {avg_loss:.4f}, "

    logger.info(log_info)  

    # wandb  
    wandb_log_dict = {  
        "Validation Total Loss": avg_loss,  
        "Dice Mean": avg_dice,
    }  
    swanlab.log({"val/total_loss": avg_loss,
                "Dice Mean":avg_dice}, step=epoch)
    # wandb  
    wandb.log(wandb_log_dict, step=epoch)  

    return 1-avg_dice


def calculate_dice_per_organ(pred, target, threshold=0.5):  
    # pred shape: [batch_size, num_classes, D, H, W]  
    pred = (pred > threshold).float()  
    # dice，batch  
    dice_scores = []  
    batch_size = pred.size(0)  
    num_classes = pred.size(1)  
    
    for i in range(num_classes):  
        #   
        pred_organ = pred[:, i:i+1, ...]  # [batch_size, 1, D, H, W]  
        target_organ = target[:, i:i+1, ...]  # [batch_size, 1, D, H, W]  
        
        intersection = (pred_organ * target_organ).sum(dim=(2, 3, 4))  # [batch_size, 1]  
        union = pred_organ.sum(dim=(2, 3, 4)) + target_organ.sum(dim=(2, 3, 4))  # [batch_size, 1]  
        dice = (2 * intersection + 1e-7) / (union + 1e-7)  # [batch_size, 1]  
        
        # batchdice  
        dice_scores.append(dice.mean().item())  
    
    return dice_scores  # num_classes，dice  

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
    
    #   
    save_path = f'roc_curves_epoch_{epoch}.png'  
    plt.savefig(save_path)  
    plt.close()  
    return save_path  


def save_nifti(data, filename):  
    """ NIfTI """  
    img = nib.Nifti1Image(data, affine=np.eye(4))  #   
    nib.save(img, filename)  

def visualize_segmentation(image, target, pred, organ_names, epoch, save_dir='visualization'):  
    """  
    ，， NIfTI   
    
    Args:  
        image: [C, D, H, W]  
        target: [C, D, H, W]  
        pred: [C, D, H, W]  
        organ_names:   
        epoch:   
        save_dir:   
    """  
    os.makedirs(save_dir, exist_ok=True)  
    
    #   
    if image.dim() == 5:  #  5D   
        image = image[0, :, :, :, :]  
        target = target[0, :, :, :, :]  
        pred = pred[0, :, :, :, :]  

    image = image[0, :, :, :]  # 
    #   
    image_filename = os.path.join(save_dir, f'original_image_epoch_{epoch}.nii.gz')  
    save_nifti(image.cpu().numpy(), image_filename)  
    
    #  NIfTI   
    for i, organ_name in enumerate(organ_names):  
        target_filename = os.path.join(save_dir, f'target_{organ_name}_epoch_{epoch}.nii.gz')  
        pred_filename = os.path.join(save_dir, f'pred_{organ_name}_epoch_{epoch}.nii.gz')  
        
        save_nifti(target[i].cpu().numpy(), target_filename)  
        save_nifti((pred[i] > 0.5).float().cpu().numpy(), pred_filename)  

    print(f"Saved original image, target, and prediction NIfTI files for epoch {epoch} in '{save_dir}'.")  

    return save_dir


def plot_confusion_matrix(cm, classes, title):  
    """"""  
    fig, ax = plt.subplots()  
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  
    ax.figure.colorbar(im, ax=ax)  
    ax.set(xticks=np.arange(cm.shape[1]),  
           yticks=np.arange(cm.shape[0]),  
           xticklabels=classes, yticklabels=classes,  
           title=title,  
           ylabel='True label',  
           xlabel='Predicted label')  
    
    #   
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
    """"""  
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
      
    
    :  
    epoch : int  
          
    metrics_per_disease : list of dict  
          
    organ_metrics : dict  
          
    disease_names : list  
          
    organ_names : list  
          
    best_thresholds : numpy array  
          
    
    :  
    str :   
    """  
    import numpy as np  
    
    #   
    avg_metrics = {  
        'precision': np.mean([m['precision'] for m in metrics_per_disease]),  
        'recall': np.mean([m['recall'] for m in metrics_per_disease]),  
        'f1': np.mean([m['f1'] for m in metrics_per_disease]),  
        'accuracy': np.mean([m['accuracy'] for m in metrics_per_disease]),  
        'auroc': np.mean([m['auroc'] for m in metrics_per_disease])  
    }  
    
    #   
    avg_organ_metrics = {  
        'precision': np.mean([m['precision'] for m in organ_metrics.values()]),  
        'recall': np.mean([m['recall'] for m in organ_metrics.values()]),  
        'f1': np.mean([m['f1'] for m in organ_metrics.values()]),  
        'accuracy': np.mean([m['accuracy'] for m in organ_metrics.values()]),  
        'auroc': np.mean([m['auroc'] for m in organ_metrics.values()])  
    }  
    
    #   
    log_info = f"\nTest Epoch {epoch} Summary:\n"  
    log_info += "=" * 50 + "\n"  
    
    #   
    log_info += "\nOverall Disease Metrics:\n"  
    log_info += "-" * 30 + "\n"  
    log_info += f"Average Precision: {avg_metrics['precision']:.4f}\n"  
    log_info += f"Average Recall: {avg_metrics['recall']:.4f}\n"  
    log_info += f"Average F1-Score: {avg_metrics['f1']:.4f}\n"  
    log_info += f"Average Accuracy: {avg_metrics['accuracy']:.4f}\n"  
    log_info += f"Average AUROC: {avg_metrics['auroc']:.4f}\n"  
    
    #   
    log_info += "\nOverall Organ Metrics:\n"  
    log_info += "-" * 30 + "\n"  
    log_info += f"Average Precision: {avg_organ_metrics['precision']:.4f}\n"  
    log_info += f"Average Recall: {avg_organ_metrics['recall']:.4f}\n"  
    log_info += f"Average F1-Score: {avg_organ_metrics['f1']:.4f}\n"  
    log_info += f"Average Accuracy: {avg_organ_metrics['accuracy']:.4f}\n"  
    log_info += f"Average AUROC: {avg_organ_metrics['auroc']:.4f}\n"  
    
    #   
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
    
    #   
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


def test_one_epoch(test_loader, model, segmentation_criterion,  epoch, logger,
                    config, writer, device,  save_heatmap=True):  
    model.eval()  
    loss_list = []  
    seg_loss_list = []  

    sample_idx = 0  
    np.random.seed(42)  
    
    dataset_size = len(test_loader.dataset)  
    selected_indices = set(np.random.choice(dataset_size, min(40, dataset_size), replace=False))  
    current_dir = config.work_dir  

    if save_heatmap:
        heatmap_dir = os.path.join(config.work_dir, "test_results", "heatmap")
        os.makedirs(heatmap_dir, exist_ok=True)
    

    logger.info(f"Test epoch {epoch}: ")  

    test_loader = tqdm(test_loader, desc=f"Epoch {epoch} Test", leave=True, dynamic_ncols=True)  

    dice_scores = []  
    dice_scores_organ = []
    nsd_scores_organ = []


    #   
    processed_predictions = [[] for _ in range(16)]  #   
    all_targets = [[] for _ in range(16)]  


    seg_save_dir = os.path.join(config.work_dir, "test_results", "segmentations")
    os.makedirs(seg_save_dir, exist_ok=True)
    with torch.no_grad():  
        for iter, data in enumerate(test_loader):  
            # if iter>100:
            #     break
            images, seg_targets, sample_names = data  
            images, seg_targets= images.to(device), seg_targets.to(device)

            seg_pred = model(images)  
            seg_loss = segmentation_criterion(seg_pred, seg_targets)  
            
            total_loss =  seg_loss 

            loss_list.append(total_loss.item())  
            seg_loss_list.append(seg_loss.item())  

            dice_score_organ = calculate_dice(seg_pred, seg_targets) 
            dice_score = dice_score_organ.mean().item()  
            dice_score_organ = dice_score_organ.cpu().numpy()

            dice_scores.append(dice_score)
            dice_scores_organ.append(dice_score_organ)

            # NSD
            nsd_per_sample = calculate_nsd(seg_pred, seg_targets)  # [B, C]
            nsd_scores_organ.append(nsd_per_sample.cpu().numpy())

            seg_pred_bin=(seg_pred > 0.5).float()
            batch_size = images.size(0)
            for b in range(batch_size):
                # ， batch 
                name = sample_names[b] if isinstance(sample_names, (list, tuple)) else sample_names
                name = name if isinstance(name, str) else str(name)
                base = os.path.basename(name)
                #  .nii.gz 
                if base.endswith(".nii.gz"):
                    base = base[:-7]
                else:
                    base = os.path.splitext(base)[0]
                
                
                
                vol = seg_pred_bin[b].squeeze().detach().cpu().numpy()  # [D,H,W]
                # if vol.shape!=(2,64,128,128):
                #     vol_filled=np.zeros((2,64,128,128),dtype=vol.dtype)
                #     vol_filled[:,:,24:88,22:102]=vol
                #     vol=vol_filled
                
                nifti_img = nib.Nifti1Image(vol, np.eye(4))
                nib.save(nifti_img, os.path.join(seg_save_dir, f"{base}.nii.gz"))

                if save_heatmap:
                    prob_vol = seg_pred[b].squeeze().detach().cpu().numpy()  #  [0,1]
                    nifti_prob = nib.Nifti1Image(prob_vol, np.eye(4))
                    nib.save(nifti_prob, os.path.join(heatmap_dir, f"{base}.nii.gz"))
    # DiceNumPy  
    dice_scores_organ = np.array(dice_scores_organ)  

    # .npy  
    np.save('dice_scores.npy', dice_scores_organ) 

    #   
    confusion_matrix_dir = os.path.join(config.work_dir, "test_results","test_confusion_matrices", f"epoch_{epoch}")  
    os.makedirs(confusion_matrix_dir, exist_ok=True) 
    #  ROC   
    roc_curve_dir = os.path.join(config.work_dir, "test_results", "roc_curves", f"epoch_{epoch}")  
    os.makedirs(roc_curve_dir, exist_ok=True) 

    #   
    

    avg_dice = np.mean(dice_scores) if dice_scores else 0.0  
    avg_loss = np.mean(loss_list) if loss_list else 0.0  
    avg_seg_loss = np.mean(seg_loss_list) if seg_loss_list else 0.0   
 

    #   
    confusion_matrices = {}  
    log_info = f"Test Epoch {epoch} Summary:\n"  
    log_info += "Overall Metrics:\n"  
    log_info += f"Total Loss: {avg_loss:.4f}, "  
    log_info += f"Seg Loss: {avg_seg_loss:.4f}, "  
    log_info += f"Seg Dice Score: {avg_dice:.4f}\n"  
    log_info += f"Average Disease Metrics: "  



    logger.info(log_info)  

    return avg_loss  # 