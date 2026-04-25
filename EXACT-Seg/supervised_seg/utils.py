import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
# import torchio as tio
import cv2
import nibabel as nib
from matplotlib.patches import Patch


from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric
import sys

def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


def save_imgs(img, msk, msk_pred, i, save_path, threshold=0.5, test_data_name=None):
    # 
    region_names = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart", 
                    "esophagus", "bone", "thyroid", "abdomen"]

    #  .png  .nii.gz 
    png_path = os.path.join(save_path, "png_files")
    nii_path = os.path.join(save_path, "nii_files")
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(nii_path, exist_ok=True)

    # 
    if img.dim() == 5:
        img = img.squeeze(0)
    if img.dim() == 4:
        img = img[0]
    depth_idx = img.shape[0] // 2
    img = img[depth_idx]
    img = img.detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img

    # 
    msk = msk.squeeze(0)
    msk_pred = msk_pred.squeeze(0)

    # 
    combined_msk = np.zeros((msk.shape[2], msk.shape[3], 3), dtype=np.float32)
    combined_msk_pred = np.zeros((msk.shape[2], msk.shape[3], 3), dtype=np.float32)
    colors = list(mcolors.TABLEAU_COLORS.values())[:msk.shape[0]]
    legend_patches = []  # 

    for organ_idx, region_name in enumerate(region_names):
        organ_msk = msk[organ_idx, depth_idx]
        organ_msk_pred = msk_pred[organ_idx, depth_idx]

        if isinstance(organ_msk, torch.Tensor):
            organ_msk = organ_msk.cpu().numpy()
        if isinstance(organ_msk_pred, torch.Tensor):
            organ_msk_pred = organ_msk_pred.cpu().numpy()

        organ_msk = np.where(organ_msk > 0.5, 1, 0)
        organ_msk_pred = np.where(organ_msk_pred > threshold, 1, 0)

        #  .png 
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(organ_msk, cmap='gray')
        plt.axis('off')
        plt.title(f"Organ {organ_idx} - {region_name} - Actual Mask")

        plt.subplot(1, 2, 2)
        plt.imshow(organ_msk_pred, cmap='gray')
        plt.axis('off')
        plt.title(f"Organ {organ_idx} - {region_name} - Predicted Mask")

        png_file_path = os.path.join(png_path, f"{i}_{region_name}_debug.png")
        plt.savefig(png_file_path)
        print(f"[DEBUG] Saved organ {region_name} debug image at: {png_file_path}")
        plt.close()

        # 
        color_hex = colors[organ_idx]  # 
        color = np.array(mcolors.to_rgb(color_hex))  #  RGB 
        combined_msk += np.stack([organ_msk] * 3, axis=-1) * color
        combined_msk_pred += np.stack([organ_msk_pred] * 3, axis=-1) * color
        legend_patches.append(Patch(facecolor=color_hex, label=region_name))

        #  .nii.gz 
        organ_pred_binary = np.where(msk_pred[organ_idx] > threshold, 1, 0).astype(np.uint8)
        nii_file_path = os.path.join(nii_path, f"{i}_{region_name}_pred_mask.nii.gz")
        nii_img = nib.Nifti1Image(organ_pred_binary, affine=np.eye(4))
        nib.save(nii_img, nii_file_path)
        print(f"[DEBUG] Saved {region_name} predicted mask as NIfTI file at: {nii_file_path}")

    # ，
    combined_msk = np.clip(combined_msk, 0, 1)
    combined_msk_pred = np.clip(combined_msk_pred, 0, 1)

    plt.figure(figsize=(10, 15))
    plt.subplot(3, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'Input Image - Slice {depth_idx}')

    plt.subplot(3, 1, 2)
    plt.imshow(combined_msk)
    plt.axis('off')
    plt.title(f'Actual Mask - All Organs Combined - Slice {depth_idx}')
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.subplot(3, 1, 3)
    plt.imshow(combined_msk_pred)
    plt.axis('off')
    plt.title(f'Predicted Mask - All Organs Combined - Slice {depth_idx}')
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1))

    combined_png_file_path = os.path.join(png_path, f"{i}_combined_organs.png")
    plt.savefig(combined_png_file_path, bbox_inches='tight')
    print(f"[DEBUG] Saved combined organs image at: {combined_png_file_path}")
    plt.close()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss
    

class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight
    
    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target[:].long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss
    

class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss


class OrganSegmentationLoss(nn.Module):
    def __init__(self, loss_type='dice', w_seg=1.0):
        """
        
        :param loss_type: 'binary' , 'dice'  Dice , 'soft'  Soft Dice 
        :param w_seg: 
        """
        super(OrganSegmentationLoss, self).__init__()
        self.loss_type = loss_type
        self.w_seg = w_seg

    def dice_coefficient_loss(self, y_true, y_pred, smooth=1e-6):
        intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
        union = torch.sum(y_true, dim=[1, 2, 3]) + torch.sum(y_pred, dim=[1, 2, 3])
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def soft_dice_coefficient_loss(self, y_true, y_pred, smooth=1e-6):
        intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
        sum_y_true = torch.sum(y_true, dim=[1, 2, 3])
        sum_y_pred = torch.sum(y_pred, dim=[1, 2, 3])
        soft_dice = (2 * intersection + smooth) / (sum_y_true + sum_y_pred + smooth)
        return 1 - soft_dice.mean()

    def forward(self, y_pred, y_true):
        segment_loss = 0
        
        # （）
        for i in range(y_true.shape[1]):  #  y_true  y_pred  (batch_size, 9, D, H, W)
            if self.loss_type == 'binary':
                organ_loss = F.binary_cross_entropy(y_pred[:, i], y_true[:, i])
            elif self.loss_type == 'dice':
                organ_loss = self.dice_coefficient_loss(y_true[:, i], y_pred[:, i])
            elif self.loss_type == 'soft':
                organ_loss = self.soft_dice_coefficient_loss(y_true[:, i], y_pred[:, i])
            else:
                raise ValueError("Unsupported loss_type. Choose either 'binary', 'dice', or 'soft'.")
                
            segment_loss += organ_loss  # 

        # 
        segment_loss /= y_true.shape[1]
        
        return self.w_seg * segment_loss  # 


# class AorticDissectionLoss(nn.Module):
#     def __init__(self, w_class=None, smooth=1e-6):
#         """
#         （ TL/FL  Dice + BCE）:
#          y_pred, y_true: (B, 2, D, H, W)  (B, 2, H, W, D)
#         : 0=TL, 1=FL
#         w_class: 2 [w_TL, w_FL]， 1.0
#         """
#         super(AorticDissectionLoss, self).__init__()
#         self.smooth = smooth
#         if w_class is None:
#             self.w_class = [1.0, 1.0]
#         else:
#             assert len(w_class) == 2, "w_class length must be 2 (TL, FL)"
#             self.w_class = w_class

#     def dice_loss(self, y_pred, y_true):
#         # y_pred,y_true: (B, D, H, W)  (B, H, W, D);  batch 
#         y_pred = y_pred.float()
#         y_true = y_true.float()
#         spatial_dims = list(range(1, y_pred.ndim))
#         intersection = torch.sum(y_pred * y_true, dim=spatial_dims)
#         union = torch.sum(y_pred, dim=spatial_dims) + torch.sum(y_true, dim=spatial_dims)
#         dice = (2. * intersection + self.smooth) / (union + self.smooth)
#         return 1.0 - dice.mean()

#     def forward(self, y_pred, y_true):
#         # y_pred: logits (B,2,...)  y_true: (B,2,...)
#         assert y_pred.shape[1] == 2 and y_true.shape[1] == 2, "2"
#         y_pred_prob = torch.sigmoid(y_pred)

#         total_loss = 0.0
#         for i in range(2):
#             prob_i = y_pred_prob[:, i]
#             true_i = y_true[:, i].float()
#             dice_l = self.dice_loss(prob_i, true_i)
#             bce_l = F.binary_cross_entropy_with_logits(y_pred[:, i], true_i)
#             term_loss = dice_l + bce_l
#             total_loss += self.w_class[i] * term_loss

#         # （）
#         total_loss = total_loss / (self.w_class[0] + self.w_class[1])
#         return total_loss



class AorticDissectionLoss(nn.Module):
    def __init__(self, loss_type='mix', w_aorta=0.0, w_class=None, smooth=1e-6):
        """
        :
         y_pred, y_true: (B, 2, D, H, W)  (B, 2, H, W, D)
        : 0=TL, 1=FL
        loss_type: 'dice'  'mix' (dice + BCE)
        w_class: 2 [w_TL, w_FL]
        w_aorta:  dice 
        """
        super(AorticDissectionLoss, self).__init__()
        self.loss_type = loss_type
        self.w_aorta = w_aorta
        self.smooth = smooth
        if w_class is None:
            self.w_class = [1.0, 1.0]
        else:
            assert len(w_class) == 2, "w_class length must be 2 (TL, FL)"
            self.w_class = w_class

    def dice_loss(self, y_pred, y_true):
        # y_pred,y_true: (B, D, H, W)  (B, H, W, D);  batch 
        y_pred = y_pred.float()
        y_true = y_true.float()
        spatial_dims = list(range(1, y_pred.ndim))
        intersection = torch.sum(y_pred * y_true, dim=spatial_dims)
        union = torch.sum(y_pred, dim=spatial_dims) + torch.sum(y_true, dim=spatial_dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, y_pred, y_true):
        # y_pred: logits (B,2,...)  y_true: (B,2,...)
        
        assert y_pred.shape[1] == 2 and y_true.shape[1] == 2, "2"
        y_pred_prob = torch.sigmoid(y_pred)

        total_loss = 0.0
        for i in range(2):
            prob_i = y_pred_prob[:, i]
            true_i = y_true[:, i]
            dice_l = self.dice_loss(prob_i, true_i)
            if self.loss_type == 'mix':
                bce_l = F.binary_cross_entropy_with_logits(y_pred[:, i], true_i.float())
                term_loss = dice_l + bce_l
            else:
                term_loss = dice_l
            total_loss += self.w_class[i] * term_loss

        total_loss /= 2.0

        if self.w_aorta > 0:
            aorta_true = torch.clamp(torch.sum(y_true, dim=1), 0, 1)
            aorta_pred = torch.clamp(torch.sum(y_pred_prob, dim=1), 0, 1)
            loc_loss = self.dice_loss(aorta_pred, aorta_true)
            total_loss += self.w_aorta * loc_loss

        return total_loss
class AbnormalSegmentationLoss(nn.Module):
    def __init__(self,
                 w_seg: float = 1.0,
                 tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7,
                 focal_alpha: float = 0.9,      # （）
                 focal_gamma: float = 2.0,      # 
                 weight_tversky: float = 0.7,
                 weight_focal: float = 0.3,
                 smooth: float = 1e-6):
        """
        : L = weight_tversky * Tversky(α,β) + weight_focal * Focal(α_pos, γ)
         Sigmoid ()；y_true  {0,1}.
        """
        super().__init__()
        self.w_seg = w_seg
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.weight_tversky = weight_tversky
        self.weight_focal = weight_focal
        self.smooth = smooth

    def _tversky_loss(self, y_true, y_pred):
        """
        y_true, y_pred: [B, D, H, W]
        Tversky = (TP + smooth) / (TP + α*FP + β*FN + smooth)
        Loss = 1 - Tversky
        """
        y_true = y_true.float()
        y_pred = y_pred.float()

        tp = torch.sum(y_true * y_pred, dim=[1, 2, 3])
        fp = torch.sum((1 - y_true) * y_pred, dim=[1, 2, 3])
        fn = torch.sum(y_true * (1 - y_pred), dim=[1, 2, 3])

        denom = tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.smooth
        tversky_index = (tp + self.smooth) / denom

        # ， (tversky_index≈1)
        # ，
        return 1 - tversky_index.mean()

    def _focal_loss(self, y_true, y_pred):
        """
         Focal Loss
        Loss = -α (1-p)^γ y log(p) - (1-α) p^γ (1-y) log(1-p)
        """
        eps = 1e-6
        y_true = y_true.float()
        p = y_pred.clamp(min=eps, max=1 - eps)

        loss_pos = -self.focal_alpha * (1 - p).pow(self.focal_gamma) * torch.log(p) * y_true
        loss_neg = -(1 - self.focal_alpha) * p.pow(self.focal_gamma) * torch.log(1 - p) * (1 - y_true)
        loss = (loss_pos + loss_neg).mean()
        return loss

    def forward(self, y_pred, y_true):
        """
        y_pred, y_true: [B, C, D, H, W]
        """
        assert y_pred.shape == y_true.shape, ""
        C = y_true.shape[1]
        total_loss = 0.0

        for c in range(C):
            gt_c = y_true[:, c]
            pred_c = y_pred[:, c]

            tversky_l = self._tversky_loss(gt_c, pred_c)
            focal_l = self._focal_loss(gt_c, pred_c)
            comb_l = self.weight_tversky * tversky_l + self.weight_focal * focal_l

            total_loss += comb_l

        total_loss /= C
        return self.w_seg * total_loss
class MyToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data

        # # ：
        # print(f"Original image type: {type(image)}, shape: {getattr(image, 'shape', 'N/A')}")
        # print(f"Original mask type: {type(mask)}, shape: {getattr(mask, 'shape', 'N/A')}")

        # #  PyTorch 
        # image = torch.tensor(image)
        # mask = torch.tensor(mask)
        image = torch.as_tensor(image, dtype=torch.float32)  #  float 
        mask = torch.as_tensor(mask, dtype=torch.float32)    # 

        # # ：
        # print(f"Converted image shape: {image.shape}")
        # print(f"Converted mask shape: {mask.shape}")

        return image, mask




# class MyResize:
#     def __init__(self, target_shape=(256, 512, 512)):
#         self.target_shape = target_shape
#         self.resize_transform = tio.Resize(self.target_shape)

#     def __call__(self, data):
#         image, mask = data

#         # # ：
#         # print("Before resizing:")
#         # print(f"Image shape: {image.shape}")
#         # print(f"Mask shape: {mask.shape}")

#         #  TorchIO  3D 
#         image = self.resize_transform(image)  #  image  3D resize
#         mask = self.resize_transform(mask)    #  mask  3D resize

#         # # ：
#         # print("After resizing:")
#         # print(f"Image shape: {image.shape}")
#         # print(f"Mask shape: {mask.shape}")

#         return image, mask


       
# class MyResample:
#     def __init__(self, target_spacing=(1.0, 1.0, 1.0)):
#         self.target_spacing = target_spacing

#     def __call__(self, data):
#         image, mask = data

#         # # 
#         # print("Before resampling:")
#         # print(f"Image shape: {image.shape}")
#         # print(f"Mask shape: {mask.shape}")

#         resample_transform = tio.Resample(self.target_spacing)

#         # 
#         image = resample_transform(image)
#         mask = resample_transform(mask)

#         # # 
#         # print("After resampling:")
#         # print(f"Image shape: {image.shape}")
#         # print(f"Mask shape: {mask.shape}")

#         return image, mask


class MyRandomFlip:
    def __init__(self, p=0.5, axes=(0, 1, 2)):
        self.p = p
        self.axes = axes

    def __call__(self, data):
        image, mask = data
        for axis in self.axes:
            if random.random() < self.p:
                print(f'axis:{axis}')
                image = np.flip(image, axis=axis)
                mask = np.flip(mask, axis=axis)
        return image, mask
        #  3D 

# class MyRandomRotation:
#     def __init__(self, p=0.5, degree=(-10, 10)):
#         self.p = p
#         self.degree = degree

#     def __call__(self, data):
#         image, mask = data
#         if random.random() < self.p:
#             angle = random.uniform(self.degree[0], self.degree[1])
#             #  torchio  3D 
#             rotate_transform = tio.RandomAffine(scales=1, degrees=angle)
#             image = rotate_transform(image)
#             mask = rotate_transform(mask)
#         return image, mask
# class myRandomHorizontalFlip:
#     def __init__(self, p=0.5):
#         self.p = p
#     def __call__(self, data):
#         image, mask = data
#         if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
#         else: return image, mask
            

# class myRandomVerticalFlip:
#     def __init__(self, p=0.5):
#         self.p = p
#     def __call__(self, data):
#         image, mask = data
#         if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
#         else: return image, mask


# class myRandomRotation:
#     def __init__(self, p=0.5, degree=[0,360]):
#         self.angle = random.uniform(degree[0], degree[1])
#         self.p = p
#     def __call__(self, data):
#         image, mask = data
#         if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
#         else: return image, mask 


class MyNormalize:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), window_min=-2000, window_max=1000):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.window_min = window_min
        self.window_max = window_max

    # Step 1: Adaptive windowing based on specified min and max values
    def adaptive_windowing(self,img, min_val=-2000, max_val=1000):
        # print(f"[DEBUG] Input type for adaptive_windowing: {type(img)}")
        # sys.exit()
        img_filtered = np.array(img)  # 
        first_min = img_filtered.min()
        img_filtered[img_filtered < first_min + 100] = np.nan
        min_display = np.nanpercentile(img_filtered, 0.5)
        max_display = np.nanpercentile(img_filtered, 99.5)
        img_windowed = np.clip(img, min_display, max_display)
        img_windowed = ((img_windowed - min_display) / (max_display - min_display) * 255).astype(np.uint8)
        return img_windowed


    # Step 2: Apply CLAHE to the windowed image
    def apply_clahe(self, img):
        # Apply CLAHE to each 2D slice (H, W) in the 3D array (D, H, W)
        clahe_slices = []
        for i in range(img.shape[0]):
            slice_img = img[i, :, :]
            # print(f"[DEBUG] Slice {i}: dtype={slice_img.dtype}, shape={slice_img.shape}, min={slice_img.min()}, max={slice_img.max()}")
            
            # Apply CLAHE to each slice
            clahe_img = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size).apply(slice_img)
            clahe_slices.append(clahe_img)
        
        img_clahe = np.stack(clahe_slices, axis=0)  # Stack processed slices back into (D, H, W)

        # # Debugging output after CLAHE
        # print(f"[DEBUG] Image after CLAHE: dtype={img_clahe.dtype}, shape={img_clahe.shape}, min={img_clahe.min()}, max={img_clahe.max()}")
        
        return img_clahe

    # Step 3: Normalize the CLAHE-adjusted image to [0, 1]
    def normalize(self, img):
        img_normalized = img.astype(np.float32) / 255.0
        # # Debugging output for normalized image
        # print(f"[DEBUG] Image after normalization: dtype={img_normalized.dtype}, shape={img_normalized.shape}, min={img_normalized.min()}, max={img_normalized.max()}")
        return img_normalized

    # Process method to apply the entire pipeline to an image and mask
    def __call__(self, data):
        image, mask = data
        # print(f"[DEBUG] Input data: image shape={image.shape}, mask shape={mask.shape}")

        # Remove singleton dimension (1, D, H, W) -> (D, H, W)
        image = np.squeeze(image, axis=0)
        # print(f"[DEBUG] Squeezed image: shape={image.shape}")
        # print(f"[DEBUG] Input type for adaptive_windowing: {type(image)}")

        # Apply adaptive windowing
        image_windowed = self.adaptive_windowing(image)

        # Apply CLAHE
        image_clahe = self.apply_clahe(image_windowed)

        # Normalize to [0, 1]
        image_normalized = self.normalize(image_clahe)

        # Add back the removed dimension to make (1, D, H, W)
        image_normalized = np.expand_dims(image_normalized, axis=0)

        # # Debugging output after adding back the dimension
        # print(f"[DEBUG] Final processed image shape: {image_normalized.shape}")

        return image_normalized, mask


    


from thop import profile		 ## thop
def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 
    print('params',params/1e6)			## 

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')






def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0



def test_single_volume(image, label, net, classes, patch_size=[256, 256], 
                    test_save_path=None, case=None, z_spacing=1, val_or_test=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and val_or_test is True:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
        # cv2.imwrite(test_save_path + '/'+case + '.png', prediction*255)
    return metric_list