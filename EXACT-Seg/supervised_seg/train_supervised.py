# /data/birth/lmx/anaconda3/envs/CT2/bin/python /FM_data/bxg/CT_Report/CT_Report15_18abn_2decoder/train_supervised.py
import torch  
import torch.nn as nn  
from torch.utils.data import DataLoader, Subset, Dataset  
import timm  
import h5py  
from tensorboardX import SummaryWriter  
from models.vmunet.vmunet import VMUNet  
from models.vmunet.segmamba import SegMamba  
from sklearn.model_selection import KFold  
import numpy as np  
import torch.nn.functional as F  
from tqdm import tqdm  
import os  
import sys  
import uuid  
import time  
import json
import wandb  
from pathlib import Path
from datetime import datetime

from datasets.dataset import NW_datasets_5fold  ,NW_datasets_supervised
from engine_supervised import *  

from utils import *  
from configs.config_setting import setting_config  
import random  
import warnings  
import swanlab

warnings.filterwarnings("ignore")

from monai.networks.blocks.dynunet_block import UnetOutBlock
def _strip_prefix(state_dict, prefix="module."):
    if len(state_dict) == 0:
        return state_dict
    if list(state_dict.keys())[0].startswith(prefix):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def _pick_encoder_weights(sd):
    # Compatible with two checkpoint layouts: vit./encoderX. at root, or prefixed with backbone.
    sd = _strip_prefix(sd, "module.")
    has_backbone = any(k.startswith("backbone.") for k in sd.keys())
    encoder_prefixes = ["vit.", "encoder1.", "encoder2.", "encoder3.", "encoder4.", "encoder5."]

    picked = {}
    for k, v in sd.items():
        k2 = k
        if has_backbone:
            if not k.startswith("backbone."):
                continue
            k2 = k[len("backbone."):]  # Remove backbone. prefix for direct submodule loading.
        if any(k2.startswith(p) for p in encoder_prefixes):
            picked[k2] = v
    return picked

def load_encoder_only_into_wrapper(model_wrapper, ckpt_state_dict, logger=None):
    """
    Load only encoder weights (vit, encoder1~encoder5) into SegMambaSegOnly.backbone.
    Supports wrappers under DataParallel.
    """
    enc_sd = _pick_encoder_weights(ckpt_state_dict)

    target = model_wrapper.module.backbone if isinstance(model_wrapper, nn.DataParallel) else model_wrapper.backbone
    incompatible = target.load_state_dict(enc_sd, strict=False)

    msg = (f"Loaded encoder weights: {len(enc_sd)} params; "
           f"missing: {len(incompatible.missing_keys)}, unexpected: {len(incompatible.unexpected_keys)}")
    print(msg)
    if logger is not None:
        logger.info(msg)
class SegMambaSegOnly(nn.Module):
    """
    Wrap a SegMamba model and use only its encoder and segmentation decoder.
    The forward pass returns only segmentation_output.
    When freeze_backbone=True, encoder params are frozen (requires_grad=False)
    without no_grad, so graph construction and gradient propagation are preserved.
    """
    def __init__(self, segmamba: SegMamba, freeze_backbone: bool = False):
        super().__init__()
        # Accept either a DataParallel wrapper or a plain model.
        self.backbone = segmamba.module if isinstance(segmamba, nn.DataParallel) else segmamba
        self.freeze_backbone = freeze_backbone

        # If freezing is enabled, freeze encoder modules (including vit) only.
        # Do not switch to eval mode and do not use no_grad.
        if self.freeze_backbone:
            self._set_encoder_requires_grad(False)
   
            
        # Output head (kept trainable)
        ref = next(self.backbone.parameters())
        self.final_conv_segmentation = UnetOutBlock(
            spatial_dims=self.backbone.spatial_dims,
            in_channels=self.backbone.feat_size[0],
            out_channels=1
        ).to(ref.device)
        self.final_layer=nn.Conv3d(segmamba.num_abnormal_classes,1,kernel_size=1).to(ref.device)
    def _encoder_modules(self):
        mods = []
        for name in ["vit", "encoder1", "encoder2", "encoder3", "encoder4", "encoder5"]:
            if hasattr(self.backbone, name):
                mods.append(getattr(self.backbone, name))
        return mods

    def _set_encoder_requires_grad(self, flag: bool):
        for m in self._encoder_modules():
            for p in m.parameters():
                p.requires_grad = flag

    def forward(self, x: torch.Tensor):
        # Always allow gradient propagation; do not use torch.no_grad().
        bb = self.backbone
        outs = bb.vit(x)
        enc1 = bb.encoder1(x)
        enc2 = bb.encoder2(outs[0])
        enc3 = bb.encoder3(outs[1])
        enc4 = bb.encoder4(outs[2])
        enc_hidden = bb.encoder5(outs[3])

        seg_dec4 = bb.seg_decoder5(enc_hidden, enc4)
        seg_dec3 = bb.seg_decoder4(seg_dec4, enc3)
        seg_dec2 = bb.seg_decoder3(seg_dec3, enc2)
        seg_dec1 = bb.seg_decoder2(seg_dec2, enc1)
        seg_out  = bb.seg_decoder1(seg_dec1)

        seg=bb.activation_segmentation(self.final_layer(bb.final_conv_abnormal_high(seg_out)))
        return seg

def main(config,args):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir=config.work_dir + '/checkpoints/'

    resume_model=args.resume_model  
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)


    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

    print('#----------Checking available GPUs----------#')
    num_gpus = torch.cuda.device_count()
    print(f'Number of available GPUs: {num_gpus}')

    # Print info for each GPU.
    for i in range(num_gpus):
        gpu_properties = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {gpu_properties.name}, Memory: {gpu_properties.total_memory / (1024 ** 2):.2f} MB')






    print('#----------Preparing dataset----------#')
    train_dataset = NW_datasets_supervised(config.train_data_path, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NW_datasets_supervised(config.train_data_path, val=True)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    test_dataset = NW_datasets_supervised(config.test_data_path, test=True)  # Load test set.
    # test_dataset= NW_datasets_supervised(config.train_data_path, val=True)  # Load test set.
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=config.num_workers,
                            drop_last=False)

    print('#----------Preparing Model----------#')
    model_cfg = {
        'num_classes': config.num_classes,
        'num_abnormal_classes': config.num_abnormal_classes,
        'input_channels': config.input_channels,
        'depths': config.model_depth,
        'n_base_filters': config.n_base_filters,
        'batch_normalization': True,
        'load_ckpt_path': None
    }


    full_model = SegMamba(  
        # Basic parameters
        in_chans=model_cfg['input_channels'],      # Number of input channels
        num_classes=model_cfg["num_classes"], 
        num_abnormal_classes=model_cfg['num_abnormal_classes'],  # Number of abnormal-detection output classes
        
        # Architecture parameters
        depths=[2, 2, 2, 2],                      # Number of TSMamba blocks per stage
        feat_size=[48, 96, 192, 384],             # Feature channel configuration
        
        # Other optional parameters
        drop_path_rate=0,                         # Dropout rate
        layer_scale_init_value=1e-6,              # Initial value for layer scaling
        hidden_size=768,                          # Hidden size
        norm_name="instance",                     # Normalization type
        conv_block=True,                          # Whether to use convolution blocks
        res_block=True,                           # Whether to use residual blocks
        spatial_dims=3,                           # Spatial dimensions (3D)
        
        # # Checkpoint path (if needed)
        # load_ckpt_path=model_cfg.get('load_ckpt_path', None)  # Optional checkpoint path
    )



    # Use DataParallel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_model = full_model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        full_model = nn.DataParallel(full_model)  # Wrap model for data parallelism.
        # print(model)

    # Check whether the model is running in multi-GPU mode.
    if isinstance(full_model, torch.nn.DataParallel):
        print("Model is wrapped in DataParallel.")
        logger.info("Model is using DataParallel for multi-GPU training.")
    else:
        print("Model is NOT wrapped in DataParallel, running on a single GPU.")
        logger.info("Model is running on a single GPU.")

    # Setup swanlab and wandb with a project name based on the network name and timestamp
    # network name: handle DataParallel wrapper if present
    network_name = full_model.module.__class__.__name__ if isinstance(full_model, nn.DataParallel) else full_model.__class__.__name__
    project_name = network_name + '__' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')

    # Use swanlab.init to set target project, keep experiment name as project_name, and sync with wandb.
    swanlab.init(
        project="CT_Report",
        workspace="meixixixi",
        experiment_name=project_name,  # Keep your original naming.
        config={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.lr
        },
        sync_wandb=True
    )

    # Keep your original wandb naming and project behavior.
    wandb.init(
        project=project_name,
        config={"epochs": config.epochs, "batch_size": config.batch_size, "learning_rate": config.lr}
    )
    print('#----------Prepareing loss, opt, sch and amp----------#')
    # criterion = config.criterion
    segmentation_criterion= config.segmentation_criterion  # Segmentation loss function

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    if args.task=="test":
        assert os.path.exists(resume_model), f"Resume model path does not exist: {resume_model}"
        print('#----------Loading model for testing----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))

        # Check whether state_dict key prefixes need adjustment.
        state_dict = checkpoint['model_state_dict']

        if list(state_dict.keys())[0].startswith('module') and not isinstance(full_model, torch.nn.DataParallel):
            # If checkpoint keys have "module." but current model is not DataParallel, strip the prefix.
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not list(state_dict.keys())[0].startswith('module') and isinstance(full_model, torch.nn.DataParallel):
            # If checkpoint keys do not have "module." but current model is DataParallel, add the prefix.
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        
        model=SegMambaSegOnly(full_model)  # Wrap model to output segmentation only.
        model.load_state_dict(state_dict)
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)  # Wrap model for data parallelism.
        print(f'Model loaded from {resume_model} for testing.')
        epoch_for_log=0

        _ = test_one_epoch(
            test_loader,
            model,
            config.segmentation_criterion,  # Segmentation loss function
            epoch_for_log,
            logger,
            config,
            writer,
            device,
            save_heatmap=True,        # Whether to save heatmaps (optional)
        )

        return

    # assert os.path.exists(resume_model), f"Resume model path does not exist: {resume_model}"
    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))

        # Check whether state_dict key prefixes need adjustment.
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module') and not isinstance(full_model, torch.nn.DataParallel):
            # If checkpoint keys have "module." but current model is not DataParallel, strip the prefix.
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not list(state_dict.keys())[0].startswith('module') and isinstance(full_model, torch.nn.DataParallel):
            # If checkpoint keys do not have "module." but current model is DataParallel, add the prefix.
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        
        # model=SegMambaSegOnly(full_model, freeze_backbone=False)  # Wrap model to output segmentation only.
        # load_encoder_only_into_wrapper(model, state_dict, logger)
        # Load optimizer and scheduler states.
        
        
        # Restore other training parameters.
        weight_only = args.weight_only

        if not weight_only:
            model=SegMambaSegOnly(full_model, freeze_backbone=args)
            model.load_state_dict(state_dict)
            optimizer = get_optimizer(config, model)
            scheduler = get_scheduler(config, optimizer)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Load optimizer and scheduler states if available
            try:
                if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # Move optimizer state tensors to the current device
                    for state in optimizer.state.values():
                        for k, v in list(state.items()):
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load optimizer/scheduler state: {e}")

            start_epoch=checkpoint['epoch'] + 1
            step=checkpoint['global_step'] if 'global_step' in checkpoint else 0
            min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        else:
            full_model.load_state_dict(state_dict)
            model=SegMambaSegOnly(full_model, freeze_backbone=args.freeze)

            optimizer = get_optimizer(config, model)
            scheduler = get_scheduler(config, optimizer)
            # Only load weights, keep optimizer/scheduler and training counters as initialized
            start_epoch=0
            step=0
            logger.info("weight_only=True: loaded model weights only, skipping optimizer/scheduler and epoch/metric restore.")
            min_loss=999
            min_epoch=0


        log_info = f'resuming model from {resume_model}.'
        logger.info(log_info)
        print(log_info)


    else:
        model=SegMambaSegOnly(full_model, freeze_backbone=False)  # Wrap model to output segmentation only.
        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer)
        epoch=0
        step=0
    step = 0

    

    print('#   ----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        # Training phase
        step = train_one_epoch(
            train_loader,
            model,
            segmentation_criterion,  # Segmentation loss function
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            device
        )

        # Validation phase
        loss = valid_one_epoch(
            val_loader,
            model,
            segmentation_criterion,  # Segmentation loss function
            epoch,
            logger,
            config,
            writer,
            device
        )

        # Save best model based on minimum validation loss.
        if loss < min_loss:
            min_loss = loss
            min_epoch = epoch
            # Save best model in the same format as other checkpoints.
            torch.save(
                {
                    'epoch': epoch,
                    'min_epoch': min_epoch,   # Epoch that achieved the best (minimum) loss.
                    'min_loss': min_loss,
                    'loss': loss,
                    'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, 'best.pth'))
        
        # Save checkpoint every 5 epochs.
        if epoch % config.save_interval == 0:  
            torch.save(  
                {  
                    'epoch': epoch,    
                    'loss': loss,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),  
                    'optimizer_state_dict': optimizer.state_dict(),  
                    'scheduler_state_dict': scheduler.state_dict(),  
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')) 

        torch.save(
            {
                'epoch': epoch,
                'min_epoch': min_epoch,
                'min_loss': min_loss,
                'loss': loss,
                'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),

            }, os.path.join(checkpoint_dir, 'latest.pth')) 


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train', choices=['train', 'test'],
                        help='train: train and validate; test: load weights and run test only')
    parser.add_argument("--freeze",type=bool,default=False,help="Whether to freeze encoder weights")
    parser.add_argument("--weight_only",type=bool,default=True,help="Load weights only, without restoring optimizer/scheduler states")
    parser.add_argument("--resume_model",type=str,default="/path/to/pretrained_model.pth",help="Path to pretrained model. Leave empty to skip loading")

    args = parser.parse_args()
    config = setting_config
    main(config, args)