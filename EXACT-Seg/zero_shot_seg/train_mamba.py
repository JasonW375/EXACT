# /path/to/lmx/anaconda3/envs/CT2/bin/python /FM_data/bxg/CT_Report/CT_Report9_test/train_mamba.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset 
import timm
from datasets.dataset import NW_datasets
from tensorboardX import SummaryWriter
from models.vmunet.segmamba import SegMamba

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config
import random
import warnings

warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    # wandb.init(project="CT_Report", config={"epochs": config.epochs, "batch_size": config.batch_size, "learning_rate": config.lr})
    print("============================")
    print(config.work_dir)
    print("============================")
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir="/path/to/checkpoint_dir/"
    resume_model = os.path.join(checkpoint_dir, 'best.pth')
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

    # Print information for each GPU.
    for i in range(num_gpus):
        gpu_properties = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {gpu_properties.name}, Memory: {gpu_properties.total_memory / (1024 ** 2):.2f} MB')


    print('#----------Preparing dataset----------#')
    train_dataset = NW_datasets(config.train_data_path, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = NW_datasets(config.train_data_path, val=True)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    test_dataset = NW_datasets(config.test_data_path, test=True)  # Load test set.
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

    model = SegMamba(
        # Core parameters.
        in_chans=model_cfg['input_channels'],      # Number of input channels.
        num_classes=model_cfg['num_classes'],      # Number of segmentation output classes.
        num_abnormal_classes=model_cfg['num_abnormal_classes'],  # Number of abnormality output classes.

        # Architecture parameters.
        depths=[2, 2, 2, 2],                      # Number of TSMamba blocks per stage.
        feat_size=[48, 96, 192, 384],             # Feature channel configuration.

        # Optional parameters.
        drop_path_rate=0,                         # Dropout rate.
        layer_scale_init_value=1e-6,              # Initial value for layer scaling.
        hidden_size=768,                          # Hidden size.
        norm_name="instance",                     # Normalization type.
        conv_block=True,                          # Whether to use convolution blocks.
        res_block=True,                           # Whether to use residual blocks.
        spatial_dims=3,                           # Spatial dimensions (3D).
    )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)  # Wrap model with DataParallel.
        # print(model)

    # Check whether the model is running in multi-GPU mode.
    if isinstance(model, torch.nn.DataParallel):
        print("Model is wrapped in DataParallel.")
        logger.info("Model is using DataParallel for multi-GPU training.")
    else:
        print("Model is NOT wrapped in DataParallel, running on a single GPU.")
        logger.info("Model is running on a single GPU.")


    print('#----------Prepareing loss, opt, sch and amp----------#')
    # criterion = config.criterion
    segmentation_criterion= config.segmentation_criterion  # Segmentation loss function.
    abnormal_criterion= config.abnormal_criterion  # Abnormality detection loss function.
    # Ensure pos_weight is provided if it is missing.
    pos_weight = torch.tensor([10.0], dtype=torch.float)  # Example value.
    pos_weight = pos_weight.to(device)
    abnormal_criterion.pos_weight = pos_weight  # Safe when using nn.BCEWithLogitsLoss.

    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)


    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))

        # Check whether state_dict key prefixes need to be adjusted.
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module') and not isinstance(model, torch.nn.DataParallel):
            # Remove "module." prefix if checkpoint was saved with DataParallel.
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not list(state_dict.keys())[0].startswith('module') and isinstance(model, torch.nn.DataParallel):
            # Add "module." prefix if current model is wrapped with DataParallel.
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        # Load optimizer and scheduler states.
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore other training states.
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        # min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}'
        logger.info(log_info)
        print(log_info)


    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
    # if os.path.exists('/path/to/lmx/work/Class_projects/bxg/CT_Report/CT_Report8_16abn_2decoder/results/segmamba__Saturday_04_January_2025_15h_02m_55s/checkpoints/best.pth'):  
        print('#----------Testing----------#')  
        # Load checkpoint.
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
        # checkpoint = torch.load('/path/to/lmx/work/Class_projects/bxg/CT_Report/CT_Report8_16abn_2decoder/results/segmamba__Saturday_04_January_2025_15h_02m_55s/checkpoints/best.pth',   
                            # map_location=torch.device('cpu'))  

        # Load model weights from checkpoint.
        if torch.cuda.device_count() > 1:  
            model.module.load_state_dict(checkpoint['model_state_dict'])  
        else:  
            model.load_state_dict(checkpoint['model_state_dict'])  
        
        # If needed, recover the training epoch from checkpoint.
        # epoch = checkpoint['epoch']  
        # For test-only runs, epoch can be set manually.
        epoch = checkpoint['epoch']
      
        file_path="/path/to/best_thresholds.npy"
        best_thresholds = np.load(file_path)  
        # print("best thres",best_thresholds)
        # assert False
        test_loss = test_one_epoch(
            test_loader,             # Ensure this is the test DataLoader.
            model,  
            segmentation_criterion,  # Segmentation loss function.
            abnormal_criterion,      # Abnormality detection loss function.
            epoch,  
            logger,  
            config,  
            writer,                  # TensorBoard writer (if enabled).
            device,  
            best_thresholds,         # Best thresholds from validation.
            save_heatmap=True        # Whether to save heatmaps (optional).
        )



if __name__ == '__main__':
    config = setting_config
    main(config)