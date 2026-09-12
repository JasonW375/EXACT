import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset 
import timm
from datasets.dataset import my_datasets
from tensorboardX import SummaryWriter
from models.ymamba.ymamba import YMamba

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config
import warnings

warnings.filterwarnings("ignore")


def setup_tracking(backend):
    """Configure optional metric streaming.

    Off by default: wandb runs in "disabled" mode, which turns every wandb.log
    and wandb.Image call in engine.py into a no-op, so a fresh clone trains
    without an account, an API key or network access. Pass --track to opt in.
    """
    if backend == "none":
        os.environ["WANDB_MODE"] = "disabled"
    elif backend == "offline":
        os.environ["WANDB_MODE"] = "offline"
    elif backend == "wandb":
        os.environ.setdefault("WANDB_MODE", "online")
    elif backend == "swanlab":
        import swanlab
        swanlab.sync_wandb()
        os.environ.setdefault("WANDB_MODE", "online")


def main(config):

    print('#----------Creating logger----------#')
    wandb.init(project="CT_Report", config={"epochs": config.epochs, "batch_size": config.batch_size, "learning_rate": config.lr})
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
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

    # Report what each GPU is
    for i in range(num_gpus):
        gpu_properties = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {gpu_properties.name}, Memory: {gpu_properties.total_memory / (1024 ** 2):.2f} MB')






    print('#----------Preparing dataset----------#')
    train_dataset = my_datasets(config.train_data_path, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = my_datasets(config.train_data_path, val=True)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    test_dataset = my_datasets(config.test_data_path, test=True)
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
    model = YMamba(
        # Basic parameters
        in_chans=model_cfg['input_channels'],      # input channels
        num_classes=model_cfg['num_classes'],      # segmentation output classes
        num_abnormal_classes=model_cfg['num_abnormal_classes'],  # abnormality classes

        # Architecture
        depths=[2, 2, 2, 2],                      # TSMamba blocks per stage
        feat_size=[48, 96, 192, 384],             # feature channels per stage

        # Optional
        drop_path_rate=0,                         # drop path rate
        layer_scale_init_value=1e-6,              # layer scale init value
        hidden_size=768,                          # hidden size
        norm_name="instance",                     # normalisation
        conv_block=True,                          # use convolution blocks
        res_block=True,                           # use residual blocks
        spatial_dims=3,                           # 3D
    )

    # Wrap for multi-GPU training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Report which mode the model ended up in
    if isinstance(model, torch.nn.DataParallel):
        print("Model is wrapped in DataParallel.")
        logger.info("Model is using DataParallel for multi-GPU training.")
    else:
        print("Model is NOT wrapped in DataParallel, running on a single GPU.")
        logger.info("Model is running on a single GPU.")


    print('#----------Prepareing loss, opt, sch and amp----------#')
    segmentation_criterion = config.segmentation_criterion
    abnormal_criterion = config.abnormal_criterion
    # Positive-class weight for the abnormality head, in case the config left it unset
    pos_weight = torch.tensor([10.0], dtype=torch.float)
    pos_weight = pos_weight.to(device)
    abnormal_criterion.pos_weight = pos_weight

    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)


    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))

        # The checkpoint may or may not carry the DataParallel "module." prefix
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module') and not isinstance(model, torch.nn.DataParallel):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not list(state_dict.keys())[0].startswith('module') and isinstance(model, torch.nn.DataParallel):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        # Optimizer and scheduler state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Remaining bookkeeping
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        # min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}'
        logger.info(log_info)
        print(log_info)





    step = 0
    avg_auroc=0
    max_auroc=0
    print('#   ----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        # Train
        step = train_one_epoch(
            train_loader,
            model,
            segmentation_criterion,
            abnormal_criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            device
        )

        # Validate
        loss,avg_auroc,best_thresholds = valid_one_epoch(
            val_loader,
            model,
            segmentation_criterion,
            abnormal_criterion,
            epoch,
            logger,
            config,
            writer,
            device
        )


        if avg_auroc > max_auroc:
            # Best model so far, saved in the same format as the other checkpoints
            torch.save(  
                {  
                    'epoch': epoch,  
                    'avg_auroc': avg_auroc,   
                    'min_epoch': epoch,      # epoch that reached the best AUROC
                    'loss': loss,  
                    'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),  
                    'optimizer_state_dict': optimizer.state_dict(),  
                    'scheduler_state_dict': scheduler.state_dict(),  
                    'best_thresholds':best_thresholds,
                }, os.path.join(checkpoint_dir, 'best.pth'))  
            max_auroc = avg_auroc
            min_epoch = epoch
        
        # Periodic checkpoint
        if epoch % config.save_interval == 0:  
            torch.save(  
                {  
                    'epoch': epoch,  
                    'avg_auroc': avg_auroc,   
                    'loss': loss,  
                    'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),  
                    'optimizer_state_dict': optimizer.state_dict(),  
                    'scheduler_state_dict': scheduler.state_dict(),  
                    'best_thresholds':best_thresholds,
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')) 

        torch.save(
            {
                'epoch': epoch,
                'max_auroc': max_auroc,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_thresholds':best_thresholds
            }, os.path.join(checkpoint_dir, 'latest.pth')) 


    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')  
        # Load the best checkpoint back
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))

        # Restore the weights
        if torch.cuda.device_count() > 1:  
            model.module.load_state_dict(checkpoint['model_state_dict'])  
        else:  
            model.load_state_dict(checkpoint['model_state_dict'])  
        
        epoch = checkpoint['epoch']
        best_thresholds = checkpoint['best_thresholds']

        print("Successfully loaded model weights")  

        # Validate
        loss,avg_auroc,best_thresholds = valid_one_epoch(
            val_loader,
            model,
            segmentation_criterion,
            abnormal_criterion,
            epoch,
            logger,
            config,
            writer,
            device
        )

        test_loss = test_one_epoch(
            test_loader,
            model,
            segmentation_criterion,
            abnormal_criterion,
            epoch,
            logger,
            config,
            writer,
            device,
            best_thresholds,         # thresholds picked on the validation split
            save_heatmap=True
        )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 1 pre-training of the Y-Mamba backbone.")
    parser.add_argument("--track", default="none",
        choices=["none", "offline", "wandb", "swanlab"],
        help="Where to stream metrics. Default none: no account or "
            "network needed. offline buffers a wandb run on disk.")
    args = parser.parse_args()
    setup_tracking(args.track)
    config = setting_config
    main(config)