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
import swanlab
swanlab.sync_wandb()
os.environ["WANDB_MODE"] = "offline"

warnings.filterwarnings("ignore")



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

    # 打印每个 GPU 的信息
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
    test_dataset = my_datasets(config.test_data_path, test=True)  # 加载测试集
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
        # 基础参数  
        in_chans=model_cfg['input_channels'],      # 输入通道数  
        num_classes=model_cfg['num_classes'],      # 分割输出类别数（替换原来的out_chans）  
        num_abnormal_classes=model_cfg['num_abnormal_classes'],  # 异常检测输出类别数（新增）  
        
        # 架构参数  
        depths=[2, 2, 2, 2],                      # 每个stage的TSMamba block数量  
        feat_size=[48, 96, 192, 384],             # 特征通道数配置  
        
        # 其他可选参数  
        drop_path_rate=0,                         # dropout率  
        layer_scale_init_value=1e-6,              # 层缩放初始值  
        hidden_size=768,                          # 隐藏层大小  
        norm_name="instance",                     # 归一化类型  
        conv_block=True,                          # 是否使用卷积块  
        res_block=True,                           # 是否使用残差块  
        spatial_dims=3,                           # 空间维度(3D)  
        
        # # 权重加载路径（如果需要）  
        # load_ckpt_path=model_cfg.get('load_ckpt_path', None)  # 可选的权重加载路径  
    )

    # 使用DataParallel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)  # 包裹模型，实现数据并行
        # print(model)

    # 检查模型是否在多卡模式下运行
    if isinstance(model, torch.nn.DataParallel):
        print("Model is wrapped in DataParallel.")
        logger.info("Model is using DataParallel for multi-GPU training.")
    else:
        print("Model is NOT wrapped in DataParallel, running on a single GPU.")
        logger.info("Model is running on a single GPU.")


    print('#----------Prepareing loss, opt, sch and amp----------#')
    # criterion = config.criterion
    segmentation_criterion= config.segmentation_criterion  # 分割任务损失函数
    abnormal_criterion= config.abnormal_criterion  # 异常检测任务损失函数
    # 确保传入 pos_weight（如果原先没有传入）
    pos_weight = torch.tensor([10.0], dtype=torch.float)  # 示例值
    pos_weight = pos_weight.to(device)
    abnormal_criterion.pos_weight = pos_weight  # 如果是 nn.BCEWithLogitsLoss 的实例，可以直接修改

    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)


    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))

        # 检查是否需要调整键前缀
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module') and not isinstance(model, torch.nn.DataParallel):
            # 如果加载的模型有“module.”前缀，但当前模型没有用DataParallel，则去掉“module.”前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not list(state_dict.keys())[0].startswith('module') and isinstance(model, torch.nn.DataParallel):
            # 如果加载的模型没有“module.”前缀，但当前模型用了DataParallel，则加上“module.”前缀
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        # 加载优化器和调度器的状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复其他参数
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

        # 训练阶段
        step = train_one_epoch(
            train_loader,
            model,
            segmentation_criterion,  # 分割任务损失函数
            abnormal_criterion,      # 异常检测任务损失函数
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            device
        )

        # 验证阶段
        loss,avg_auroc,best_thresholds = valid_one_epoch(
            val_loader,
            model,
            segmentation_criterion,  # 分割任务损失函数
            abnormal_criterion,      # 异常检测任务损失函数
            epoch,
            logger,
            config,
            writer,
            device
        )


        if avg_auroc > max_auroc:
            # 保存最佳模型，使用与其他检查点相同的格式  
            torch.save(  
                {  
                    'epoch': epoch,  
                    'avg_auroc': avg_auroc,   
                    'min_epoch': epoch,      # 达到最佳AUROC的epoch  
                    'loss': loss,  
                    'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),  
                    'optimizer_state_dict': optimizer.state_dict(),  
                    'scheduler_state_dict': scheduler.state_dict(),  
                    'best_thresholds':best_thresholds,
                }, os.path.join(checkpoint_dir, 'best.pth'))  
            max_auroc = avg_auroc
            min_epoch = epoch
        
        # 每5个epoch保存一次权重  
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
        # 加载检查点  
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))

        # 从检查点中获取模型权重  
        if torch.cuda.device_count() > 1:  
            model.module.load_state_dict(checkpoint['model_state_dict'])  
        else:  
            model.load_state_dict(checkpoint['model_state_dict'])  
        
        # 如果需要恢复训练时的epoch，可以从checkpoint中获取  
        # epoch = checkpoint['epoch']  
        # 如果只是测试，可以直接设置为0  
        epoch = checkpoint['epoch']
        best_thresholds = checkpoint['best_thresholds']

        print("Successfully loaded model weights")  

        # 验证阶段
        loss,avg_auroc,best_thresholds = valid_one_epoch(
            val_loader,
            model,
            segmentation_criterion,  # 分割任务损失函数
            abnormal_criterion,      # 异常检测任务损失函数
            epoch,
            logger,
            config,
            writer,
            device
        )

        test_loss = test_one_epoch(  
            test_loader,             # 确保是测试数据的 DataLoader  
            model,  
            segmentation_criterion,  # 分割任务损失函数  
            abnormal_criterion,      # 异常检测任务损失函数  
            epoch,  
            logger,  
            config,  
            writer,                  # 如果使用了 tensorboard 的 writer  
            device,  
            best_thresholds,         # 从验证集获取的最佳阈值  
            save_heatmap=True        # 是否保存热图，可选  
        )



if __name__ == '__main__':
    config = setting_config
    main(config)