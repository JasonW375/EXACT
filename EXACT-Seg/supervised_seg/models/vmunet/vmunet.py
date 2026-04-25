from .vmamba import VSSM
import torch
from torch import nn
import torch.nn.functional as F


import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class VMUNet(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 num_classes=6,  # 分割6个解剖区域
                 num_abnormal_classes=17,  # 17个区域的异常检测（每个区域二分类）
                 depths=4, 
                 n_base_filters=32,
                 batch_normalization=False,
                 load_ckpt_path=None,
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.num_abnormal_classes = num_abnormal_classes
        self.depth = depths
        self.n_base_filters = n_base_filters
        self.batch_normalization = batch_normalization
        
        # 编码器和解码器逻辑保持不变
        self.encoder_levels = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        current_layer = input_channels
        
        for layer_depth in range(self.depth):
            layer1 = self.create_convolution_block(current_layer, n_base_filters * (2 ** layer_depth), batch_normalization)
            layer2 = self.create_convolution_block(n_base_filters * (2 ** layer_depth), n_base_filters * (2 ** layer_depth) * 2, batch_normalization)
            self.encoder_levels.append(nn.Sequential(layer1, layer2))
            if layer_depth < self.depth - 1:
                self.pooling_layers.append(nn.MaxPool3d(kernel_size=2))
            current_layer = n_base_filters * (2 ** layer_depth) * 2

        self.decoder_levels = nn.ModuleList()
        for layer_depth in range(self.depth - 2, -1, -1):
            up_convolution = nn.ConvTranspose3d(current_layer, n_base_filters * (2 ** layer_depth) * 2, kernel_size=2, stride=2)
            self.decoder_levels.append(up_convolution)
            current_layer = n_base_filters * (2 ** layer_depth) * 2
            layer1 = self.create_convolution_block(current_layer + n_base_filters * (2 ** layer_depth) * 2, n_base_filters * (2 ** layer_depth) * 2, batch_normalization)
            layer2 = self.create_convolution_block(n_base_filters * (2 ** layer_depth) * 2, n_base_filters * (2 ** layer_depth), batch_normalization)
            self.decoder_levels.append(nn.Sequential(layer1, layer2))
            current_layer = n_base_filters * (2 ** layer_depth)

        # 输出头
        self.final_conv_segmentation = nn.Conv3d(current_layer, num_classes, kernel_size=1)  # 分割9个解剖区域
        self.final_conv_abnormal = nn.Conv3d(current_layer, num_abnormal_classes, kernel_size=1)  # 每个区域的异常检测（二分类）

        # self.activation_segmentation = nn.Softmax(dim=1)  # 多类别分割
        self.activation_segmentation = nn.Sigmoid()  # 每个类别独立判定
        self.activation_abnormal = nn.Sigmoid()  # 异常检测的二分类

    def create_convolution_block(self, in_channels, out_channels, batch_normalization):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if batch_normalization:
            layers.append(nn.BatchNorm3d(out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 编码器部分
        encoder_outputs = []
        for level_idx, (level, pool) in enumerate(zip(self.encoder_levels, self.pooling_layers + [None])):
            x = level(x)
            encoder_outputs.append(x)
            if pool is not None:
                x = pool(x)

        # 解码器部分
        for idx in range(0, len(self.decoder_levels), 2):
            up_convolution = self.decoder_levels[idx]
            convolution_block = self.decoder_levels[idx + 1]
            x = up_convolution(x)
            
            encoder_idx = self.depth - 2 - (idx // 2)
            enc_output = encoder_outputs[encoder_idx]
            if x.shape[2:] != enc_output.shape[2:]:
                enc_output = F.interpolate(enc_output, size=x.shape[2:], mode='trilinear', align_corners=False)

            x = torch.cat((x, enc_output), dim=1)
            x = convolution_block(x)

        # 输出
        segmentation_output = self.final_conv_segmentation(x)  # 分割输出
        abnormal_output = self.final_conv_abnormal(x)  # 异常检测输出

        return self.activation_segmentation(segmentation_output), self.activation_abnormal(abnormal_output)


    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("Model weights loaded finished!")

