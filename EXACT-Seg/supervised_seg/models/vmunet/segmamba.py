# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba_type="v3",
                # nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SegMamba(nn.Module):  
    def __init__(self, in_chans=1, num_classes=7, num_abnormal_classes=18, depths=[2, 2, 2, 2],   
                 feat_size=[48, 96, 192, 384], drop_path_rate=0, layer_scale_init_value=1e-6,   
                 hidden_size: int = 768, norm_name="instance", conv_block: bool = True,   
                 res_block: bool = True, spatial_dims=3) -> None:  
        super().__init__()  

        # 基础参数设置  
        self.hidden_size = hidden_size  
        self.in_chans = in_chans  
        self.num_classes = num_classes  
        self.num_abnormal_classes = num_abnormal_classes  
        self.depths = depths  
        self.drop_path_rate = drop_path_rate  
        self.feat_size = feat_size  
        self.layer_scale_init_value = layer_scale_init_value  
        self.spatial_dims = spatial_dims  

        # 共享编码器  
        self.vit = MambaEncoder(in_chans, depths=depths, dims=feat_size, drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value)  

        # 共享的基础编码块  
        self.encoder1 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.in_chans, out_channels=self.feat_size[0], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)  
        self.encoder2 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.feat_size[1], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)  
        self.encoder3 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.feat_size[2], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)  
        self.encoder4 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=self.feat_size[3], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)  
        self.encoder5 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=self.hidden_size, kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)  

        # 分割任务的解码器  
        self.seg_decoder5 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.hidden_size, out_channels=self.feat_size[3], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)  
        self.seg_decoder4 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=self.feat_size[2], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)  
        self.seg_decoder3 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=self.feat_size[1], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)  
        self.seg_decoder2 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.feat_size[0], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)  
        self.seg_decoder1 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.feat_size[0], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)  

        # 异常检测任务的解码器 - 注意调整了输入通道数以适应特征融合  
        self.abn_decoder5 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.hidden_size, out_channels=self.feat_size[3], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)  
        self.abn_decoder4 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[3]*2, out_channels=self.feat_size[2], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)  
        self.abn_decoder3 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[2]*2, out_channels=self.feat_size[1], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)  
        self.abn_decoder2 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[1]*2, out_channels=self.feat_size[0], kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=res_block)  
        self.abn_decoder1 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0]*2, out_channels=self.feat_size[0], kernel_size=3, stride=1, norm_name=norm_name, res_block=res_block)

        # 输出头  
        self.final_conv_segmentation = UnetOutBlock(spatial_dims=self.spatial_dims, in_channels=self.feat_size[0], out_channels=num_classes)  
        # 异常检测的两个输出头  
        self.final_conv_abnormal_low = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.num_abnormal_classes)  # 低分辨率输出  
        self.final_conv_abnormal_high = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.num_abnormal_classes)  # 高分辨率输出  
        
        # 激活函数  
        self.activation_segmentation = nn.Sigmoid()  
        self.activation_abnormal = nn.Sigmoid()

    def forward(self, x_in):  
        # 获取编码器特征  
        outs = self.vit(x_in)  
        
        # 共享编码器前向传播  
        enc1 = self.encoder1(x_in)  
        # print("enc1 shape:", enc1.shape)  
        x2 = outs[0]  
        # print("x2 shape:", x2.shape)  
        enc2 = self.encoder2(x2)  
        # print("enc2 shape:", enc2.shape)  
        x3 = outs[1]  
        # print("x3 shape:", x3.shape)  
        enc3 = self.encoder3(x3)  
        # print("enc3 shape:", enc3.shape)  
        x4 = outs[2]  
        # print("x4 shape:", x4.shape)    
        enc4 = self.encoder4(x4)  
        # print("enc4 shape:", enc4.shape)  
        enc_hidden = self.encoder5(outs[3])  
        # print("enc_hidden shape:", enc_hidden.shape)

        # 分割解码器分支  
        seg_dec4 = self.seg_decoder5(enc_hidden, enc4)  
        # print(f"seg_dec4 shape: {seg_dec4.shape}")  
        seg_dec3 = self.seg_decoder4(seg_dec4, enc3)  
        # print(f"seg_dec3 shape: {seg_dec3.shape}")  
        seg_dec2 = self.seg_decoder3(seg_dec3, enc2)  
        # print(f"seg_dec2 shape: {seg_dec2.shape}")  
        seg_dec1 = self.seg_decoder2(seg_dec2, enc1)  
        # print(f"seg_dec1 shape: {seg_dec1.shape}")  
        seg_out = self.seg_decoder1(seg_dec1)  
        # print(f"seg_out shape: {seg_out.shape}")  

        # 异常检测解码器分支 - 直接融合特征  
        abn_dec4 = self.abn_decoder5(enc_hidden, enc4)  
        # print(f"abn_dec4 shape: {abn_dec4.shape}")  
        fused_dec4 = torch.cat([abn_dec4, seg_dec4], dim=1)  
        # print(f"fused_dec4 shape: {fused_dec4.shape}")  
        abn_dec3 = self.abn_decoder4(fused_dec4, enc3)  
        # print(f"abn_dec3 shape: {abn_dec3.shape}")  
        fused_dec3 = torch.cat([abn_dec3, seg_dec3], dim=1)  
        # print(f"fused_dec3 shape: {fused_dec3.shape}")  
        abn_dec2 = self.abn_decoder3(fused_dec3, enc2)  
        # print(f"abn_dec2 shape: {abn_dec2.shape}")  
        fused_dec2 = torch.cat([abn_dec2, seg_dec2], dim=1)  
        # print(f"fused_dec2 shape: {fused_dec2.shape}")  
        abnormal_output_low = self.activation_abnormal(self.final_conv_abnormal_low(abn_dec2))  
        # print(f"abnormal_output_low shape: {abnormal_output_low.shape}")  

        # 继续上采样得到高分辨率输出  
        abn_dec1 = self.abn_decoder2(fused_dec2, enc1)  
        # print(f"abn_dec1 shape: {abn_dec1.shape}")  
        fused_dec1 = torch.cat([abn_dec1, seg_dec1], dim=1)  
        # print(f"fused_dec1 shape: {fused_dec1.shape}")  
        abn_out = self.abn_decoder1(fused_dec1)  
        # print(f"abn_out shape: {abn_out.shape}")  
        abnormal_output_high = self.activation_abnormal(self.final_conv_abnormal_high(abn_out))   
        # print(f"abnormal_output_high shape: {abnormal_output_high.shape}")  

        # 最终输出  
        segmentation_output = self.activation_segmentation(  
            self.final_conv_segmentation(seg_out)  
        )  
        # print(f"segmentation_output shape: {segmentation_output.shape}") 
        
        return (  
        segmentation_output,  
        [abnormal_output_low, abnormal_output_high]  # 返回低分辨率和高分辨率的异常检测结果  
    ) 

