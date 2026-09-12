# llava/model/multimodal_encoder/heatmap_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ymamba import YMamba


class LightweightHeatmapEncoder(nn.Module):
    """
    Lightweight heatmap feature extractor based on YMamba encoder.
    Uses only the first three encoder stages to reduce computation.
    """
    def __init__(self, ymamba_checkpoint_path, freeze=True,in_channels=1):
        super().__init__()
        
        # Build a full YMamba model and reuse its encoder blocks.
        full_model = YMamba(
            in_chans=in_channels,
            num_classes=7,
            num_abnormal_classes=18,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size=768,
            norm_name="instance",
            conv_block=True,
            res_block=True,
            spatial_dims=3,
        )
        

        self.vit = full_model.vit
        self.encoder1 = full_model.encoder1
        self.encoder2 = full_model.encoder2
        self.encoder3 = full_model.encoder3
        
        # Feature projection: 192 -> 768
        self.feature_proj = nn.Sequential(
            nn.Conv3d(192, 384, kernel_size=1),
            nn.InstanceNorm3d(384),
            nn.GELU(),
            nn.Conv3d(384, 768, kernel_size=1),
        )

        if freeze:
            for module in [self.vit, self.encoder1, self.encoder2, self.encoder3]:
                for param in module.parameters():
                    param.requires_grad = False
        
        self.fused_dim = 768
    
    def forward(self, x):
        """
        Input: [B, 1, 64, 128, 128]
        Output: [B, 16, 32, 32, 768]
        """
        # ===== Important fix: handle vit output format correctly =====
        with torch.no_grad():
            vit_output = self.vit(x)
            
            # Validate vit output type
            if isinstance(vit_output, tuple):
                # Ensure at least two outputs are returned
                if len(vit_output) < 2:
                    raise ValueError(
                        f"Expected vit to return at least 2 outputs, got {len(vit_output)}"
                    )
                outs = vit_output
            else:
                # If only one tensor is returned, the architecture does not match expectation
                raise TypeError(
                    f"Expected vit to return tuple, got {type(vit_output)}. "
                    f"Check your YMamba model architecture."
                )
            
            # Encoder forward
            enc1 = self.encoder1(x)         # [B, 48, 32, 64, 64]
            enc2 = self.encoder2(outs[0])   # [B, 96, 16, 32, 32]
            enc3 = self.encoder3(outs[1])   # [B, 192, 16, 32, 32]
        # =============================================
        
        # Project to 768-dim features
        features = self.feature_proj(enc3)  # [B, 768, 16, 32, 32]
        
        # Reorder dims: [B, 768, 16, 32, 32] -> [B, 16, 32, 32, 768]
        features = features.permute(0, 2, 3, 4, 1)
        
        return features


class HeatmapFeatureExtractor(nn.Module):
    """Heatmap encoder that preserves full spatial information."""
    def __init__(self, ymamba_checkpoint_path):
        super().__init__()
        self.encoder = LightweightHeatmapEncoder(ymamba_checkpoint_path, freeze=True)
        self.output_dim = 768  # keep feature dimension unchanged
        # Do not apply global pooling here.
    
    def forward(self, heatmaps):
        """
        Input: [B, 18, 1, 64, 128, 128]
        Output: [B, 18, 16, 32, 32, 768] (spatial dimensions preserved)
        """
        B, num_diseases = heatmaps.shape[0], heatmaps.shape[1]
        
        all_features = []
        for disease_idx in range(num_diseases):
            single_heatmap = heatmaps[:, disease_idx, :, :, :, :]
            features = self.encoder(single_heatmap)  # [B, 16, 32, 32, 768]
            all_features.append(features)
        
        # [B, 18, 16, 32, 32, 768] - keep full spatial information
        return torch.stack(all_features, dim=1)