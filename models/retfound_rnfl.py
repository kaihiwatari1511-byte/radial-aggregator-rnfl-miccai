"""
models/retfound_rnfl.py
RETFound-based RNFL thickness prediction model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm


class RadialAggregator(nn.Module):
    """
    Circularly samples features at a fixed radius from center.
    
    Key contribution: Enforces geometric prior for retinal structure.
    Samples 360 points at 35% radius around optic disc.
    """
    def __init__(self, radius_ratio=0.35, n_points=360):
        super().__init__()
        self.radius_ratio = radius_ratio
        self.n_points = n_points

    def forward(self, x):
        """
        Args:
            x: Spatial feature map (B, C, H, W) or (B, H, W)
        Returns:
            Circular samples (B, n_points)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        B, _, H, W = x.shape
        
        # Generate circular sampling grid
        angles = torch.linspace(0, 2*np.pi, self.n_points, device=x.device)
        radius = min(H, W) * self.radius_ratio
        
        # Convert to normalized coordinates [-1, 1]
        x_coords = (W//2 + radius * torch.cos(angles)) * 2.0 / (W-1) - 1.0
        y_coords = (H//2 + radius * torch.sin(angles)) * 2.0 / (H-1) - 1.0
        
        # Create grid for sampling
        grid = torch.stack([x_coords, y_coords], -1).view(1, self.n_points, 1, 2)
        grid = grid.expand(B, -1, -1, -1)
        
        # Sample using bilinear interpolation
        samples = F.grid_sample(
            x, grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        return samples.squeeze(1).squeeze(-1)


class RNFLHead(nn.Module):
    """
    MLP head for refining radial samples to final RNFL thickness profile.
    
    Architecture: 360 → 512 → 512 → 360
    Uses LayerNorm and Dropout for regularization.
    """
    def __init__(self, input_dim=360, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


class RETFoundRNFL(nn.Module):
    """
    Complete RNFL prediction model using RETFound backbone.
    
    Architecture:
        1. RETFound encoder (ViT-L/16) → patch features
        2. Projection head → spatial depth map (56x56)
        3. Radial aggregator → circular samples (360)
        4. RNFL head → refined thickness profile (360)
    """
    def __init__(self, retfound_path=None, freeze_layers=0, dropout=0.2):
        super().__init__()
        
        # Backbone: RETFound (ViT-L/16)
        self.encoder = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
            num_classes=0  # Remove classification head
        )
        
        # Load RETFound pretrained weights
        if retfound_path and os.path.exists(retfound_path):
            checkpoint = torch.load(retfound_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)
            
            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v 
                         for k, v in state_dict.items()}
            
            self.encoder.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded RETFound weights from {retfound_path}")
        
        # Projection head: Patch features → Spatial depth map
        self.projection_head = nn.Sequential(
            nn.Linear(1024, 256),  # ViT-L output is 1024-dim
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # Single depth value per patch
        )
        
        # Radial aggregator: Circular sampling
        self.radial_aggregator = RadialAggregator(
            radius_ratio=0.35,
            n_points=360
        )
        
        # RNFL head: Refine samples to final prediction
        self.rnfl_head = RNFLHead(
            input_dim=360,
            hidden_dim=512,
            dropout=dropout
        )

        # Apply freezing after architecture is defined
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, n_blocks):
        """
        Freeze first n transformer blocks in encoder.
        """
        # Freeze patch embedding
        for param in self.encoder.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze first n blocks
        for i in range(min(n_blocks, len(self.encoder.blocks))):
            for param in self.encoder.blocks[i].parameters():
                param.requires_grad = False
        
        print(f"✓ Froze {n_blocks}/24 encoder blocks")

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input fundus images (B, 3, 224, 224)
        """
        B = x.shape[0]
        
        # 1. Extract features using timm's internal method
        features = self.encoder.forward_features(x)
        
        # 2. Extract patch tokens (exclude CLS token if it exists)
        if self.encoder.has_class_token:
             patches = features[:, 1:, :]
        else:
             patches = features
             
        # Apply layer norm if necessary
        if hasattr(self.encoder, 'fc_norm'):
             patches = self.encoder.fc_norm(patches)
             
        # 3. Project to depth values
        depth_values = self.projection_head(patches)  # (B, 196, 1)
        
        # 4. Reshape to 2D spatial map (Fixed 14x14 as proposed in paper)
        depth_map = depth_values.view(B, 1, 14, 14)
        
        # 5. Upsample to 56x56 for smoother sampling
        depth_map = F.interpolate(
            depth_map, 
            size=(56, 56),
            mode='bilinear',
            align_corners=False
        )
        
        # 6. Circular sampling at fixed radius
        radial_samples = self.radial_aggregator(depth_map)  # (B, 360)
        
        # 7. Refine to final RNFL profile
        rnfl_prediction = self.rnfl_head(radial_samples)  # (B, 360)
        
        return rnfl_prediction

    def get_trainable_parameters(self):
        """
        Get trainable parameter groups for differential learning rates.
        """
        encoder_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
        
        return encoder_params, head_params


def build_model(config):
    """
    Factory function to build model from config.
    """
    model = RETFoundRNFL(
        retfound_path=config.get('retfound_path'),
        freeze_layers=config.get('freeze_layers', 0),
        dropout=config.get('dropout', 0.2)
    )
    return model


if __name__ == "__main__":
    # Test model
    model = RETFoundRNFL(freeze_layers=0, dropout=0.2)
    
    # Test forward pass with 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Model test passed!")
    
    # Check parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
