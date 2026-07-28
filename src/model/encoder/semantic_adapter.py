"""
Module A: Semantic Feature Adapter

Adapts frozen DINOv2 features for refinement quality estimation via a 
lightweight bottleneck adapter. Learns to emphasize scene regions where 
Gaussian Splatting refinement quality matters most (reflective, transparent, 
or texturally complex surfaces).

Design: 
  - Input: DINOv2 mono features [B*V, 384, H/16, W/16] (already extracted, frozen)
  - Bottleneck adapter: 384 -> 64 -> 384 (LoRA-style low-rank adaptation)
  - Output: Adapted semantic features for downstream refinement gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAdapter(nn.Module):
    """DINOv2 feature adapter with bottleneck architecture.
    
    Takes frozen DINOv2 features and adapts them to be task-relevant
    for Gaussian Splatting refinement quality. Uses a low-rank bottleneck
    to minimize trainable parameters (~200K) while maintaining expressiveness.
    """
    
    def __init__(self, in_channels=384, bottleneck=64, out_channels=256):
        super().__init__()
        
        self.down = nn.Conv2d(in_channels, bottleneck, 1)
        self.up = nn.Conv2d(bottleneck, out_channels, 1)
        
        # Small spatial refinement
        self.refine = nn.Sequential(
            nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck),
            nn.GELU(),
            nn.Conv2d(bottleneck, bottleneck, 1),
        )
        
        self.norm_in = nn.GroupNorm(8, in_channels)
        self.norm_out = nn.LayerNorm(out_channels)
        
        # Initialize near-zero so adapter starts as identity-like
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
    def forward(self, x):
        # x: [B*V, 384, H', W'] DINOv2 features (frozen, no grad to DINOv2)
        with torch.no_grad():
            x_norm = self.norm_in(x)
        
        # Bottleneck
        z = self.down(x_norm)         # [B*V, 64, H', W']
        z = self.refine(z) + z        # residual
        out = self.up(z)              # [B*V, 256, H', W']
        
        out = out.permute(0, 2, 3, 1)  # [B*V, H', W', 256]
        out = self.norm_out(out)
        out = out.permute(0, 3, 1, 2)  # [B*V, 256, H', W']
        
        return out
    
    @property
    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
