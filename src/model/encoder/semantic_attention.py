"""
Module B: Semantic-Guided Refinement Attention

Produces per-gaussian attention weights from semantic features to 
control refinement intensity. Gaussians in semantically complex 
regions (reflective surfaces, transparent objects, high-frequency 
textures) receive higher attention -> stronger refinement.

Design:
  - Input: adapted semantic features + point positions + render error
  - MLP predicts per-gaussian attention [B, N, 1]
  - Applied as multiplicative gate on refinement error signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SemanticAttentionGate(nn.Module):
    """Predicts per-gaussian attention weights from semantic context.
    
    Uses semantic features projected to gaussian space to predict
    which gaussians need more refinement. Output gate modulates
    refinement error intensity per-gaussian.
    """
    
    def __init__(self, semantic_channels=256, hidden_dim=64):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(semantic_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        
        # Initialize near-zero so gate starts inactive
        nn.init.zeros_(self.gate_net[-2].weight)
        nn.init.zeros_(self.gate_net[-2].bias)
        
        self.gate_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, semantic_features):
        """
        Args:
            semantic_features: [B, N, C_sem] per-gaussian semantic features
        Returns:
            attention_gate: [B, N, 1] gate in approx [-0.1, 0.1]
        """
        gate = self.gate_net(semantic_features)
        gate = gate * self.gate_scale
        return gate


class SemanticProjector(nn.Module):
    """Projects image-space semantic features to per-gaussian features.
    
    Interpolates semantic features to match the latent Gaussians spatial 
    resolution, then flattens and projects to the refinement dimension.
    Flexible enough to handle arbitrary image and latent sizes.
    """
    
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
        )
        
    def forward(self, semantic_feat_2d, b, v, latent_h, latent_w):
        """
        Args:
            semantic_feat_2d: [B*V, C, H', W'] semantic features at DINOv2 resolution
            b, v: batch and view count
            latent_h, latent_w: target latent grid size (image_shape // latent_downsample)
            
        Returns:
            features: [B, V * latent_h * latent_w, C_out] per-gaussian features
        """
        feat = F.interpolate(
            semantic_feat_2d, 
            size=(latent_h, latent_w), 
            mode='bilinear', 
            align_corners=True
        )
        feat = rearrange(feat, '(b v) c h w -> b (v h w) c', b=b, v=v)
        feat = self.proj(feat)
        return feat
