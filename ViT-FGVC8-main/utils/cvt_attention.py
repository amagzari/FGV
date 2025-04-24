from fastai.vision.all import *
import torch
import torch.nn.functional as F
import numpy as np

def generate_batch_attention_maps_cvt(attention_weights, method='mean'):
    """Generate attention maps from CvT attention weights.
    
    CvT has a hierarchical structure with multiple stages of attention.
    This function combines attention weights across stages to generate
    meaningful attention maps.
    
    Args:
        attention_weights: List of attention weights from each stage
        method: Method to combine attention weights ('mean' or 'max')
        
    Returns:
        numpy.ndarray: Batch of attention maps
    """
    if not attention_weights:
        return None
    
    # Process each stage's attention weights
    stage_maps = []
    for stage_weights in attention_weights:
        # Get attention from CLS token
        cls_attention = stage_weights[:, :, 0, 1:]  # [B, H, N-1]
        
        # Average across heads if multiple
        if len(cls_attention.shape) > 3:
            cls_attention = cls_attention.mean(1)
        
        # Reshape to spatial dimensions
        h = w = int(np.sqrt(cls_attention.shape[-1]))
        attention_map = cls_attention.reshape(-1, h, w)
        
        # Normalize
        attention_map = F.softmax(attention_map.flatten(-2), dim=-1)
        attention_map = attention_map.reshape(-1, h, w)
        
        # Upsample to input resolution
        attention_map = F.interpolate(
            attention_map.unsqueeze(1),
            size=(224, 224),  # Default size, can be made configurable
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        stage_maps.append(attention_map)
    
    # Combine maps from different stages
    if method == 'mean':
        final_maps = torch.stack(stage_maps).mean(0)
    elif method == 'max':
        final_maps = torch.stack(stage_maps).max(0)[0]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to numpy and normalize
    attention_maps = final_maps.detach().cpu().numpy()
    attention_maps = (attention_maps - attention_maps.min()) / (
        attention_maps.max() - attention_maps.min() + 1e-8
    )
    
    return attention_maps

def get_stage_attention_maps(attention_weights, stage_idx):
    """Get attention maps from a specific stage.
    
    Args:
        attention_weights: List of attention weights from each stage
        stage_idx: Index of the stage to get attention from
        
    Returns:
        numpy.ndarray: Attention maps from specified stage
    """
    if not attention_weights or stage_idx >= len(attention_weights):
        return None
    
    stage_weights = attention_weights[stage_idx]
    cls_attention = stage_weights[:, :, 0, 1:]
    
    # Average across heads if multiple
    if len(cls_attention.shape) > 3:
        cls_attention = cls_attention.mean(1)
    
    # Reshape and normalize
    h = w = int(np.sqrt(cls_attention.shape[-1]))
    attention_map = cls_attention.reshape(-1, h, w)
    attention_map = F.softmax(attention_map.flatten(-2), dim=-1)
    attention_map = attention_map.reshape(-1, h, w)
    
    # Upsample
    attention_map = F.interpolate(
        attention_map.unsqueeze(1),
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)
    
    # Convert to numpy and normalize
    attention_maps = attention_map.detach().cpu().numpy()
    attention_maps = (attention_maps - attention_maps.min()) / (
        attention_maps.max() - attention_maps.min() + 1e-8
    )
    
    return attention_maps 