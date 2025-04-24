import torch
import torch.nn as nn
from transformers import CvtModel
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple

__all__ = ['CvTEncoder']

class CvTStage(nn.Module):
    """A single stage of CvT, including convolutional token embedding and transformer blocks"""
    def __init__(self, stage):
        super().__init__()
        self.conv_embedding = stage.conv_embedding
        self.dropout = stage.dropout
        self.layer_norm = stage.layer_norm
        self.blocks = stage.blocks
        self.downsample = stage.downsample if hasattr(stage, 'downsample') else None

    def forward(self, hidden_states: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.conv_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        attn_weights = []
        for block in self.blocks:
            if return_attn:
                hidden_states, attn = block(hidden_states, output_attentions=True)
                attn_weights.append(attn)
            else:
                hidden_states = block(hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        if return_attn:
            return hidden_states, attn_weights
        return hidden_states, None

class CvTEncoder(nn.Module):
    """CvT encoder which returns encoder outputs and optionally returns attention weights with gradient checkpointing"""
    def __init__(self, cvt_model, nblocks=12, checkpoint_nchunks=2, return_attn_wgts=True):
        super().__init__()
        
        # Initialize params from the CvT model
        self.embeddings = cvt_model.embeddings
        
        # Create stage-wise processing
        self.stages = nn.ModuleList([
            CvTStage(stage) for stage in cvt_model.encoder.stages
        ])
        
        # Take only the specified number of blocks
        self.nblocks = min(nblocks, len(self.stages))
        
        # Gradient checkpointing
        self.checkpoint_nchunks = checkpoint_nchunks
        
        # Whether to return attention weights
        self.return_attn_wgts = return_attn_wgts
        
        # Layer norm
        self.layernorm = cvt_model.layernorm
        
    def forward_features(self, x):
        # Get initial embeddings
        embedding_output = self.embeddings(x)
        
        # Initialize list for attention weights if needed
        all_attn_weights = [] if self.return_attn_wgts else None
        
        # Process through stages
        hidden_states = embedding_output
        for i, stage in enumerate(self.stages[:self.nblocks]):
            if i < self.checkpoint_nchunks:
                # Use gradient checkpointing for memory efficiency
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states, attn_weights = checkpoint(
                    create_custom_forward(stage),
                    hidden_states,
                    self.return_attn_wgts,
                    use_reentrant=False
                )
            else:
                hidden_states, attn_weights = stage(hidden_states, self.return_attn_wgts)
            
            if self.return_attn_wgts and attn_weights is not None:
                all_attn_weights.extend(attn_weights)
        
        # Apply final layer norm
        hidden_states = self.layernorm(hidden_states)
        
        if self.return_attn_wgts:
            return hidden_states, all_attn_weights
        return hidden_states
    
    def forward(self, x):
        return self.forward_features(x) 