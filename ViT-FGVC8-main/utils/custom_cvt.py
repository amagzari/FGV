import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CvTEncoder(nn.Module):
    """CvT encoder which returns encoder outputs and optionally returns attention weights with gradient checkpointing.
    Parallels ViTEncoder but handles CvT's hierarchical structure.
    """
    def __init__(self, cvt_model, nblocks=12, checkpoint_nchunks=2, return_attn_wgts=True):
        super().__init__()
        
        # Initialize params from the CvT model
        self.embeddings = cvt_model.embeddings
        self.encoder = cvt_model.encoder
        self.layernorm = cvt_model.layernorm
        
        # Take only specified number of blocks
        self.nblocks = min(nblocks, len(self.encoder.stages))
        
        # Gradient checkpointing
        self.checkpoint_nchunks = checkpoint_nchunks
        
        # Whether to return attention weights
        self.return_attn_wgts = return_attn_wgts
        
    def forward_features(self, x):
        # Get initial embeddings
        embedding_output = self.embeddings(x)
        
        # Initialize list for attention weights if needed
        all_attn_weights = [] if self.return_attn_wgts else None
        
        # Process through stages
        hidden_states = embedding_output
        for i, stage in enumerate(self.encoder.stages[:self.nblocks]):
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