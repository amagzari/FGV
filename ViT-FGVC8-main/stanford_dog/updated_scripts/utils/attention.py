from fastai.vision.all import *
'''
def generate_batch_attention_maps(attn_wgts, targ_sz=None, mode=None):
    "Generate attention flow maps with shape (targ_sz,targ_sz) from L layer attetion weights of transformer model"
    # Stack for all layers - BS x L x K x gx x gy
    #attn_wgts = [i.detach() for i in attn_wgts]
    att_mat = torch.stack(attn_wgts, dim=1)
    
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)
   
    # To account for residual connections, we add an identity matrix to the
    aug_att_mat = att_mat + torch.eye(att_mat.size(-1))[None,None,...].to(att_mat.device)
    
    # Re-normalize the weights.
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = aug_att_mat[:,0].clone()
    for n in range(1, aug_att_mat.size(1)): joint_attentions = torch.bmm(aug_att_mat[:,n], joint_attentions)

    # BS x (num_patches+1) -> BS x gx x gy
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    joint_attentions = joint_attentions[:,0,1:].view(joint_attentions.size(0),grid_size,grid_size)
    joint_attentions /= torch.amax(joint_attentions, dim=(-2,-1), keepdim=True)

    # Bilinear interpolation to target size
    if mode == 'bilinear':
        joint_attentions = F.interpolate(joint_attentions[None,...], 
                                         (targ_sz,targ_sz), 
                                         mode=mode, align_corners=True)[0]
    elif mode == 'nearest':
        joint_attentions = F.interpolate(joint_attentions[None,...], 
                                         (targ_sz,targ_sz), 
                                         mode=mode)[0]
    elif mode is None:
        joint_attentions = joint_attentions
    
    return joint_attentions
'''

def generate_batch_attention_maps(attn_wgts, targ_sz=None, mode=None):
    "Memory-efficient attention map generator"

    bs = attn_wgts[0].size(0)
    num_layers = len(attn_wgts)
    num_heads = attn_wgts[0].size(1)
    num_tokens = attn_wgts[0].size(-1)

    # Init with identity matrix for residual connection
    joint_attn = torch.eye(num_tokens, device=attn_wgts[0].device).unsqueeze(0).expand(bs, -1, -1)

    for layer_attn in attn_wgts:
        # Average across heads: BS x H x N x N -> BS x N x N
        att = layer_attn.mean(dim=1)

        # Add residual connection
        att += torch.eye(num_tokens, device=att.device).unsqueeze(0)

        # Normalize
        att = att / att.sum(dim=-1, keepdim=True)

        # Multiply into joint attention
        joint_attn = torch.bmm(att, joint_attn)

    # From [CLS + Patches] to [Patches] only
    grid_size = int(np.sqrt(num_tokens - 1))
    joint_attn = joint_attn[:, 0, 1:].view(bs, grid_size, grid_size)
    joint_attn /= joint_attn.amax(dim=(-2, -1), keepdim=True)

    # Resize
    if mode in ['bilinear', 'nearest']:
        joint_attn = F.interpolate(joint_attn[None], (targ_sz, targ_sz), mode=mode, align_corners=(mode=='bilinear'))[0]

    return joint_attn
