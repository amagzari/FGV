import torch
import torch.nn as nn
from transformers import CvtForImageClassification
from utils.cvt_attention import generate_batch_attention_maps_cvt
from utils.cvt_object_crops import generate_attention_coordinates_cvt

class CvTEncoder(nn.Module):
    """Wrapper for CvT model that handles attention weights and object crops.
    
    CvT has a hierarchical structure with multiple stages, each operating at 
    different resolutions. This wrapper manages the complexity of handling
    attention weights across these stages.
    """
    
    def __init__(self, 
                 num_classes=1000,
                 pretrained=True):
        """Initialize CvT encoder using HuggingFace pretrained model.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to load pretrained weights
        """
        super().__init__()
        
        self.model = CvtForImageClassification.from_pretrained(
            "microsoft/cvt-13",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            output_attentions=True  # Enable attention output
        )
        
        # Store attention weights during forward pass
        self.attention_weights = []
    
    def clear_attention_weights(self):
        """Clear stored attention weights."""
        self.attention_weights = []
    
    def forward(self, x, return_features=False):
        """Forward pass through CvT model.
        
        Args:
            x: Input tensor
            return_features: Whether to return features instead of logits
            
        Returns:
            tensor: Model output (features or logits)
            list: Attention weights if available
        """
        # Clear previous attention weights
        self.clear_attention_weights()
        
        # Forward pass through model
        outputs = self.model(
            x,
            output_attentions=True,  # Get attention weights
            return_dict=True
        )
        
        # Store attention weights from each stage
        self.attention_weights = outputs.attentions
        
        if return_features:
            return outputs.hidden_states[-1], self.attention_weights
        
        return outputs.logits, self.attention_weights

class MultiCropCvT(nn.Module):
    """CvT model that supports multi-crop training using attention maps.
    
    This model uses attention weights from CvT to identify object regions
    and performs additional forward passes on cropped regions.
    """
    
    def __init__(self,
                 img_size=224,
                 num_classes=1000,
                 high_res=448,
                 min_obj_area=32*32,
                 crop_sz=112,
                 **model_config):
        """Initialize multi-crop CvT model.
        
        Args:
            img_size: Base input image size
            num_classes: Number of output classes
            high_res: High resolution input size for attention
            min_obj_area: Minimum area for object detection
            crop_sz: Size of object crops
            **model_config: Additional config for CvT model
        """
        super().__init__()
        
        self.high_res = high_res
        self.min_obj_area = min_obj_area
        self.crop_sz = crop_sz
        
        # Create CvT encoder
        self.image_encoder = CvTEncoder(
            num_classes=num_classes,
            pretrained=True
        )
        
        # Classifier head
        self.classifier = nn.Linear(
            self.image_encoder.model.num_features,
            num_classes
        )
    
    def forward(self, x):
        """Forward pass with multi-crop processing.
        
        Args:
            x: Input tensor (high resolution images)
            
        Returns:
            tensor: Classification logits
            list: Attention maps if generated
        """
        # Get features and attention from high-res image
        features_global, attn_weights = self.image_encoder(x, return_features=True)
        
        if not attn_weights:  # No attention weights available
            return self.classifier(features_global)
        
        # Generate attention maps
        attention_maps = generate_batch_attention_maps_cvt(attn_weights)
        
        # Process each image in batch
        batch_features = []
        for idx in range(x.shape[0]):
            # Get coordinates for object crops
            coords = generate_attention_coordinates_cvt(
                attention_maps[idx],
                min_area=self.min_obj_area,
                random_crop_sz=self.crop_sz
            )
            
            # Extract crops and get features
            img_features = [features_global[idx].unsqueeze(0)]
            for x1, y1, x2, y2 in coords:
                crop = x[idx:idx+1, :, y1:y2, x1:x2]
                crop = nn.functional.interpolate(
                    crop,
                    size=(self.crop_sz, self.crop_sz),
                    mode='bilinear',
                    align_corners=False
                )
                crop_features, _ = self.image_encoder(crop, return_features=True)
                img_features.append(crop_features)
            
            # Combine features
            combined_features = torch.cat(img_features, dim=0)
            batch_features.append(combined_features.mean(0, keepdim=True))
        
        # Get final features and classify
        final_features = torch.cat(batch_features, dim=0)
        return self.classifier(final_features) 