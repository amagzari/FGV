import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import timm
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from functools import partial
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.amp import autocast, GradScaler
from utils.custom_cvt import ConvolutionalVisionTransformer as CustomCvT
from utils.attention import generate_batch_attention_maps 
from utils.object_crops import generate_attention_coordinates 
from utils.part_crops import nms, generate_batch_crops 

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5,7"
device =  "cuda:7"
epochs = 15
best_model_path = "/u/amagzari/wattrel-amagzari/REPOS/FGC/models/model_torch_accA_15ep"

class CUBDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.loc[idx, 'filename'])
        label = self.dataframe.loc[idx, 'label']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

datapath = "/u/amagzari/wattrel-amagzari/DATA/CUB_200_2011"
root_dir = os.path.join(datapath, "images")

# Metadata
images_df = pd.read_csv(os.path.join(datapath, 'images.txt'), delimiter=' ', names=['image_id', 'filename'])
split_df = pd.read_csv(os.path.join(datapath, 'train_test_split.txt'), delimiter=' ', names=['image_id', 'is_train'])
labels_df = pd.read_csv(os.path.join(datapath, 'image_class_labels.txt'), delimiter=' ', names=['image_id', 'label'])

df = images_df.merge(split_df, on='image_id').merge(labels_df, on='image_id')
df['label'] -= 1 # s.t. labels start at 0

# Splits
trainval_df = df[df['is_train'] == 1]
test_df = df[df['is_train'] == 0]
train_df, val_df = train_test_split(trainval_df, test_size=0.2, stratify=trainval_df['label'], random_state=42)

# Transforms
size = 448
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and Dataloaders
batch_size = 20
train_dataset = CUBDataset(train_df, root_dir, transform=train_transform)
val_dataset = CUBDataset(val_df, root_dir, transform=test_transform)
test_dataset = CUBDataset(test_df, root_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Loss function
class LossFuncC(nn.Module):  # inherit from torch.nn.Module
    def __init__(self):
        super().__init__()  # <-- ✅ this is required
        self.lf = nn.CrossEntropyLoss()

    def forward(self, preds, targs):
        return (
            self.lf(preds[0], targs) +
            self.lf(preds[1], targs) +
            (self.lf(preds[2], targs) + self.lf(preds[3], targs)) / 2
        )
def accuracyA(preds, targs): return accuracy-score(preds[0], targs) # full
def accuracyB(preds, targs): return accuracy_score(preds[1], targs) # Authors say: full, object; but it seems it's only object branch
def accuracyC(preds, targs): return accuracy_score((preds[2]+preds[3])/2, targs) # full, object, crops

# Model
def interpolate_bil(x,sz): return nn.functional.interpolate(x,mode='bilinear',align_corners=True, size=(sz,sz))

def apply_attn_erasing(x, attn_maps, thresh, p=0.5): 
    "x: bs x c x h x w, attn_maps: bs x h x w"
    erasing_mask = (attn_maps>thresh).unsqueeze(1)
    ps = torch.zeros(erasing_mask.size(0)).float().bernoulli(p).to(erasing_mask.device)
    rand_erasing_mask = 1-erasing_mask*ps[...,None,None,None]
    return rand_erasing_mask*x

class CvTEncoder(nn.Module):
    "Timm ViT encoder which return encoder outputs and optionally returns attention weights with gradient checkpointing"
    def __init__(self, cvt, nblocks=12, checkpoint_nchunks=2, return_attn_wgts=True):
                
        super().__init__() 

        # initialize params
        self.patch_embed = cvt.patch_embed
        self.cls_token = cvt.cls_token
        self.pos_embed = cvt.pos_embed
        self.pos_drop = cvt.pos_drop
        
        # until any desired layers
        self.blocks = cvt.blocks[:nblocks]        
        
        # gradient checkpointing
        self.checkpoint_nchunks = checkpoint_nchunks
        
        # return attention weights from L layers
        self.return_attn_wgts = return_attn_wgts
         
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # collect attn_wgts from all layers
        if self.return_attn_wgts:
            attn_wgts = []
            for i,blk in enumerate(self.blocks):
                if i<self.checkpoint_nchunks: x,attn_wgt = checkpoint(blk, x, use_reentrant=False)
                else:                         x,attn_wgt = blk(x)
                attn_wgts.append(attn_wgt)
            return x,attn_wgts
        
        else:
            for i,blk in enumerate(self.blocks):
                if i<self.checkpoint_nchunks: x,_ = checkpoint(blk, x, use_reentrant=False)
                else:                         x,_ = blk(x)
            return x
        
    def forward(self, x):
        return self.forward_features(x)
    
class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, out_features)
        )

    def forward(self, x):
        return self.net(x)

class MultiCropViT(nn.Module):
    "Multi Scale Multi Crop ViT Model"
    def __init__(self, 
                 encoder, 
                 num_classes,
                 input_res=384, high_res=786, min_obj_area=112*112, crop_sz=224,
                 crop_object=True, crop_object_parts=True,
                 do_attn_erasing=True, p_attn_erasing=0.5, attn_erasing_thresh=0.7,
                 encoder_nblocks=12, checkpoint_nchunks=12):

        super().__init__()  

        # Save constructor arguments
        self.input_res = input_res
        self.high_res = high_res
        self.min_obj_area = min_obj_area
        self.crop_sz = crop_sz
        self.crop_object = crop_object
        self.crop_object_parts = crop_object_parts
        self.do_attn_erasing = do_attn_erasing
        self.p_attn_erasing = p_attn_erasing
        self.attn_erasing_thresh = attn_erasing_thresh
        self.encoder_nblocks = encoder_nblocks
        self.checkpoint_nchunks = checkpoint_nchunks
        
        self.image_encoder = ViTEncoder(encoder, nblocks=encoder_nblocks, checkpoint_nchunks=checkpoint_nchunks)
        self.norm = partial(nn.LayerNorm, eps=1e-6)(768)        
        self.classifier = LinearClassifier(768, num_classes)
    
        
    def forward(self, xb_high_res):

        '''# start of bypass
        self.image_encoder.return_attn_wgts = False
        xb_input_res = nn.functional.interpolate(xb_high_res, size=(self.input_res,self.input_res))
        if not self.crop_object and not self.do_attn_erasing:
            # Skip attention computation entirely
            x_full = self.image_encoder(xb_input_res)
            x_full = self.norm(x_full)[:,0]
            return self.classifier(x_full)
        # End of bypass'''
        
        # get full image attention weigths / feature
        self.image_encoder.return_attn_wgts = True
        xb_input_res = nn.functional.interpolate(xb_high_res, size=(self.input_res,self.input_res))
        _, attn_wgts = self.image_encoder(xb_input_res)
        attn_wgts = attn_wgts[-1:]  # ✅ Keep only the last layer's attention to reduce memory
        self.image_encoder.return_attn_wgts = False
        
        # get attention maps
        attn_maps = generate_batch_attention_maps(attn_wgts, None, mode=None).detach()
        attn_maps_high_res = interpolate_bil(attn_maps[None,...],self.high_res)[0]
        attn_maps_input_res = interpolate_bil(attn_maps[None,...],self.input_res)[0]
        

        
        #### ORIGINAL IMAGE ####
        # original image attention erasing and features
        if (self.training and self.do_attn_erasing):
            xb_input_res = apply_attn_erasing(xb_input_res, attn_maps_input_res, self.attn_erasing_thresh, self.p_attn_erasing)
        x_full = self.image_encoder(xb_input_res)

        
        
        #### OBJECT CROP ####        
        if self.crop_object:
            # get object bboxes
            batch_object_bboxes = np.vstack([generate_attention_coordinates(attn_map, 
                                                                            num_bboxes=1,
                                                                            min_area=self.min_obj_area,
                                                                            random_crop_sz=self.input_res)
                                                    for attn_map in attn_maps_high_res.detach().cpu().numpy()])
            # crop objects
            xb_objects, attn_maps_objects = [], []
            for i, obj_bbox in enumerate(batch_object_bboxes):
                minr, minc, maxr, maxc = obj_bbox
                xb_objects        += [interpolate_bil(xb_high_res[i][:,minr:maxr,minc:maxc][None,...],self.input_res)[0]]
                attn_maps_objects += [interpolate_bil(attn_maps_high_res[i][minr:maxr,minc:maxc][None,None,...],self.input_res)[0][0]]
            xb_objects,attn_maps_objects = torch.stack(xb_objects),torch.stack(attn_maps_objects)

            # object image attention erasing and features
            if (self.training and self.do_attn_erasing):
                xb_objects = apply_attn_erasing(xb_objects, attn_maps_objects, self.attn_erasing_thresh, self.p_attn_erasing)
            x_object = self.image_encoder(xb_objects)
                    
        

        #### OBJECT CROP PARTS ####
        if self.crop_object_parts:
            #get object crop bboxes
            small_attn_maps_objects = interpolate_bil(attn_maps_objects[None,],self.input_res//3)[0] # to speed up calculation
            batch_crop_bboxes = generate_batch_crops(small_attn_maps_objects.cpu(),
                                                     source_sz=self.input_res//3, 
                                                     targ_sz=self.input_res, 
                                                     targ_bbox_sz=self.crop_sz,
                                                     num_bboxes=2,
                                                     nms_thresh=0.1)

            # crop object parts
            xb_crops1,xb_crops2 = [],[]
            for i, crop_bboxes in enumerate(batch_crop_bboxes):
                minr, minc, maxr, maxc = crop_bboxes[0]
                xb_crops1 += [interpolate_bil(xb_objects[i][:,minr:maxr,minc:maxc][None,...],self.input_res)[0]]
                minr, minc, maxr, maxc = crop_bboxes[1]
                xb_crops2 += [interpolate_bil(xb_objects[i][:,minr:maxr,minc:maxc][None,...],self.input_res)[0]]
            xb_crops1,xb_crops2 = torch.stack(xb_crops1),torch.stack(xb_crops2)

            # crop features
            x_crops1 = self.image_encoder(xb_crops1)
            x_crops2 = self.image_encoder(xb_crops2)
        
        
        # predict
        x_full = self.norm(x_full)[:,0]
        if self.crop_object:
            x_object = self.norm(x_object)[:,0]
            if self.crop_object_parts:
                x_crops1 = self.norm(x_crops1)[:,0]
                x_crops2 = self.norm(x_crops2)[:,0]
                return self.classifier(x_full), self.classifier(x_object), self.classifier(x_crops1), self.classifier(x_crops2)
            return self.classifier(x_full), self.classifier(x_object)
        return  self.classifier(x_full)

# Model setup

# exp 8 - full image + object + crops
model_config = dict(crop_object=False, crop_object_parts=False, do_attn_erasing=False, p_attn_erasing=0.5, attn_erasing_thresh=0.7)
loss_func = nn.CrossEntropyLoss()
#metrics =[accuracyA, accuracyB, accuracyC]
metrics = accuracy_score

# modified timm vit encoder
arch = "microsoft/cvt-13"
# Load a pretrained timm encoder model with 3 input channels
_encoder = CvtForImageClassification.from_pretrained(
    arch,
    ignore_mismatched_sizes=True  # replaces classification head
)

# Create a Vision Transformer (ViT) model instance with architecture matching the encoder
encoder = CustomCvT(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    
# Remove classification head to get features only
encoder.head = nn.Identity() 

# Load pretrained weights from the original encoder into the custom ViT
encoder.load_state_dict(_encoder.state_dict(), strict=False) # strict=Flase only loads matching layers
        
high_res=size
min_obj_area=64*64
crop_sz=128

mcvit_model = MultiCropViT(
    encoder, num_classes=200, input_res=384, high_res=high_res, min_obj_area=min_obj_area, crop_sz=crop_sz,
    encoder_nblocks=12, checkpoint_nchunks=12, **model_config
).to(device)

#mcvit_model = nn.DataParallel(mcvit_model)
#mcvit_model.to(device)

optimizer = torch.optim.AdamW(mcvit_model.parameters(), lr=1e-4)
scaler = GradScaler()
best_val_acc = 0.0

for epoch in range(epochs):
    mcvit_model.train()
    total_loss = 0
    correct = 0
    total = 0

    # for i, (images, labels) in tqdm(train_loader):
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")):
        images, labels = images.to(device), labels.to(device)

        with autocast(device_type=device):  # Enable autocast (FP16 precision where safe)
            outputs = mcvit_model(images)
            loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()      # Scales loss for safe FP16 backprop
        scaler.step(optimizer)             # Unscales gradients and updates
        scaler.update()                    # Updates scale for next iteration
        
        #print(f"[Batch {i}] Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        torch.cuda.empty_cache()

        total_loss += loss.item()
        #correct += (outputs[1].argmax(dim=1) == labels).sum().item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # Validation
    mcvit_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type=device):  # FP16 inference
                outputs = mcvit_model(images)
            #correct += (outputs[1].argmax(dim=1) == labels).sum().item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(mcvit_model.state_dict(), best_model_path)
        print("✅ Best model saved.")

# Load best model
mcvit_model.load_state_dict(torch.load(best_model_path))
mcvit_model.eval()

# Evaluate on test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = mcvit_model(images)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

print(f"🎯 Final Test Accuracy: {correct / total:.4f}")