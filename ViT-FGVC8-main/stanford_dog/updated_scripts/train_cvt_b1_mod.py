import sys
import os
import json
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
from transformers import CvtForImageClassification
from utils.custom_cvt import ConvolutionalVisionTransformer as CustomCvT
from utils.attention_cvt4 import generate_batch_attention_maps 
from utils.object_crops import generate_attention_coordinates 
from utils.part_crops import nms, generate_batch_crops 

model_version = int(sys.argv[1])
S_KV = int(sys.argv[2]) # KV stride 1 or 2
cls_drop = float(sys.argv[3]) # Classifier dropout: (0,1)


#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5,7"
device =  "cuda:6"
epochs = 15
best_model_path = f"/u/amagzari/wattrel-amagzari/REPOS/FGC/models/cvt{model_version}_b1_skv{S_KV}_d{cls_drop}.th"
results_path = f"/u/amagzari/wattrel-amagzari/REPOS/FGC/results/cvt{model_version}_b1_skv{S_KV}_d{cls_drop}.json"

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
batch_size = 10
train_dataset = CUBDataset(train_df, root_dir, transform=train_transform)
val_dataset = CUBDataset(val_df, root_dir, transform=test_transform)
test_dataset = CUBDataset(test_df, root_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Loss function
class LossFuncC(nn.Module):  # inherit from torch.nn.Module
    def __init__(self):
        super().__init__()  
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
    def __init__(self, cvt_model, return_attn_wgts=True):          
        super().__init__() 
        self.cvt_model = cvt_model
        self.return_attn_wgts = return_attn_wgts
         
    def forward_features(self, x):
        attn_maps = []
        for i in range(self.cvt_model.num_stages):
            x, cls_tokens, attns = getattr(self.cvt_model, f'stage{i}')(x)
            attn_maps.append(attns)

        if self.cvt_model.cls_token:
            x = self.cvt_model.norm(cls_tokens) # shape: [B, 1, C]
            x =  torch.squeeze(x, dim=1) # shape: [B, C]
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        if self.return_attn_wgts:
            last_stage_attn = attn_maps[-1] # keep last stage only: [attn_block1, ..., attn_blockN]
            return x, last_stage_attn
        else:
            return x

    def forward(self, x):
        return self.forward_features(x)
    
class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features, cls_drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_drop),
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
        
        self.image_encoder = CvTEncoder(encoder)
        embed_dim = 1024
        self.norm = partial(nn.LayerNorm, eps=1e-6)(embed_dim)        
        self.classifier = LinearClassifier(embed_dim, num_classes, cls_drop)
    
        
    def forward(self, xb_high_res):

        # start of bypass
        self.image_encoder.return_attn_wgts = False
        xb_input_res = nn.functional.interpolate(xb_high_res, size=(self.input_res,self.input_res))
        if not self.crop_object and not self.do_attn_erasing:
            # Skip attention computation entirely
            x_full = self.image_encoder(xb_input_res)
            x_full = self.norm(x_full)
            return self.classifier(x_full)
        # End of bypass'''
        
        # get full image attention weigths / feature
        self.image_encoder.return_attn_wgts = True
        xb_input_res = nn.functional.interpolate(xb_high_res, size=(self.input_res,self.input_res))
        with torch.no_grad():
            _, attn_wgts = self.image_encoder(xb_input_res)
        #print(f"🔍 Number of blocks in last stage: {len(attn_wgts)}") # 10
        #print(f"🧩 Shape of attention from first block: {attn_wgts[0].shape}") # torch.Size([20, 6, 577, 145])
        self.image_encoder.return_attn_wgts = False
        
        # get attention maps
        attn_maps = generate_batch_attention_maps(attn_wgts, None, mode=None).detach()
        attn_maps_high_res = interpolate_bil(attn_maps[None,...],self.high_res)[0]
        attn_maps_input_res = interpolate_bil(attn_maps[None,...],self.input_res)[0]
        

        
        #### ORIGINAL IMAGE ####
        # original image attention erasing and features
        if (self.training and self.do_attn_erasing):
            with torch.no_grad():
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
                with torch.no_grad():
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
        x_full = self.norm(x_full)
        if self.crop_object:
            x_object = self.norm(x_object)
            if self.crop_object_parts:
                x_crops1 = self.norm(x_crops1)
                x_crops2 = self.norm(x_crops2)
                return self.classifier(x_full), self.classifier(x_object), self.classifier(x_crops1), self.classifier(x_crops2)
            return self.classifier(x_full), self.classifier(x_object)
        return  self.classifier(x_full)

# Model setup

# exp 8 - full image + object + crops
model_config = dict(crop_object=False, crop_object_parts=False, do_attn_erasing=False, p_attn_erasing=0.5, attn_erasing_thresh=0.7)
loss_func = nn.CrossEntropyLoss()
#metrics =[accuracyA, accuracyB, accuracyC]
metrics = accuracy_score

# Downloaded from model zoo
if model_version == 13:
    state_dict = torch.load("/u/amagzari/wattrel-amagzari/REPOS/CvT/MODELS/CvT-13-384x384-IN-22k.pth", map_location='cpu')
    spec = {
        'NUM_STAGES': 3,
        'PATCH_SIZE': [7, 3, 3],
        'PATCH_STRIDE': [4, 2, 2],
        'PATCH_PADDING': [2, 1, 1],
        'DIM_EMBED': [64, 192, 384],
        'DEPTH': [1, 2, 10],
        'NUM_HEADS': [1, 3, 6],
        'MLP_RATIO': [4.0, 4.0, 4.0],
        'QKV_BIAS': [True, True, True],
        'DROP_RATE': [0.0, 0.0, 0.0],
        'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
        'DROP_PATH_RATE': [0.0, 0.0, 0.1],
        'CLS_TOKEN': [False, False, True],
        'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
        'KERNEL_QKV': [3, 3, 3],
        'PADDING_KV': [1, 1, 1],
        'STRIDE_KV': [S_KV, S_KV, S_KV],
        'PADDING_Q': [1, 1, 1],
        'STRIDE_Q': [1, 1, 1]
    }
elif model_version == 21:
    state_dict = torch.load("/u/amagzari/wattrel-amagzari/REPOS/CvT/MODELS/CvT-21-384x384-IN-22k.pth", map_location='cpu')
    spec = {
        'NUM_STAGES': 3,
        'PATCH_SIZE': [7, 3, 3],
        'PATCH_STRIDE': [4, 2, 2],
        'PATCH_PADDING': [2, 1, 1],
        'DIM_EMBED': [64, 192, 384],
        'DEPTH': [1, 4, 16],
        'NUM_HEADS': [1, 3, 6],
        'MLP_RATIO': [4.0, 4.0, 4.0],
        'QKV_BIAS': [True, True, True],
        'DROP_RATE': [0.0, 0.0, 0.0],
        'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
        'DROP_PATH_RATE': [0.0, 0.0, 0.1],
        'CLS_TOKEN': [False, False, True],
        'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
        'KERNEL_QKV': [3, 3, 3],
        'PADDING_KV': [1, 1, 1],
        'STRIDE_KV': [S_KV, S_KV, S_KV],
        'PADDING_Q': [1, 1, 1],
        'STRIDE_Q': [1, 1, 1]
    }
elif model_version == 24:
    state_dict = torch.load("/u/amagzari/wattrel-amagzari/REPOS/CvT/MODELS/CvT-w24-384x384-IN-22k.pth", map_location='cpu')
    spec = {
        'NUM_STAGES': 3,
        'PATCH_SIZE': [7, 3, 3],
        'PATCH_STRIDE': [4, 2, 2],
        'PATCH_PADDING': [2, 1, 1],
        'DIM_EMBED': [192, 768, 1024],
        'DEPTH': [2, 2, 20],
        'NUM_HEADS': [3, 12, 16],
        'MLP_RATIO': [4.0, 4.0, 4.0],
        'QKV_BIAS': [True, True, True],
        'DROP_RATE': [0.0, 0.0, 0.0],
        'ATTN_DROP_RATE': [0.0, 0.0, 0.0],
        'DROP_PATH_RATE': [0.0, 0.0, 0.3],
        'CLS_TOKEN': [False, False, True],
        'QKV_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
        'KERNEL_QKV': [3, 3, 3],
        'PADDING_KV': [1, 1, 1],
        'STRIDE_KV': [S_KV, S_KV, S_KV],
        'PADDING_Q': [1, 1, 1],
        'STRIDE_Q': [1, 1, 1]
    }
else:
    print("Please enter a valid CvT version: 13, 21, 24")

encoder = CustomCvT(spec=spec)
    
# Remove classification head to get features only
encoder.head = nn.Identity() 

# Load pretrained weights from the original encoder into the custom ViT
encoder.load_state_dict(state_dict, strict=False) 

# Important to verify if the pretrained weights have been collected and there is no architecture mistmatch
'''incompatible = encoder.load_state_dict(state_dict, strict=False)
print("Missing keys:", incompatible.missing_keys)
print("Unexpected keys:", incompatible.unexpected_keys)
exit()'''

high_res=size
min_obj_area=64*64
crop_sz=128

mcvit_model = MultiCropViT(
    encoder, num_classes=200, input_res=384, high_res=high_res, min_obj_area=min_obj_area, crop_sz=crop_sz,
    **model_config
).to(device)


optimizer = torch.optim.AdamW(mcvit_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True
)
scaler = GradScaler()
best_val_acc = 0.0
results = []

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
    avg_loss = total_loss/total

    # Validation
    mcvit_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type=device):  # FP16 inference
                outputs = mcvit_model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']

    epoch_result = {
        "epoch": epoch + 1,
        "loss": avg_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "lr": current_lr
    }
    results.append(epoch_result)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(mcvit_model.state_dict(), best_model_path)
        print("Best model saved.")

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

test_acc = correct / total
test_result = {
    "test_acc": test_acc
}
results.append(test_result)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Final Test Accuracy: {test_acc:.4f}")

print(f"Final Test Accuracy: {correct / total:.4f}")