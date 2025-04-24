import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
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
from transformers import CvtModel, CvtConfig
from utils.custom_cvt import CvTEncoder
from utils.attention import generate_batch_attention_maps 
from utils.object_crops import generate_attention_coordinates 
from utils.part_crops import nms, generate_batch_crops 

device = "cuda:7"
epochs = 15
best_model_path = "/u/amagzari/wattrel-amagzari/REPOS/FGC/models/model_cvt_accA_15ep"

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
df['label'] -= 1  # s.t. labels start at 0

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

def interpolate_bil(x,sz): 
    return nn.functional.interpolate(x, mode='bilinear', align_corners=True, size=(sz,sz))

def apply_attn_erasing(x, attn_maps, thresh, p=0.5): 
    "x: bs x c x h x w, attn_maps: bs x h x w"
    erasing_mask = (attn_maps>thresh).unsqueeze(1)
    ps = torch.zeros(erasing_mask.size(0)).float().bernoulli(p).to(erasing_mask.device)
    rand_erasing_mask = 1-erasing_mask*ps[...,None,None,None]
    return rand_erasing_mask*x

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

class MultiCropCvT(nn.Module):
    "Multi Scale Multi Crop CvT Model"
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
        
        self.image_encoder = CvTEncoder(encoder, nblocks=encoder_nblocks, checkpoint_nchunks=checkpoint_nchunks)
        self.norm = partial(nn.LayerNorm, eps=1e-6)(384)  # CvT-13 has 384 hidden dim
        self.classifier = LinearClassifier(384, num_classes)
        
    def forward(self, xb_high_res):
        # get full image attention weigths / feature
        self.image_encoder.return_attn_wgts = True
        xb_input_res = nn.functional.interpolate(xb_high_res, size=(self.input_res,self.input_res))
        x_full, attn_wgts = self.image_encoder(xb_input_res)
        attn_wgts = attn_wgts[-1:]  # Keep only the last layer's attention to reduce memory
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
        x_full = self.norm(x_full)[:,0]  # Take CLS token
        if self.crop_object:
            x_object = self.norm(x_object)[:,0]
            if self.crop_object_parts:
                x_crops1 = self.norm(x_crops1)[:,0]
                x_crops2 = self.norm(x_crops2)[:,0]
                return self.classifier(x_full), self.classifier(x_object), self.classifier(x_crops1), self.classifier(x_crops2)
            return self.classifier(x_full), self.classifier(x_object)
        return self.classifier(x_full)

# Model setup
model_config = dict(crop_object=True, crop_object_parts=True, do_attn_erasing=True, 
                   p_attn_erasing=0.5, attn_erasing_thresh=0.7)
loss_func = nn.CrossEntropyLoss()

# Load pretrained CvT model
encoder = CvtModel.from_pretrained("microsoft/cvt-13")

high_res = size
min_obj_area = 64*64
crop_sz = 128

mcvt_model = MultiCropCvT(
    encoder, num_classes=200, input_res=384, high_res=high_res, min_obj_area=min_obj_area, crop_sz=crop_sz,
    encoder_nblocks=12, checkpoint_nchunks=12, **model_config
).to(device)

optimizer = torch.optim.AdamW(mcvt_model.parameters(), lr=1e-4)
scaler = GradScaler()
best_val_acc = 0.0

for epoch in range(epochs):
    mcvt_model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")):
        images, labels = images.to(device), labels.to(device)

        with autocast(device_type=device):  # Enable autocast (FP16 precision where safe)
            outputs = mcvt_model(images)
            if isinstance(outputs, tuple):
                loss = (loss_func(outputs[0], labels) + 
                       loss_func(outputs[1], labels) + 
                       (loss_func(outputs[2], labels) + loss_func(outputs[3], labels)) / 2)
                pred = outputs[1]  # Use object branch predictions
            else:
                loss = loss_func(outputs, labels)
                pred = outputs

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.empty_cache()

        total_loss += loss.item()
        correct += (pred.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # Validation
    mcvt_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type=device):
                outputs = mcvt_model(images)
                if isinstance(outputs, tuple):
                    pred = outputs[1]  # Use object branch predictions
                else:
                    pred = outputs
            correct += (pred.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(mcvt_model.state_dict(), best_model_path)
        print("✅ Best model saved.")

# Load best model and evaluate
mcvt_model.load_state_dict(torch.load(best_model_path))
mcvt_model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = mcvt_model(images)
        if isinstance(outputs, tuple):
            pred = outputs[1]  # Use object branch predictions
        else:
            pred = outputs
        correct += (pred.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

print(f"🎯 Final Test Accuracy: {correct / total:.4f}") 