from fastai.vision.all import *
from self_supervised.layers import *
import sklearn
from scipy.io import loadmat
import torch
from torch.utils.checkpoint import checkpoint
from sklearn.model_selection import train_test_split
from utils.custom_vit import VisionTransformer as CustomViT
from utils.attention import *
from utils.object_crops import *
from utils.part_crops import *
#from utils.multi_crop_model import *

epochs = 3
output_path = '/u/amagzari/wattrel-amagzari/REPOS/FGC/models/model_exp_val'
exp = 7

# Paths
datapath = Path("/u/amagzari/wattrel-amagzari/DATA/CUB_200_2011")
root_dir = datapath/"images"

# Metadata
images_df = pd.read_csv(datapath/'images.txt', delimiter=' ', names=['image_id', 'filename'])
split_df  = pd.read_csv(datapath/'train_test_split.txt', delimiter=' ', names=['image_id', 'is_train'])
labels_df = pd.read_csv(datapath/'image_class_labels.txt', delimiter=' ', names=['image_id', 'label'])

# Merge
df = images_df.merge(split_df, on='image_id').merge(labels_df, on='image_id')
df['label'] -= 1 # s.t. labels start at 0
df['filepath'] = df['filename'].apply(lambda fn: root_dir/fn)

# Split into train/val/test
trainval_df = df[df['is_train'] == 1].copy()
test_df     = df[df['is_train'] == 0].copy()

train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.2,
    stratify=trainval_df['label'],
    random_state=42
)

# Create single DataFrame and define splits... fastai
full_df = pd.concat([train_df, val_df], ignore_index=True)
splits = [
    list(full_df.index[full_df.index < len(train_df)]),       # train idxs
    list(full_df.index[full_df.index >= len(train_df)])       # val idxs
]

size, bs = 448, 20
stats = imagenet_stats

# Define input and label transforms
item_tfms = RandomResizedCrop(size, min_scale=0.75)
batch_tfms = aug_transforms(size=size) + [Normalize.from_stats(*stats)]

# Create Datasets and DataLoaders using DataBlock
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader('filepath'),
    get_y=ColReader('label'),
    splitter=IndexSplitter(splits[1]),  # second list is validation
    item_tfms=item_tfms,
    batch_tfms=batch_tfms
)

dls = dblock.dataloaders(full_df, bs=bs)

test_files = test_df['filepath'].tolist()
test_labels = test_df['label'].tolist()
test_dl = dls.test_dl(test_files)


# Read an image from its filename using fastai's PILImage
def read_image(filename):      return PILImage.create(filename)

# Get image size from a filename
def read_image_size(filename): return PILImage.create(filename).shape

# Get name of parent dir (label)
def read_label(filename): return filename.parent.name

###############################################################################################

# Splits the model params into three seperate blocks to apply different LRs later. m.norm is the neormalization layer after encoder
def model_splitter(m): return L(m.image_encoder, m.norm, m.classifier).map(params)

# Custom loss class for object branch only
class LossFuncA(Module): # only object
    def __init__(self):               self.lf = LabelSmoothingCrossEntropyFlat(0.1)
    def forward(self, preds, targs):  return self.lf(preds[1],targs)
    
class LossFuncB(Module): # full + object
    def __init__(self):               self.lf = LabelSmoothingCrossEntropyFlat(0.1)
    def forward(self, preds, targs):  return self.lf(preds[0],targs) + self.lf(preds[1],targs)
    
class LossFuncC(Module): # full + object + crops (2 since they use top-2)
    def __init__(self):               self.lf = LabelSmoothingCrossEntropyFlat(0.1)
    def forward(self, preds, targs):  return self.lf(preds[0],targs) + self.lf(preds[1],targs) + (self.lf(preds[2],targs)+self.lf(preds[3],targs))/2

def accuracyA(preds, targs): return accuracy(preds[0], targs) # full
def accuracyB(preds, targs): return accuracy(preds[1], targs) # Authors say: full, object; but it seems it's only object branch
def accuracyC(preds, targs): return accuracy((preds[2]+preds[3])/2, targs) # full, object, crops

def interpolate_bil(x,sz): return F.interpolate(x,mode='bilinear',align_corners=True, size=(sz,sz))

def apply_attn_erasing(x, attn_maps, thresh, p=0.5): 
    "x: bs x c x h x w, attn_maps: bs x h x w"
    erasing_mask = (attn_maps>thresh).unsqueeze(1)
    ps = torch.zeros(erasing_mask.size(0)).float().bernoulli(p).to(erasing_mask.device)
    rand_erasing_mask = 1-erasing_mask*ps[...,None,None,None]
    return rand_erasing_mask*x

class ViTEncoder(Module):
    "Timm ViT encoder which return encoder outputs and optionally returns attention weights with gradient checkpointing"
    def __init__(self, vit, nblocks=12, checkpoint_nchunks=2, return_attn_wgts=True):
                
        # initialize params
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        
        # until any desired layers
        self.blocks = vit.blocks[:nblocks]        
        
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
    
    
class MultiCropViT(Module):
    "Multi Scale Multi Crop ViT Model"
    def __init__(self, 
                 encoder, 
                 num_classes,
                 input_res=384, high_res=786, min_obj_area=112*112, crop_sz=224,
                 crop_object=True, crop_object_parts=True,
                 do_attn_erasing=True, p_attn_erasing=0.5, attn_erasing_thresh=0.7,
                 encoder_nblocks=12, checkpoint_nchunks=12):
        
        store_attr()

        self.image_encoder = ViTEncoder(encoder, nblocks=encoder_nblocks, checkpoint_nchunks=checkpoint_nchunks)
        self.norm = partial(nn.LayerNorm, eps=1e-6)(768)        
        self.classifier = create_cls_module(768, num_classes, lin_ftrs=[768], use_bn=False, first_bn=False, ps=0.)
    
        
    def forward(self, xb_high_res):

        # get full image attention weigths / feature
        self.image_encoder.return_attn_wgts = True
        xb_input_res = F.interpolate(xb_high_res, size=(self.input_res,self.input_res))
        _, attn_wgts = self.image_encoder(xb_input_res)
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
                                                    for attn_map in to_np(attn_maps_high_res)])
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

# grabage collection to free up memory
import gc

# Experiments
for i in [exp]:

    if i == 0:
        # exp 1 - full image
        model_config = dict(crop_object=False, crop_object_parts=False, do_attn_erasing=False, p_attn_erasing=0.5, attn_erasing_thresh=0.7)
        loss_func = LabelSmoothingCrossEntropyFlat(0.1)
        metrics =[accuracy] 

    if i == 1:
        # exp 2 - full image
        model_config = dict(crop_object=False, crop_object_parts=False, do_attn_erasing=True, p_attn_erasing=0.5, attn_erasing_thresh=0.7)
        loss_func = LabelSmoothingCrossEntropyFlat(0.1)
        metrics =[accuracy] 

    if i == 2:
        # exp 3 - object
        model_config = dict(crop_object=True, crop_object_parts=False, do_attn_erasing=False, p_attn_erasing=0.5, attn_erasing_thresh=0.7)
        loss_func = LossFuncA()
        metrics =[accuracyB] 

    if i == 3:
        # exp 4 - object
        model_config = dict(crop_object=True, crop_object_parts=False, do_attn_erasing=True, p_attn_erasing=0.5, attn_erasing_thresh=0.7)
        loss_func = LossFuncA()
        metrics =[accuracyB] 

    if i == 4:
        # exp 5 - full image + object
        model_config = dict(crop_object=True, crop_object_parts=False, do_attn_erasing=False, p_attn_erasing=0.5, attn_erasing_thresh=0.7)
        loss_func = LossFuncB()
        metrics =[accuracyA, accuracyB] 

    if i == 5:
        # exp 6 - full image + object
        model_config = dict(crop_object=True, crop_object_parts=False, do_attn_erasing=True, p_attn_erasing=0.5, attn_erasing_thresh=0.7)
        loss_func = LossFuncB()
        metrics =[accuracyA, accuracyB]

    if i == 6:
        # exp 7 - full image + object + crops
        model_config = dict(crop_object=True, crop_object_parts=True, do_attn_erasing=False, p_attn_erasing=0.5, attn_erasing_thresh=0.7)
        loss_func = LossFuncC()
        metrics =[accuracyA, accuracyB, accuracyC]

    if i == 7:
        # exp 8 - full image + object + crops
        model_config = dict(crop_object=True, crop_object_parts=True, do_attn_erasing=True, p_attn_erasing=0.5, attn_erasing_thresh=0.7)
        loss_func = LossFuncC()
        metrics =[accuracyA, accuracyB, accuracyC]

    # modified timm vit encoder
    arch = "vit_base_patch16_384"
    # Load a pretrained timm encoder model with 3 input channels
    _encoder = create_encoder(arch, pretrained=True, n_in=3)

    # Create a Vision Transformer (ViT) model instance with architecture matching the encoder
    encoder = CustomViT(img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12)
    
    # Remove classification head to get features only
    encoder.head = Identity() 

    # Load pretrained weights from the original encoder into the custom ViT
    encoder.load_state_dict(_encoder.state_dict())
    
    
    high_res=size
    min_obj_area=64*64
    crop_sz=128
    
    # print(f"nClasses: {dls.c}") # 200
    mcvit_model = MultiCropViT(encoder, num_classes=dls.c, input_res=384, high_res=high_res, min_obj_area=min_obj_area, crop_sz=crop_sz,
                                 encoder_nblocks=12, checkpoint_nchunks=12, **model_config)

    # Initialize the Learner with the dataloaders, model, loss function, metrics, and optimizer
    gc.collect()
    torch.cuda.empty_cache()
    cbs=[
        SaveModelCallback(monitor='accuracyA', fname='best_model'),
        ]
    learn = Learner(dls, mcvit_model, opt_func=ranger, cbs=cbs, metrics=metrics, loss_func=loss_func, splitter=model_splitter)
    # Use mixed precision training (faster & more memory efficient)
    learn.to_fp16() 
                
    lr = 3e-3
    
    # Freezes the veything but the last layer
    learn.freeze_to(-1)
    learn.fit_one_cycle(epochs, lr_max=(lr), pct_start=0.5)
    learn.save(output_path)

    # Inference on test
    preds, _ = learn.get_preds(dl=test_dl)
    true_labels = tensor(test_labels).to(preds[0].device)
    test_acc = accuracy(preds, true_labels)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    