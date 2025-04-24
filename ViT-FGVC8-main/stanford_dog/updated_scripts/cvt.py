import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import CvtForImageClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import timm
from tqdm import tqdm

model_version = int(sys.argv[1]) # 13, 21, 24
best_model_path = f"/u/amagzari/wattrel-amagzari/REPOS/FGC/models/cvt{model_version}.th"
results_path = f"/u/amagzari/wattrel-amagzari/REPOS/FGC/results/cvt{model_version}.json"

# ----------- Settings -----------
DATA_DIR = '/u/amagzari/wattrel-amagzari/DATA/CUB_200_2011/images'
NUM_CLASSES = 200
BATCH_SIZE = 16
EPOCHS = 5
VAL_SPLIT = 0.2
DEVICE = 'cuda:7'

# ----------- Transforms -----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ----------- Dataset + Subset (first NUM_CLASSES) -----------
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Get class indices < NUM_CLASSES
#subset_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label < NUM_CLASSES]
#subset = torch.utils.data.Subset(full_dataset, subset_indices)
subset = full_dataset

# ----------- Train/Test Split -----------
total = len(subset)
val_size = int(total * VAL_SPLIT)
train_size = total - val_size
train_data, val_data = random_split(subset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# ----------- Model Setup -----------
model_name = ""

if model_version == 13:
    model_name = "microsoft/cvt-13"
elif model_version == 21:
    model_name = "microsoft/cvt-21"
elif model_version == 24:
    model_name = "microsoft/cvt-w24-384-22k"
else:
    print("Please enter a valid CvT version: 13, 21, 24")

model = CvtForImageClassification.from_pretrained(
    model_name,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True  # replaces classification head
)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# ----------- Metrics Helper -----------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    return acc, precision, recall, f1

# ----------- Training Loop -----------
results = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    y_true_train, y_pred_train = [], []

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        # CvtForImageClassification returns a ImageClassifierOutputWithNoAttention object — not a raw tensor. You need to access the logits
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(preds.cpu().numpy())

    train_acc, _, _, _ = compute_metrics(y_true_train, y_pred_train)
    avg_tr_loss = train_loss / len(train_loader)

    # ----------- Validation/Test Evaluation -----------
    model.eval()
    val_loss = 0.0
    y_true_val, y_pred_val = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Test]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(preds.cpu().numpy())

    val_acc, _, _, _ = compute_metrics(y_true_val, y_pred_val)

    epoch_result = {
        "epoch": epoch + 1,
        "loss": avg_tr_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
    }
    results.append(epoch_result)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Epoch {epoch+1}: Train Loss={avg_tr_loss:.4f}, train_acc = {train_acc}, val_acc={val_acc:.4f}")

    