import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import os

# Define dataset path
DATASET_PATH = '../data/cub_200_2011/CUB_200_2011/images'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
train_dataset = ImageFolder(root=DATASET_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple ViT model
class SimpleViT(nn.Module):
    def __init__(self, num_classes=200):
        super(SimpleViT, self).__init__()
        self.model = models.vit_b_16(pretrained=True)
        self.model.heads = nn.Linear(self.model.heads.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Initialize model, loss, and optimizer
model = SimpleViT(num_classes=len(train_dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Number of epochs
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

print('Training complete') 