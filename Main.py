import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import random
import numpy as np
import os

from models.vit_model import create_vit_model
from utils.dataset import BrainMRIDataset
from utils.load_data import load_image_paths_and_labels
from analysis import visualize_gradcam, visualize_tsne, plot_performance_metrics

# ---- Setup ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

standard_label_map = {
    'glioma': 0,
    'meningioma': 1,
    'pituitary': 2,
    'no_tumor': 3
}

# ---- Load Paths ----
dataset_path = "datasets/merged"  # Assume you merged BraTS + Crystal Clean here
image_paths, labels = load_image_paths_and_labels(dataset_path, standard_label_map)

# ---- Dataset ----
dataset = BrainMRIDataset(image_paths, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# ---- Model ----
model = create_vit_model(num_classes=len(standard_label_map))
model.to(device)

# ---- Optimizer & Loss ----
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ---- Training Loop ----
def train(num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(pixel_values=imgs).logits
            loss = criterion(outputs, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

# ---- Validation ----
def validate():
    model.eval()
    correct, total = 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(pixel_values=imgs).logits
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(lbls.cpu().numpy())
    acc = correct / total
    print(f"[INFO] Validation Accuracy: {acc:.4f}")
    return labels_all, preds_all

# ---- Run ----
if __name__ == "__main__":
    train(num_epochs=5)
    labels, preds = validate()
    visualize_gradcam(model, val_dataset, device, list(standard_label_map.keys()))
    visualize_tsne(model, val_loader, device, list(standard_label_map.keys()))
    plot_performance_metrics(labels, preds, list(standard_label_map.keys()))
