from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import time
import copy

def train_few_shot_model(model, optimizer, criterion, train_loader, val_loader, device, epochs=10):
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        val_acc = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return accuracy_score(true_labels, preds)


import torch.nn.functional as F

def predict_with_clip(model, processor, images, prompt_dict):
    all_prompts = []
    class_labels = []

    for label, prompts in prompt_dict.items():
        all_prompts.extend(prompts)
        class_labels.extend([label] * len(prompts))

    # Encode text prompts
    text_inputs = processor(text=all_prompts, return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    text_features = F.normalize(text_features, dim=-1)

    results = []

    for img in images:
        img = img.unsqueeze(0).to(device)
        image_features = model.get_image_features(pixel_values=img)
        image_features = F.normalize(image_features, dim=-1)

        sims = (image_features @ text_features.T).squeeze(0)
        avg_sims = {}
        for i, label in enumerate(class_labels):
            avg_sims.setdefault(label, []).append(sims[i].item())

        # Average similarity over multiple prompts
        avg_sims = {cls: sum(scores) / len(scores) for cls, scores in avg_sims.items()}
        predicted_class = max(avg_sims, key=avg_sims.get)
        results.append((predicted_class, avg_sims))

    return results
