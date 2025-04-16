import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import seaborn as sns

def visualize_gradcam(model, dataset, device, class_names, num_samples=4):
    print("[INFO] Generating Grad-CAM visualizations...")
    model.eval()
    target_layers = [model.vit.encoder.layer[-1].output]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    fig, axes = plt.subplots(1, num_samples, figsize=(18, 5))
    for i in range(num_samples):
        img, label = dataset[i]
        input_tensor = img.unsqueeze(0).to(device)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(label)])[0]
        rgb_img = img.permute(1, 2, 0).numpy()
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        axes[i].imshow(visualization)
        axes[i].set_title(f"True: {class_names[label]}")
        axes[i].axis('off')
    plt.suptitle("Grad-CAM Visualizations on MRI Samples", fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_tsne(model, loader, device, class_names):
    print("[INFO] Extracting features for t-SNE...")
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            output = model.vit(pixel_values=imgs).last_hidden_state[:, 0, :]  # CLS token
            features.append(output.cpu().numpy())
            labels.extend(lbls)

    features = np.vstack(features)
    labels = np.array(labels)

    print("[INFO] Running t-SNE...")
    reduced = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(features)

    plt.figure(figsize=(8,6))
    for i in np.unique(labels):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=class_names[i], alpha=0.6)
    plt.legend()
    plt.title("t-SNE of ViT Feature Embeddings", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_performance_metrics(y_true, y_pred, class_names):
    print("[INFO] Plotting Precision, Recall, F1 per class...")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-score')

    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Performance Metrics per Class", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names, fmt='.2f')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.show()
