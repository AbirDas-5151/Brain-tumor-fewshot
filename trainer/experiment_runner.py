if __name__ == "__main__":
    import random
    from torch.utils.data import random_split

    # Set seed
    random.seed(42)
    torch.manual_seed(42)

    # Step 1: Load subset of Crystal Clean dataset
    normal_imgs = load_images_from_folder(os.path.join(crystal_path, "Normal"), 30)
    glioma_imgs = load_images_from_folder(os.path.join(crystal_path, "Tumor", "glioma_tumor"), 30)
    meningioma_imgs = load_images_from_folder(os.path.join(crystal_path, "Tumor", "meningioma_tumor"), 30)
    pituitary_imgs = load_images_from_folder(os.path.join(crystal_path, "Tumor", "pituitary_tumor"), 30)

    # Combine and create labels
    all_images = normal_imgs + glioma_imgs + meningioma_imgs + pituitary_imgs
    all_labels = ([0] * len(normal_imgs) + [1] * len(glioma_imgs) +
                  [2] * len(meningioma_imgs) + [3] * len(pituitary_imgs))

    # Dataset splitting (Few-shot: 1, 5, 10, 20 shots per class)
    from sklearn.model_selection import StratifiedShuffleSplit
    from torch.utils.data import TensorDataset

    from torchvision.models import resnet18
    import torch.nn as nn

    class SimpleClassifier(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.backbone = resnet18(pretrained=True)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        def forward(self, x):
            return self.backbone(x)

    for k in [1, 5, 10, 20]:
        print(f"\n📊 Running {k}-shot learning...")
        shot_images, shot_labels = [], []

        for label in range(4):
            cls_imgs = [img for img, lbl in zip(all_images, all_labels) if lbl == label]
            cls_imgs = cls_imgs[:k]
            shot_images.extend(cls_imgs)
            shot_labels.extend([label] * len(cls_imgs))

        # Convert to TensorDataset
        inputs = torch.stack(shot_images)
        targets = torch.tensor(shot_labels)
        dataset = TensorDataset(inputs, targets)

        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4)

        model = SimpleClassifier(num_classes=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        trained_model = train_few_shot_model(model, optimizer, criterion, train_loader, val_loader, device)
