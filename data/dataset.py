import os
import random
import numpy as np
import nibabel as nib
import cv2
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import brats_path, crystal_path, IMAGE_SIZE, NUM_SAMPLES_BRA_TS_TRAIN, NUM_SAMPLES_BRA_TS_VAL, NUM_SAMPLES_CRYSTAL_NORMAL, NUM_SAMPLES_PER_TUMOR_CLASS, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT




def load_brats_samples(brats_dir, subset='Training', max_samples=30):
    image_paths = sorted(glob(os.path.join(brats_dir, f"{subset} data", "*")))[:max_samples]
    samples = []

    for sample_dir in image_paths:
        try:
            flair = nib.load(os.path.join(sample_dir, f"{os.path.basename(sample_dir)}_flair.nii")).get_fdata()
            t1 = nib.load(os.path.join(sample_dir, f"{os.path.basename(sample_dir)}_t1.nii")).get_fdata()
            t1ce = nib.load(os.path.join(sample_dir, f"{os.path.basename(sample_dir)}_t1ce.nii")).get_fdata()
            t2 = nib.load(os.path.join(sample_dir, f"{os.path.basename(sample_dir)}_t2.nii")).get_fdata()
            
            combined = np.stack([t1, t2, t1ce, flair], axis=0)  # shape: (4, H, W, D)
            combined = np.moveaxis(combined, -1, 0)  # shape: (D, 4, H, W)

            for slice_idx in range(combined.shape[0]):
                img_slice = combined[slice_idx]
                if img_slice.max() > 0:
                    samples.append(img_slice)
        except Exception as e:
            print(f"Failed to load {sample_dir}: {e}")
    return samples


def load_crystal_samples(crystal_dir, max_per_class=30):
    normal_imgs = glob(os.path.join(crystal_dir, 'Normal', '*.jpg'))[:max_per_class]
    
    tumor_classes = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
    tumor_imgs = []

    for label, cls in enumerate(tumor_classes):
        paths = glob(os.path.join(crystal_dir, 'Tumor', cls, '*.jpg'))[:max_per_class]
        tumor_imgs.extend([(p, label + 1) for p in paths])  # label 1-3

    labeled_samples = [(p, 0) for p in normal_imgs] + tumor_imgs
    random.shuffle(labeled_samples)
    return labeled_samples

class BraTSDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        img = np.clip(img, 0, np.percentile(img, 99))
        img = img / img.max()
        img = torch.tensor(img, dtype=torch.float32)
        img = transforms.Resize(IMAGE_SIZE)(img)
        return img  # no label for now

class CrystalDataset(Dataset):
    def __init__(self, labeled_samples, transform=None):
        self.samples = labeled_samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

crystal_samples = load_crystal_samples(crystal_path, NUM_SAMPLES_PER_TUMOR_CLASS)
crystal_dataset = CrystalDataset(crystal_samples, transform=transform)

# Split Crystal Clean Dataset
train_size = int(TRAIN_SPLIT * len(crystal_dataset))
val_size = int(VAL_SPLIT * len(crystal_dataset))
test_size = len(crystal_dataset) - train_size - val_size
crystal_train, crystal_val, crystal_test = random_split(crystal_dataset, [train_size, val_size, test_size])

crystal_loaders = {
    'train': DataLoader(crystal_train, batch_size=8, shuffle=True),
    'val': DataLoader(crystal_val, batch_size=8, shuffle=False),
    'test': DataLoader(crystal_test, batch_size=8, shuffle=False),
}

brats_train_samples = load_brats_samples(brats_path, 'Training', NUM_SAMPLES_BRA_TS_TRAIN)
brats_val_samples = load_brats_samples(brats_path, 'Validation', NUM_SAMPLES_BRA_TS_VAL)

brats_dataset = BraTSDataset(brats_train_samples + brats_val_samples)
brats_loader = DataLoader(brats_dataset, batch_size=8, shuffle=True)
