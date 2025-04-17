import torch
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Paths
crystal_path = "/Users/yourname/path/to/CrystalClean"
brats_path = "/Users/yourname/path/to/BraTS2020"

# Hyperparameters
batch_size = 4
learning_rate = 1e-3
epochs = 10
