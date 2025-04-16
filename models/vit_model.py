import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

def create_vit_model(num_classes):
    config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
    config.num_labels = num_classes
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', config=config)
    return model
