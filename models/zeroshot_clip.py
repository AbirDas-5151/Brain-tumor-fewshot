from transformers import CLIPProcessor, CLIPModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

clip_model_name = "openai/clip-vit-base-patch32"  # You can change to BioCLIP or MedCLIP
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)


prompts = {
    "glioma": [
        "MRI scan of a glioma tumor",
        "MRI image showing glioma",
        "brain MRI with glioma tumor",
        "glioma detected in an MRI",
        "patient with glioma in MRI scan"
    ],
    "meningioma": [
        "MRI scan of a meningioma tumor",
        "MRI image showing meningioma",
        "brain MRI with meningioma",
        "MRI scan showing meningioma growth",
        "meningioma tumor in patient brain MRI"
    ],
    "pituitary": [
        "MRI scan of a pituitary tumor",
        "MRI of pituitary adenoma",
        "brain MRI with pituitary tumor",
        "pituitary tumor identified in MRI",
        "MRI showing pituitary mass"
    ],
    "normal": [
        "MRI scan of a healthy brain",
        "normal brain MRI image",
        "no tumor in MRI scan",
        "MRI scan of a patient with no brain tumor",
        "clean brain MRI"
    ]
}

from torchvision import transforms
from PIL import Image
import os

# Preprocessing as required by CLIP
clip_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

def load_images_from_folder(folder, max_imgs=10):
    images = []
    for idx, fname in enumerate(os.listdir(folder)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and idx < max_imgs:
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            img = clip_preprocess(img)
            images.append(img)
    return images

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

# Example usage with Crystal Clean dataset
test_imgs = []
for cls in ["normal", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]:
    folder = os.path.join(crystal_path, "Tumor" if cls != "normal" else "", cls)
    test_imgs += load_images_from_folder(folder, max_imgs=5)  # Load 5 images per class

results = predict_with_clip(clip_model, clip_processor, test_imgs, prompts)

for i, (pred, scores) in enumerate(results):
    print(f"Image {i+1} -> Predicted: {pred}")
    for cls, score in scores.items():
        print(f"  {cls}: {score:.4f}")
    print()
