import os

def load_image_paths_and_labels(root_dir, label_map):
    image_paths = []
    labels = []
    for label_name, label_id in label_map.items():
        folder = os.path.join(root_dir, label_name)
        if not os.path.isdir(folder): continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(folder, fname))
                labels.append(label_id)
    return image_paths, labels
