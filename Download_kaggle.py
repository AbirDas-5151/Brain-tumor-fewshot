import kagglehub

def download_brats():
    path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
    print("Downloaded BraTS dataset to:", path)

def download_crystal():
    path = kagglehub.dataset_download("mohammadhossein77/brain-tumors-dataset")
    print("Downloaded Crystal Clean dataset to:", path)

if __name__ == "__main__":
    download_brats()
    download_crystal()
