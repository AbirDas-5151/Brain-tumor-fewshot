from data.dataset import load_crystal_clean_data
from trainer.experiment_runner import run_few_shot_experiments
from config import device, crystal_path

if __name__ == "__main__":
    train_loaders, val_loaders = load_crystal_clean_data(crystal_path)
    run_few_shot_experiments(train_loaders, val_loaders, device)
