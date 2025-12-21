from src.modeling.cross_validation import k_fold_cross_validation
from src.dataset.dataset_loader import DatasetLoader
from src.config import AppConfig


app_config = AppConfig()
dataset_configs = {
    "acceptors": app_config.ACCEPTORS,
    "donors": app_config.DONORS
}

for dataset_name, config in dataset_configs.items():
    print(f"\n--Running experiments for {dataset_name}--")
    sequences, labels, split_index = DatasetLoader.load(
        save_path=config.dataset_path,
        url=config.url
    )
    results = k_fold_cross_validation(
        sequences=sequences,
        labels=labels,
        config=config,
        k=5
    )
    print("\nTrain Dataset Confusion Matrix")
    print(results["train_cm"])
    print("Validation Dataset Confusion Matrix")
    print(results["val_cm"])
