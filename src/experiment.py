import wandb
import os
import numpy as np
from datetime import datetime

from src.dataset.dataset_loader import DatasetLoader
from src.modeling.cross_validation import k_fold_cross_validation
from src.config import AppConfig

app_config = AppConfig()
dataset_configs = {
    "donors": app_config.DONORS,
    "acceptors": app_config.ACCEPTORS
}

experiment_values = {
    "max_depth": [2, 5, 10, 20, 30, 50],
    "min_samples_leaf": [1, 5, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 15, 30, 50, 70],
    "tokenizer_size": np.arange(30, 100, 450)
}

wandb.login(key=app_config.WANDB_API_KEY)

for dataset_name, config in dataset_configs.items():
    print(f"\n--Running experiments for {dataset_name}--")
    original_regex_path = config.regex_path

    sequences, labels, split_index = DatasetLoader.load(
        save_path=config.dataset_path,
        url=config.url
    )

    for exp_name, values_list in experiment_values.items():
        now = datetime.now().strftime("%H%M%S")
        run_name = f"{dataset_name}_{exp_name}_{now}"

        with wandb.init(
            project="Protein_cleavage_site_experiments",
            name=run_name,
            group=dataset_name,
            config={
                "dataset": dataset_name,
                "experiment": exp_name,
            }
        ) as run:
            run.define_metric("hp_value", hidden=True)
            run.define_metric("val_accuracy", step_metric="hp_value")
            run.define_metric("train_accuracy", step_metric="hp_value")
            run.define_metric("overfitting_gap", step_metric="hp_value")

            for value in values_list:
                hiperparams = {exp_name: value}

                if exp_name == "tokenizer_size":
                    hiperparams["force_regex_recreation"] = True
                    path_part, ext = os.path.splitext(config.regex_path)
                    run_regex_path = f"{path_part}_{value}{ext}"
                    config.regex_path = run_regex_path

                print(f"Testing {exp_name} = {value}")

                results = k_fold_cross_validation(
                    sequences=sequences,
                    labels=labels,
                    config=config,
                    **hiperparams
                )

                run.log({
                    "hp_value": value,
                    "val_accuracy": results["val_accuracy"],
                    "train_accuracy": results["train_accuracy"],
                    "overfitting_gap": results["train_accuracy"] - results["val_accuracy"]
                }, step=value)

                config.regex_path = original_regex_path


print("Results on https://wandb.ai/")
