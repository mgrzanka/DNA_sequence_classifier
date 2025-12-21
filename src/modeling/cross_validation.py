import numpy as np
from sklearn.model_selection import KFold
from src.modeling.confusion_matrix import ConfusionMatrix
from src.modeling.train import train_classifier


def k_fold_cross_validation(
        sequences: list[str],
        labels: np.ndarray,
        config,
        k=10,
        max_depth: int = 6,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        tokenizer_size: int = 100,
        force_regex_recreation: bool = False
    ):
    total_val_cm = None
    total_train_cm = None

    sequences_np = np.array(sequences, dtype=object)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for index, (train_index, val_index) in enumerate(kf.split(sequences_np)):
        print(f"Evaluating {index+1}/{k} cross validation batch...")

        X_train = sequences_np[train_index].tolist()
        X_val = sequences_np[val_index].tolist()
        y_train, y_val = labels[train_index], labels[val_index]

        clf = train_classifier(
            config=config,
            sequences=X_train,
            labels=y_train,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            tokenizer_size=tokenizer_size,
            force_regex_recreation=force_regex_recreation
        )

        preds_val = clf.predict(X_val)
        preds_train = clf.predict(X_train)
        if total_val_cm is None or total_train_cm is None:
            total_val_cm = ConfusionMatrix(y_val, preds_val)
            total_train_cm = ConfusionMatrix(y_train, preds_train)
        else:
            total_val_cm += ConfusionMatrix(y_val, preds_val)
            total_train_cm += ConfusionMatrix(y_train, preds_train)

    val_acc = total_val_cm.accuracy() if total_val_cm else 0
    train_acc = total_train_cm.accuracy() if total_train_cm else 0

    print(f"\nTrain Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return {
        "val_accuracy": val_acc,
        "train_accuracy": train_acc,
        "val_cm": total_val_cm,
        "train_cm": total_train_cm
    }
