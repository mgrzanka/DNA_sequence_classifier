from sklearn.model_selection import train_test_split
import pandas as pd

from src.modeling.regex_decision_tree import RegexDecisionTree
from src.dataset.dataset_loader import DatasetLoader
from src.config import AppConfig
from src.dataset.bpe_tokenizer import BPETokenizer
from src.dataset.regex_feature_extractor import RegexFeatureExtractor
from src.modeling.confusion_matrix import ConfusionMatrix



def analyze_custom_tree_structure(clf):
    def explore_node(node, current_depth):
        if node.regex is None:
            return 1, current_depth

        n_leaves_left, depth_left = explore_node(node.left, current_depth + 1)
        n_leaves_right, depth_right = explore_node(node.right, current_depth + 1)

        return n_leaves_left + n_leaves_right, max(depth_left, depth_right)

    total_leaves, real_depth = explore_node(clf.tree_, 0)

    print(f"Tree's depth: {real_depth}")
    print(f"Number of leaves: {total_leaves}")

    queue = [(clf.tree_, 0)]
    found_features = []
    while queue and len(found_features) < 5:
        node, depth = queue.pop(0)

        if node.regex is not None:
            feature_info = f"Regex '{node.regex.pattern}' in position {node.offset} (Depth: {depth})"
            found_features.append(feature_info)
            if node.left: queue.append((node.left, depth + 1))
            if node.right: queue.append((node.right, depth + 1))

    print(f"\n--- TOP {len(found_features)} MOST IMPORTANT TESTS ---")
    for i, feature in enumerate(found_features):
        print(f"{i+1}. {feature}")


if __name__ == '__main__':
    app_config = AppConfig()
    dataset_configs = {
        "donors": {
            "config": app_config.DONORS,
            "params": {
                "max_depth": 5,
                "vocab_size": 200,
                "min_samples_leaf": 5,
                "min_samples_split": 50
            }
        },
        "acceptors": {
            "config": app_config.ACCEPTORS,
            "params": {
                "max_depth": 10,
                "vocab_size": 200,
                "min_samples_leaf": 20,
                "min_samples_split": 30
            }
        }
    }

    for dataset_name, setup in dataset_configs.items():
        config = setup["config"]
        params = setup["params"]

        print(f"\n\n{'='*60}")
        print(f"-- Running analysis for {dataset_name.upper()} --")
        print(f"Params: {params}")

        sequences, labels, split_index = DatasetLoader.load(
            save_path=config.dataset_path,
            url=config.url
        )

        majority_acc = pd.Series(labels).value_counts(normalize=True).max()
        print(f"Baseline (Majority Class Accuracy): {majority_acc:.4f}")

        Xtr, Xte, ytr, yte = train_test_split(
            sequences, labels, test_size=0.2, stratify=labels, random_state=42
        )

        tokenizer = BPETokenizer(params["vocab_size"])
        extractor = RegexFeatureExtractor(tokenizer)
        regexes = extractor.load_regexes(
            save_path=config.regex_path,
            dna_labels=ytr,
            dna_values=Xtr,
            force_regex_recreation=True
        )

        Xtr_transformed = extractor.transform(Xtr)
        Xte_transformed = extractor.transform(Xte)

        clf = RegexDecisionTree(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params["min_samples_split"],
            regexes=regexes,
            seq_length=len(sequences[0])
        )
        clf.fit(Xtr_transformed, ytr)
        preds_val = clf.predict(Xte)
        preds_train = clf.predict(Xtr)
        total_val_cm = ConfusionMatrix(yte, preds_val)
        total_train_cm = ConfusionMatrix(ytr, preds_train)

        print(f"Train F1: {total_train_cm.f1_score():.5f}, Val F1: {total_val_cm.f1_score():.5f}")
        print(f"Train Acc: {total_train_cm.accuracy():.5f}, Val Acc: {total_val_cm.accuracy():.5f}")

        analyze_custom_tree_structure(clf)

        print(clf)
