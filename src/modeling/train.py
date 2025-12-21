import numpy as np

from src.dataset.regex_feature_extractor import RegexFeatureExtractor
from src.dataset.bpe_tokenizer import BPETokenizer
from src.config import Config
from src.modeling.regex_decision_tree import RegexDecisionTree


def train_classifier(config: Config,
                     sequences: list[str],
                     labels: np.ndarray,
                     max_depth: int = 6,
                     min_samples_leaf: int = 1,
                     min_samples_split: int = 2,
                     tokenizer_size: int = 100,
                     force_regex_recreation: bool = False
                     ):
    tokenizer = BPETokenizer(tokenizer_size)
    extractor = RegexFeatureExtractor(tokenizer)

    regexes = extractor.load_regexes(
        save_path=config.regex_path,
        dna_labels=labels,
        dna_values=sequences,
        force_regex_recreation=force_regex_recreation
    )
    print("Transforming data to Decision Tree format...")
    X_train = extractor.transform(sequences)

    clf = RegexDecisionTree(regexes=regexes,
                            seq_length=len(sequences[0]),
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            min_samples_split=min_samples_split)
    print("Training the tree...")
    clf.fit(X_train, labels)

    return clf
