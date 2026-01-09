from sklearn.tree import DecisionTreeClassifier
import numpy as np
from src.dataset.dataset_loader import DatasetLoader
from src.config import AppConfig
from src.modeling.confusion_matrix import ConfusionMatrix
from src.dataset.regex_feature_extractor import RegexFeatureExtractor
from src.dataset.bpe_tokenizer import BPETokenizer
from src.modeling.regex_decision_tree import RegexDecisionTree
from sklearn.model_selection import KFold

RANDOM_STATE_SPLIT = 42
RANDOM_STATE_CLF = 42
MAX_DEPTH = 20


def dna_one_hot_flat(sequences):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'S': 5}

    n = len(sequences)
    l = len(sequences[0])

    one_hot_flat = np.zeros((n, l * len(mapping)), dtype=np.int8)

    for i, seq in enumerate(sequences):
        for j, nucleotide in enumerate(seq):
            one_hot_flat[i, j*len(mapping) + mapping[nucleotide]] = 1

    return one_hot_flat


app_config = AppConfig()
config = app_config.DONORS
val_cm = None
regex_val_cm = None
train_cm = None
regex_train_cm = None

sequences, labels, split_index = DatasetLoader.load(
    save_path=config.dataset_path,
    url=config.url
)

sequences_np = np.array(sequences, dtype=object)
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE_SPLIT)

for index, (train_index, val_index) in enumerate(kf.split(sequences_np)):
    X_train = sequences_np[train_index].tolist()
    X_val = sequences_np[val_index].tolist()
    y_train, y_val = labels[train_index], labels[val_index]

    # Tree without regexes
    X_train_transformed = dna_one_hot_flat(X_train)
    X_val_transformed = dna_one_hot_flat(X_val)

    clf_default = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, random_state=RANDOM_STATE_CLF)
    clf_default.fit(X_train_transformed, y_train)
    preds_train = clf_default.predict(X_train_transformed)
    preds_val = clf_default.predict(X_val_transformed)
    if val_cm is None or train_cm is None:
        val_cm = ConfusionMatrix(y_val, preds_val)
        train_cm = ConfusionMatrix(y_train, preds_train)
    else:
        val_cm += ConfusionMatrix(y_val, preds_val)
        train_cm += ConfusionMatrix(y_train, preds_train)

    # RegexDecisionTree
    tokenizer = BPETokenizer(100)
    extractor = RegexFeatureExtractor(tokenizer)
    regexes = extractor.load_regexes(
        save_path=config.regex_path,
        dna_labels=y_train,
        dna_values=X_train,
        force_regex_recreation=True
    )

    X_train_reg = extractor.transform(X_train)
    X_val_reg = extractor.transform(X_val)
    clf = RegexDecisionTree(regexes=regexes,
                            seq_length=len(sequences[0]),
                            max_depth=MAX_DEPTH)
    clf.fit(X_train_reg, y_train)
    preds_val = clf.predict(X_val)
    preds_train = clf.predict(X_train)
    if regex_val_cm is None or regex_train_cm is None:
        regex_val_cm = ConfusionMatrix(y_val, preds_val)
        regex_train_cm = ConfusionMatrix(y_train, preds_train)
    else:
        regex_val_cm = ConfusionMatrix(y_val, preds_val)
        regex_train_cm = ConfusionMatrix(y_train, preds_train)

print("Tree without regexes")
print(
    f"Train F1: {train_cm.f1_score():.5f}, Val F1: {val_cm.f1_score():.5f}")
print("Regex Tree")
print(
    f"Train F1: {regex_train_cm.f1_score():.5f}, Val F1: {regex_val_cm.f1_score():.5f}")
