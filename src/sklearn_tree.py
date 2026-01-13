from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
from sklearn.metrics import f1_score, accuracy_score
from src.dataset.dataset_loader import DatasetLoader
from src.config import AppConfig
from src.dataset.regex_feature_extractor import RegexFeatureExtractor
from src.dataset.bpe_tokenizer import BPETokenizer
from src.modeling.regex_decision_tree import RegexDecisionTree
from src.modeling.confusion_matrix import ConfusionMatrix

RANDOM_STATE_SPLIT = 42
RANDOM_STATE_CLF = 3132
MAX_DEPTH = 3

app_config = AppConfig()
dataset_configs = {
    "donors": app_config.DONORS,
    "acceptors": app_config.ACCEPTORS
}

for dataset_name, config in dataset_configs.items():
    print(f"--Running custom/sklearn tree comparison for {dataset_name}--")

    sequences, labels, split_index = DatasetLoader.load(
        save_path=config.dataset_path,
        url=config.url
    )

    Xtr, Xte, ytr, yte = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE_SPLIT)

    tokenizer = BPETokenizer(100)
    extractor = RegexFeatureExtractor(tokenizer)
    regexes = extractor.load_regexes(
        save_path=config.regex_path,
        dna_labels=ytr,
        dna_values=Xtr,
        force_regex_recreation=True
    )

    Xtr_transformed = extractor.transform(Xtr)
    Xte_transformed = extractor.transform(Xte)

    seq_length = len(sequences[0])
    regex_idexes = [feature_idx // seq_length for feature_idx,
                    _ in enumerate(Xtr_transformed[0])]
    offsets = [feature_idx % seq_length for feature_idx,
            _ in enumerate(Xtr_transformed[0])]
    feature_names = [f"pat='{regexes[regex_index].pattern}', offset={offset}" for regex_index,
                    offset in zip(regex_idexes, offsets)]

    clf_ref = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, random_state=RANDOM_STATE_CLF)
    clf_ref.fit(Xtr_transformed, ytr)
    preds_test_sklearn = clf_ref.predict(Xte_transformed)
    preds_train_sklearn = clf_ref.predict(Xtr_transformed)

    tree_rules = export_text(clf_ref, feature_names=feature_names)
    f1_test = f1_score(yte, preds_test_sklearn, average='binary')
    f1_train = f1_score(ytr, preds_train_sklearn, average='binary')
    acc_test = accuracy_score(yte, preds_test_sklearn)
    acc_train = accuracy_score(ytr, preds_train_sklearn)

    print(f"Train Acc: {acc_train:.5f}, Val Acc: {acc_test:.5f}")
    print(f"Train F1: {f1_train:.5f}, Val F1: {f1_test:.5f}")
    print("Tree structure - sklearn")
    print(tree_rules)

    clf = RegexDecisionTree(regexes=regexes,
                            seq_length=len(sequences[0]),
                            max_depth=MAX_DEPTH)
    clf.fit(Xtr_transformed, ytr)
    preds_val = clf.predict(Xte)
    preds_train = clf.predict(Xtr)
    total_val_cm = ConfusionMatrix(yte, preds_val)
    total_train_cm = ConfusionMatrix(ytr, preds_train)

    val_acc = total_val_cm.accuracy()
    val_f1 = total_val_cm.f1_score()
    train_acc = total_train_cm.accuracy()
    train_f1 = total_train_cm.f1_score()

    print(f"Train Acc: {train_acc:.5f}, Val Acc: {val_acc:.5f}")
    print(f"Train F1: {train_f1:.5f}, Val F1: {val_f1:.5f}")
    print("Tree structure - RegexTree")
    print(clf)
