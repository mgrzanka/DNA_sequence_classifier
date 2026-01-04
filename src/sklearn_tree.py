from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
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
config = app_config.DONORS

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
print("acc train:", clf_ref.score(Xtr_transformed, ytr))
print("acc test:", clf_ref.score(Xte_transformed, yte))
tree_rules = export_text(clf_ref, feature_names=feature_names)
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
train_acc = total_train_cm.accuracy()

print(f"\nTrain Acc: {train_acc:.5f}, Val Acc: {val_acc:.5f}")
print("Tree structure - RegexTree")
print(clf)
