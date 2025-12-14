from sklearn.model_selection import train_test_split
from confusion_matrix import ConfusionMatrix
from regex_decision_tree import RegexDecisionTree
from bpe_tokenizer import BPETokenizer
import datasets

X, y, _ = datasets.DonorsDataset.load()


tokenizer = BPETokenizer(100)
regexes = tokenizer.create_regexes(
    [value for value, label in zip(X, y) if int(label) == 1])


Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y)
clf = RegexDecisionTree(regexes, max_depth=6,
                        min_samples_leaf=10,
                        min_samples_split=2)
clf.fit(Xtr, ytr)
ytr_pred = clf.predict(Xtr)
yte_pred = clf.predict(Xte)
cm_tr = ConfusionMatrix(ytr, ytr_pred)
cm_te = ConfusionMatrix(yte, yte_pred)
print(clf)
print("acc train:", cm_tr.accuracy())
print("acc test:", cm_te.accuracy())
