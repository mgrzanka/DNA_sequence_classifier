"""
Działanie algorytmu chyba będzie zależeć od zastosowanego zbioru RegEx
Zbyt długie schematy - niedouczenie. Zbyt krótkie - przeuczenie (?)
Można je też spróbować uogólnić - w poniższej implementacji nie mam dowolności w postaci "kropki"
Trzeba wiedzę dziedzinową żeby określić jakie regex są "dobre" dla nauczenia modelu

Zapobieganie przeuczeniu:
    - pozwól na mismatch (np. 1 mismatch w 5-merze) - „approximate match”.
    - Hamming distance (?)
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import export_text
from collections import defaultdict
import pandas as pd
import numpy as np
import re
import datasets
from bpe_tokenizer import BPETokenizer


def extract_features_for_seq(seq, cut_index, regexes):
    """
    Create dataset containing pattern@offset attributes from regexes to fit to decision tree
    """
    features = {}
    for i, rg in enumerate(regexes):
        L = len(rg.pattern)
        for off in range(len(seq)):
            start = cut_index + off
            s = seq[start:start+L]
            match = False
            if 0 <= start and start+L <= len(seq):
                if rg.fullmatch(s) or rg.match(s):
                    match = True
            features[f"pat{i}@{off}"] = 1 if match else 0
    return features


X, y, split_index = datasets.DonorsDataset.load()

tokenizer = BPETokenizer(100)
# Szukamy schematów często występujących w klasie pozytywnej
regexes = tokenizer.create_regexes(
    [value for value, label in zip(X, y) if int(label) == 1])
# @TODO: DO EWENTUALNEJ GENERALIZACJI REGUŁ
# sorted_regexes = sorted(regexes, key=lambda x : len(x))

# Zbuduj macierz cech dla wszystkich przykładów (później selekcja bo może być za szeroka)
# Budujemy ogromną macierz binarną:
# -> kolumny = pattern i jego offset
# -> wiersze = sekwencje
# -> wartość = czy pattern wystąpił w tej pozycji
# Tutaj sprawdzamy patterny i na pozytywnych i na negatywnych, ponieważ
# Pattern (znaleziony wcześniej w pozytywnych) pochodzi tylko z pozytywów → cecha jest „donorowa”
# Jeśli pattern pojawia się w negatywie, drzewo nauczy się, że sam pattern nie wystarczy, może wziąć pod uwagę inne patterny lub pozycje.
rows = []
labels = []
for seq, label in zip(X, y):
    rows.append(extract_features_for_seq(seq, 0, regexes))
    labels.append(label)
X = pd.DataFrame(rows)
y = np.array(labels)

# Dodatkowe: Selekcja cech - mutual info, aby pozbyć się nieistotnych regexów
# (mierzy siłę związku danej cechy z etykietą klasy).
mi = mutual_info_classif(X, y, discrete_features=True)
topk = 250
top_idx = np.argsort(mi)[-topk:]
Xsel = X
Xsel_with_mi = X.iloc[:, top_idx]

# Tree training
Xtr, Xte, ytr, yte = train_test_split(Xsel, y, test_size=0.2, stratify=y)
clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10)
clf.fit(Xtr, ytr)
print("acc train:", clf.score(Xtr, ytr))
print("acc test:", clf.score(Xte, yte))
tree_rules = export_text(clf, feature_names=list(Xsel.columns))
print("Tree structure")
print(tree_rules)

Xtr_mi, Xte_mi, ytr_mi, yte_mi = train_test_split(
    Xsel_with_mi, y, test_size=0.2, stratify=y)
clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10)
clf.fit(Xtr_mi, ytr_mi)
print("acc train mi:", clf.score(Xtr_mi, ytr_mi))
print("acc test mi:", clf.score(Xte_mi, yte_mi))
tree_rules = export_text(clf, feature_names=list(Xsel_with_mi.columns))
# print("Tree structure")
# print(tree_rules)
