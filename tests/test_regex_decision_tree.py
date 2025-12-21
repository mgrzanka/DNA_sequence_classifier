import pytest
import re
import numpy as np

from src.modeling.regex_decision_tree import RegexDecisionTree
from src.dataset.regex_feature_extractor import RegexFeatureExtractor


def test_transform_data():
    data = ["ABC", "BBB"]
    seq_length = len(data[0])

    extractor = RegexFeatureExtractor()
    extractor.regexes = [re.compile("A"), re.compile("B"), re.compile("C")] # test, artifitial regexes

    X = extractor.transform(data)

    assert isinstance(X, np.ndarray)
    assert X.shape[0] == 2
    assert X.shape[1] == seq_length * len(extractor.regexes)
    assert np.all((X == 0) | (X == 1))


def test_extract_features_sequence_shorter_than_pattern():
    data = ["AB"]
    seq_length = len(data[0])

    extractor = RegexFeatureExtractor()
    extractor.regexes = [re.compile("ABCD")]

    X = extractor.transform(data)

    assert np.all(X == 0)


def test_gini_impurity():
    tree = RegexDecisionTree([], seq_length=3)
    tree.n_classes_ = 2

    y = np.array([0, 0, 1, 1])
    assert abs(tree._gini(y) - 0.5) < 1e-10


def test_gini_single_class():
    tree = RegexDecisionTree([], seq_length=3)
    tree.n_classes_ = 2

    y = np.array([1, 1, 1])
    assert tree._gini(y) == 0.0


def test_fit_builds_leaf_when_no_regexes():
    X_str = ["AAA", "BBB"]
    seq_length = len(X_str[0])
    y = np.array([0, 1])

    extractor = RegexFeatureExtractor()
    extractor.regexes = []
    X = extractor.transform(X_str)

    tree = RegexDecisionTree(regexes=[], seq_length=seq_length)
    tree.fit(X, y)

    assert tree.tree_.left is None
    assert tree.tree_.right is None


def test_fit_builds_tree():
    regexes = [re.compile("A")]
    X_str = ["AAA", "BBB", "AAC", "BBC"]
    seq_length = len(X_str[0])
    y = np.array([1, 0, 1, 0])

    extractor = RegexFeatureExtractor()
    extractor.regexes = regexes
    X = extractor.transform(X_str)

    tree = RegexDecisionTree(regexes, seq_length=seq_length, max_depth=2, min_samples_leaf=2)
    tree.fit(X, y)

    assert tree.tree_ is not None


def test_predict_simple():
    regexes = [re.compile("A")]
    X_str = ["AAA", "BBB", "AAC", "BBC"]
    seq_length = len(X_str[0])
    y = np.array([1, 0, 1, 0])

    extractor = RegexFeatureExtractor()
    extractor.regexes = regexes
    X = extractor.transform(X_str)

    tree = RegexDecisionTree(regexes, seq_length=seq_length, max_depth=2, min_samples_leaf=2)
    tree.fit(X, y)

    preds = tree.predict(["AAA", "BBB"])

    assert len(preds) == 2
    assert all(isinstance(p, (int, np.integer)) for p in preds)


def test_predict_when_sequence_shorter_than_pattern():
    regexes = [re.compile("ABC")]
    X_str = ["ABC", "XXX"]
    seq_length = len(X_str[0])
    y = np.array([1, 0])

    extractor = RegexFeatureExtractor()
    extractor.regexes = regexes
    X = extractor.transform(X_str)

    tree = RegexDecisionTree(regexes, seq_length=seq_length)
    tree.fit(X, y)

    preds = tree.predict(["A", "B"])

    # predicted class from root
    assert len(preds) == 2
    assert all(p == tree.tree_.predicted_class for p in preds)
