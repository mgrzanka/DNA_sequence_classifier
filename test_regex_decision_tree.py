import re
import numpy as np
import pandas as pd

from regex_decision_tree import RegexDecisionTree


def test_prepare_data_returns_dataframe():
    tree = RegexDecisionTree([re.compile("A")])
    X = ["ABC", "BBB"]

    df = tree._prepare_data(X)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert df.shape[1] == 3


def test_extract_features_simple_match():
    tree = RegexDecisionTree([re.compile("A")])

    features = tree._extract_features_for_seq("ABC", 0)

    assert features["pat0@0"] == 1
    assert features["pat0@1"] == 0
    assert features["pat0@2"] == 0


def test_extract_features_sequence_shorter_than_pattern():
    tree = RegexDecisionTree([re.compile("ABCD")])

    features = tree._extract_features_for_seq("AB", 0)

    assert all(v == 0 for v in features.values())


def test_gini_impurity():
    tree = RegexDecisionTree([])
    tree.n_classes_ = 2

    y = np.array([0, 0, 1, 1])
    assert tree._gini(y) == 0.5


def test_gini_single_class():
    tree = RegexDecisionTree([])
    tree.n_classes_ = 2

    y = np.array([1, 1, 1])
    assert tree._gini(y) == 0.0


def test_fit_builds_leaf_when_no_regexes():
    X = ["AAA", "BBB"]
    y = np.array([0, 1])

    tree = RegexDecisionTree(regexes=[])
    tree.fit(X, y)

    assert tree.tree_.left is None
    assert tree.tree_.right is None


def test_fit_builds_tree():
    regexes = [re.compile("A")]
    X = ["AAA", "BBB", "AAC", "BBC"]
    y = np.array([1, 0, 1, 0])

    tree = RegexDecisionTree(regexes, max_depth=2, min_samples_leaf=2)
    tree.fit(X, y)

    assert tree.tree_ is not None


def test_predict_simple():
    regexes = [re.compile("A")]
    X = ["AAA", "BBB", "AAC", "BBC"]
    y = np.array([1, 0, 1, 0])

    tree = RegexDecisionTree(regexes, max_depth=2, min_samples_leaf=2)
    tree.fit(X, y)

    preds = tree.predict(["AAA", "BBB"])

    assert preds == [1, 0]


def test_predict_when_sequence_shorter_than_pattern():
    regexes = [re.compile("ABC")]
    X = ["ABC", "XXX"]
    y = np.array([1, 0])

    tree = RegexDecisionTree(regexes)
    tree.fit(X, y)

    preds = tree.predict(["A", "B"])

    assert preds == [tree.tree_.predicted_class] * 2
