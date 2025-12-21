"""
Implementation fo Decision Tree that creates data splitting based on RegEx fullfilment
"""
import numpy as np
from dataclasses import dataclass
import re


@dataclass
class Node:
    predicted_class: int
    regex: re.Pattern | None = None
    offset: int | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class RegexDecisionTree:
    def __init__(self, regexes: list[re.Pattern], seq_length: int,
                 max_depth=6, min_samples_leaf=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.regexes = regexes
        self.seq_length = seq_length

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_classes_ = len(np.unique(y))
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X: list[str]):
        return [self._predict_one(x) for x in X]

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes_)
        probs = counts / m
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = 1.0
        best_feature = None
        best_mask_left = []
        best_mask_right = []

        for feature in range(n_features):
            mask_left = X[:, feature] == 0
            mask_right = X[:, feature] == 1

            y_left = y[mask_left]
            y_right = y[mask_right]

            if len(y_left) > self.min_samples_leaf and len(y_right) > self.min_samples_leaf:
                gini = (
                    len(y_left) * self._gini(y_left) +
                    len(y_right) * self._gini(y_right)
                ) / n_samples

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_mask_left = mask_left
                    best_mask_right = mask_right

        return best_feature, best_mask_left, best_mask_right

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth=0):
        samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = int(np.argmax(samples_per_class))

        node = Node(predicted_class=predicted_class)

        if (
            len(y) >= self.min_samples_split and
            (self.max_depth is None or depth < self.max_depth) and
            self._gini(y) > 0
        ):
            feature_idx, left_mask, right_mask = self._best_split(X, y)

            if feature_idx is not None:
                regex_index = feature_idx // self.seq_length
                offset = feature_idx % self.seq_length

                node.regex = self.regexes[regex_index]
                node.offset = offset

                node.left = self._grow_tree(
                    X[left_mask], y[left_mask], depth + 1
                )
                node.right = self._grow_tree(
                    X[right_mask], y[right_mask], depth + 1
                )

        return node

    def _predict_one(self, seq: str):
        node = self.tree_
        if node is None:
            raise Exception("Train the model before running predictions")

        while node.regex is not None and node.offset is not None:
            rg = node.regex
            start = node.offset
            L = len(rg.pattern)
            match = False

            if start + L <= len(seq):
                s = seq[start : start + L]
                if rg.fullmatch(s):
                    match = True

            next_node = node.right if match else node.left
            if next_node is None:
                break

            node = next_node

        return node.predicted_class

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if not hasattr(self, "tree_"):
            return "RegexDecisionTree(not fitted)"
        return self._str_recursive(self.tree_, depth=0)

    def _str_recursive(self, node, depth):
        indent = "|   " * depth

        if node.regex is None:
            return f"{indent}|--- class: {node.predicted_class}"

        feature_desc = f"pat='{node.regex.pattern}', offset={node.offset}"

        lines = []
        lines.append(
            f"{indent}|--- {feature_desc} == False"
        )
        lines.append(
            self._str_recursive(node.left, depth + 1)
        )
        lines.append(
            f"{indent}|--- {feature_desc} == True"
        )
        lines.append(
            self._str_recursive(node.right, depth + 1)
        )

        return "\n".join(lines)
