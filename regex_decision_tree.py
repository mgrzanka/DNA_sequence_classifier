"""
Implementacja drzewa decyzyjnego z przycinaniem dla atrybutów binarnych
określających czy spełnione jest wyrażenie regularne.
Próbki niespelniające wyrażenia regularnego umieszczane są w lewym poddrzewie.
"""

import pandas as pd
import numpy as np


class Node:
    def __init__(self, predicted_class, regex=None, offset=None):
        self.regex = regex
        self.offset = offset
        self.predicted_class = predicted_class
        self.left = None
        self.right = None


class RegexDecisionTree:
    def __init__(self, regexes, max_depth=6, min_samples_leaf=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.regexes = regexes

    def fit(self, X, y):
        X = self._prepare_data(X)
        self.n_classes_ = len(np.unique(y))
        self.tree_ = self._grow_tree(X, y)

    def _prepare_data(self, X):
        rows = []
        for seq in X:
            rows.append(self._extract_features_for_seq(seq, 0))
        X = pd.DataFrame(rows)
        return X

    def _extract_features_for_seq(self, seq, cut_index):
        features = {}
        for i, rg in enumerate(self.regexes):
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

    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes_)
        probs = counts / m
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        m, n_features = X.shape
        best_gini = 1.0
        best_feature = None
        best_mask_left = []
        best_mask_right = []
        for feature in range(n_features):
            mask_left = X[:, feature] == 0
            mask_right = X[:, feature] == 1

            y_left = y[mask_left]
            y_right = y[mask_right]

            if len(y_left) < self.min_samples_leaf:
                deficit = self.min_samples_leaf - len(y_left)
                if deficit <= len(y_right):
                    indices = np.where(mask_right)[0][:deficit]
                    mask_left[indices] = True
                    mask_right[indices] = False
                    y_left = y[mask_left]
                    y_right = y[mask_right]
                else:
                    continue

            if len(y_right) < self.min_samples_leaf:
                deficit = self.min_samples_leaf - len(y_right)
                if deficit <= len(y_left):
                    indices = np.where(mask_left)[0][:deficit]
                    mask_right[indices] = True
                    mask_left[indices] = False
                    y_left = y[mask_left]
                    y_right = y[mask_right]
                else:
                    continue

            gini = (
                len(y_left) * self._gini(y_left) +
                len(y_right) * self._gini(y_right)
            ) / m

            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_mask_left = mask_left
                best_mask_right = mask_right

        return best_feature, best_mask_left, best_mask_right

    def _grow_tree(self, X, y, depth=0):
        samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(samples_per_class)

        node = Node(predicted_class=predicted_class)

        if (
            len(y) >= self.min_samples_split and
            (self.max_depth is None or depth < self.max_depth) and
            self._gini(y) > 0
        ):
            feature_idx, left_mask, right_mask = self._best_split(X.values, y)

            if feature_idx is not None:
                feature_name = X.columns[feature_idx]
                pat_part, offset_part = feature_name.split("@")
                regex_index = int(pat_part.replace("pat", ""))
                offset = int(offset_part)

                node.regex = self.regexes[regex_index]
                node.offset = offset

                node.left = self._grow_tree(
                    X[left_mask], y[left_mask], depth + 1
                )
                node.right = self._grow_tree(
                    X[right_mask], y[right_mask], depth + 1
                )

        return node

    def _predict_one(self, seq):
        node = self.tree_
        while node.left is not None:
            rg = node.regex
            start = node.offset
            L = len(rg.pattern)
            match = False

            if start + L <= len(seq):
                s = seq[start:start + L]
                if rg.fullmatch(s) or rg.match(s):
                    match = True

            if match:
                node = node.right
            else:
                node = node.left

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
