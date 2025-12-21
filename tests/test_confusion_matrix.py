import pytest
from src.modeling.confusion_matrix import ConfusionMatrix


def test_confusion_matrix_counts():
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1]

    cm = ConfusionMatrix(y_true, y_pred)

    assert cm.TP == 3
    assert cm.TN == 2
    assert cm.FP == 1
    assert cm.FN == 1


def test_confusion_matrix_accuracy():
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1]

    cm = ConfusionMatrix(y_true, y_pred)

    expected_accuracy = (3 + 2) / 7  # (TP + TN) / total
    assert cm.accuracy() == pytest.approx(expected_accuracy)


def test_confusion_matrix_precision():
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1]

    cm = ConfusionMatrix(y_true, y_pred)

    expected_precision = 3 / (3 + 1)
    assert cm.precision() == pytest.approx(expected_precision)


def test_confusion_matrix_recall():
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1]
    cm = ConfusionMatrix(y_true, y_pred)

    expected_recall = 3 / (3 + 1)
    assert cm.recall() == pytest.approx(expected_recall)


def test_confusion_matrix_specificity():
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1]
    cm = ConfusionMatrix(y_true, y_pred)

    expected_specificity = 2 / (2 + 1)
    assert cm.specificity() == pytest.approx(expected_specificity)


def test_confusion_matrix_f1_score():
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1]
    cm = ConfusionMatrix(y_true, y_pred)

    precision = 3 / (3 + 1)
    recall = 3 / (3 + 1)
    expected_f1 = 2 * precision * recall / (precision + recall)
    assert cm.f1_score() == pytest.approx(expected_f1)


def test_confusion_matrix_empty():
    cm = ConfusionMatrix([], [])
    assert cm.TP == 0
    assert cm.TN == 0
    assert cm.FP == 0
    assert cm.FN == 0
    assert cm.accuracy() == 0
    assert cm.precision() == 0
    assert cm.recall() == 0
    assert cm.specificity() == 0
    assert cm.f1_score() == 0
