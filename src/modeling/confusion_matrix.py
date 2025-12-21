
class ConfusionMatrix:
    def __init__(self, y_true, y_pred):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        for true, pred in zip(y_true, y_pred):
            if true == 1 and pred == 1:
                self.TP += 1
            elif true == 0 and pred == 0:
                self.TN += 1
            elif true == 0 and pred == 1:
                self.FP += 1
            elif true == 1 and pred == 0:
                self.FN += 1

    def accuracy(self):
        total = self.TP + self.TN + self.FP + self.FN
        return (self.TP + self.TN) / total if total > 0 else 0

    def precision(self):
        denom = self.TP + self.FP
        return self.TP / denom if denom > 0 else 0

    def recall(self):
        denom = self.TP + self.FN
        return self.TP / denom if denom > 0 else 0

    def specificity(self):
        denom = self.TN + self.FP
        return self.TN / denom if denom > 0 else 0

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def __str__(self):
        return (
            f"Confusion Matrix:\n"
            f"TN: {self.TN} | FP: {self.FP}\n"
            f"FN: {self.FN} | TP: {self.TP}"
        )

    def __add__(self, other: "ConfusionMatrix"):
        self.TP += other.TP
        self.TN += other.TN
        self.FN += other.FN
        self.FP += other.FP
        return self
