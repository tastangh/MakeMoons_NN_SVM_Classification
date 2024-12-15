from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class MetricsEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def get_metrics(self):
        metrics = {
            "confusion_matrix": confusion_matrix(self.y_true, self.y_pred),
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred),
            "recall": recall_score(self.y_true, self.y_pred),
            "f1_score": f1_score(self.y_true, self.y_pred)
        }
        return metrics
