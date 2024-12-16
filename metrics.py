from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class MetricsEvaluator:
    """
    Sınıflandırma modelinin performans metriklerini hesaplar.
    """

    def __init__(self, y_true, y_pred):
        """
        MetricsEvaluator sınıfını başlatır.

        Args:
        - y_true (array-like): Gerçek etiketler.
        - y_pred (array-like): Tahmin edilen etiketler.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def get_metrics(self):
        """
        Performans metriklerini hesaplar.

        Returns:
        - metrics (dict): Confusion matrix, accuracy, precision, recall, ve F1-score metriklerini içeren sözlük.
        """
        metrics = {
            "confusion_matrix": confusion_matrix(self.y_true, self.y_pred),
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred),
            "recall": recall_score(self.y_true, self.y_pred),
            "f1_score": f1_score(self.y_true, self.y_pred)
        }
        return metrics

    def pretty_print(self):
        """
        Performans metriklerini daha okunabilir bir formatta yazdırır.

        Returns:
        - None
        """
        metrics = self.get_metrics()
        print("Confusion Matrix:\n", metrics["confusion_matrix"])
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"F1 Score: {metrics['f1_score']:.2f}")
