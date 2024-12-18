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

    def safe_metric(self, metric_func, **kwargs):
        """
        Güvenli metrik hesaplama. Hata durumunda 'N/A' döner.

        Args:
        - metric_func: Hesaplanacak metrik fonksiyonu.
        - kwargs: Metrik fonksiyonuna iletilecek argümanlar.

        Returns:
        - Metrik değeri veya 'N/A'.
        """
        try:
            return metric_func(self.y_true, self.y_pred, **kwargs)
        except ValueError:
            return "N/A"

    def get_metrics(self):
        """
        Performans metriklerini hesaplar.

        Returns:
        - metrics (dict): Confusion matrix, accuracy, precision, recall, ve F1-score metriklerini içeren sözlük.
        """
        metrics = {
            "confusion_matrix": confusion_matrix(self.y_true, self.y_pred),
            "accuracy": self.safe_metric(accuracy_score),
            "precision": self.safe_metric(precision_score, zero_division=0),
            "recall": self.safe_metric(recall_score, zero_division=0),
            "f1_score": self.safe_metric(f1_score, zero_division=0)
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
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1 Score: {metrics['f1_score']}")
