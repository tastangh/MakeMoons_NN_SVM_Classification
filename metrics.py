from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class MetricsEvaluator:
    """
    Sınıflandırma modelinin performans metriklerini hesaplamak için bir sınıf.

    Bu sınıf, confusion matrix, doğruluk (accuracy), kesinlik (precision), geri çağırma (recall) 
    ve F1-skora dair metrikleri hesaplar. Ayrıca, hata durumunda güvenli bir şekilde çalışabilmesi 
    için metrik hesaplamalarını güvence altına alır.
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
            metric_func (function): Hesaplanacak metrik fonksiyonu.
            kwargs: Metrik fonksiyonuna iletilecek ek argümanlar.

        Returns:
            float or str: Metrik değeri veya hata durumunda 'N/A'.
        """
        try:
            return metric_func(self.y_true, self.y_pred, **kwargs)
        except ValueError:
            return "N/A"

    def get_metrics(self):
        """
        Performans metriklerini hesaplar ve bir sözlük olarak döner.

        Returns:
            dict: Performans metriklerini içeren sözlük. 
                - confusion_matrix: Confusion matrix.
                - accuracy: Doğruluk.
                - precision: Kesinlik.
                - recall: Geri çağırma.
                - f1_score: F1-Skor.
        """
        metrics = {
            "confusion_matrix": confusion_matrix(self.y_true, self.y_pred),
            "accuracy": self.safe_metric(accuracy_score),
            "precision": self.safe_metric(precision_score, zero_division=0),
            "recall": self.safe_metric(recall_score, zero_division=0),
            "f1_score": self.safe_metric(f1_score, zero_division=0)
        }
        return metrics