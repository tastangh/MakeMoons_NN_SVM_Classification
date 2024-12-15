from metrics import MetricsEvaluator
from dataset import DatasetProcessor
import os
import pickle

class Evaluator:
    """
    Eğitimli modellerin değerlendirilmesini yapan sınıf.
    """

    def __init__(self, n_samples=400, noise=0.2):
        """
        Evaluator sınıfını başlatır.

        Args:
        - n_samples: Veri setindeki örnek sayısı
        - noise: Veri setine eklenen gürültü seviyesi
        """
        self.processor = DatasetProcessor(n_samples=n_samples, noise=noise)
        self.splits = None

    def prepare_data(self):
        """
        Veri setini oluşturur ve böler.
        """
        self.processor.create_dataset()
        self.splits = self.processor.split_dataset()

    def evaluate_models(self):
        """
        Kaydedilmiş modelleri yükler ve test setinde değerlendirir.
        """
        if not os.path.exists("models"):
            raise FileNotFoundError("Model dosyaları bulunamadı! Lütfen önce train.py dosyasını çalıştırın.")

        X_test, y_test = self.splits["test"]

        for model_file in os.listdir("models"):
            model_path = os.path.join("models", model_file)

            # Modeli yükle
            with open(model_path, "rb") as f:
                ann = pickle.load(f)

            # Tahminler
            y_pred = ann.predict(X_test).flatten()

            # Performans metriklerini hesapla
            evaluator = MetricsEvaluator(y_true=y_test, y_pred=y_pred)
            metrics = evaluator.get_metrics()

            # Sonuçları yazdır
            print(f"Model Değerlendirme: {model_file}")
            print("Confusion Matrix:")
            print(metrics["confusion_matrix"])
            print(f"Accuracy: {metrics['accuracy']:.2f}")
            print(f"Precision: {metrics['precision']:.2f}")
            print(f"Recall: {metrics['recall']:.2f}")
            print(f"F1-Score: {metrics['f1_score']:.2f}")
            print("-" * 50)

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.prepare_data()
    evaluator.evaluate_models()
