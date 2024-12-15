from metrics import MetricsEvaluator
from dataset import DatasetProcessor
from visualize import Visualizer
import os
import pickle
from sklearn.metrics import confusion_matrix

class Evaluator:
    def __init__(self, n_samples=400, noise=0.2):
        self.processor = DatasetProcessor(n_samples=n_samples, noise=noise)
        self.splits = None
        self.visualizer = Visualizer(save_dir="results/eval")
        self.metrics_list = []
        self.optimizer_names = []
        self.hidden_layers_list = []

    def prepare_data(self):
        self.processor.create_dataset()
        self.splits = self.processor.split_dataset()

    def evaluate_models(self):
        if not os.path.exists("models"):
            raise FileNotFoundError("Model dosyaları bulunamadı! Lütfen önce train.py dosyasını çalıştırın.")

        X_test, y_test = self.splits["test"]

        for model_file in os.listdir("models"):
            model_path = os.path.join("models", model_file)

            with open(model_path, "rb") as f:
                ann = pickle.load(f)

            y_pred = ann.predict(X_test).flatten()

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            evaluator = MetricsEvaluator(y_true=y_test, y_pred=y_pred)
            metrics = evaluator.get_metrics()

            # Model isimlerinden optimizer ve layer sayısını çıkar
            optimizer_name, hidden_layers = model_file.split("_")[:2]

            # Metrikleri sakla
            self.metrics_list.append(metrics)
            self.optimizer_names.append(optimizer_name)
            self.hidden_layers_list.append(int(hidden_layers))

            # Confusion Matrix Çizdir
            self.visualizer.plot_confusion_matrix(
                cm, class_labels=["Class 0", "Class 1"],
                title=f"Confusion Matrix: {model_file}",
                save_path=f"results/eval/{model_file}_confusion_matrix.png"
            )

        # Tüm metrikleri tek bir tabloya çevir
        self.visualizer.plot_combined_metrics_table(
            metrics_list=self.metrics_list,
            optimizer_names=self.optimizer_names,
            hidden_layers_list=self.hidden_layers_list,
            save_path="results/eval/combined_metrics_table.png"
        )

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.prepare_data()
    evaluator.evaluate_models()
