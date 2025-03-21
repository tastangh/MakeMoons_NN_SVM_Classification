import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from dataset import DatasetProcessor
from metrics import MetricsEvaluator
from visualize import Visualizer
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """
    Test verisi kullanarak kayıtlı ANN ve SVM modellerini değerlendiren, 
    metrik hesaplamalarını gerçekleştiren ve görselleştirme yapan sınıf.
    """
    def __init__(self):
        """
        Evaluator sınıfını başlatır. 
        Gerekli dizinleri hazırlar ve Visualizer nesnesini oluşturur.
        """
        self._initialize_directories()
        self.metrics_list = []
        self.visualizer = Visualizer(save_dir=self.plots_dir)
        print("Evaluator initialized.")

    def _initialize_directories(self):
        """
        Test metrikleri ve görselleri kaydetmek için gerekli dizinleri oluşturur.
        """
        self.save_dir = "evaluation_results"
        self.plots_dir = os.path.join(self.save_dir, "plots")
        self.metrics_path = os.path.join(self.save_dir, "test_metrics.txt")
        self.confusion_dir = os.path.join(self.plots_dir, "confusion_matrices")
        self.boundary_dir = os.path.join(self.plots_dir, "decision_boundaries")

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.confusion_dir, exist_ok=True)
        os.makedirs(self.boundary_dir, exist_ok=True)

    def prepare_test_data(self):
        """
        Test verisini hazırlar.

        DatasetProcessor sınıfını kullanarak veri setini yükler ve 
        Test verilerini `self.X_test` ve `self.y_test` olarak alır.
        """
        print("Preparing test dataset...")
        dataset_processor = DatasetProcessor()
        dataset_processor.load_dataset()
        test_splits = dataset_processor.split_dataset()
        self.X_test, self.y_test = test_splits["test"]
        print("Test data prepared.")

    def evaluate_ann(self, learning_rates, epochs_list, optimizers, layer_configurations):
        """
        Kayıtlı ANN modellerini değerlendirir.

        Args:
            learning_rates (list): Öğrenme oranlarının bir listesi.
            epochs_list (list): Epoch sayılarının bir listesi.
            optimizers (dict): Optimizasyon algoritmalarını temsil eden bir sözlük.
            layer_configurations (list): Gizli katman sayılarının bir listesi.
        """
        print("\nEvaluating ANN models...")
        model_dir = "train_results/models"

        for lr in learning_rates:
            for epochs in epochs_list:
                for optimizer_name in optimizers.keys():
                    for layers in layer_configurations:
                        model_name = f"ann_model_LR{lr}_Epoch{epochs}_Opt{optimizer_name}_Layers{layers}.keras"
                        model_path = os.path.join(model_dir, model_name)

                        if not os.path.exists(model_path):
                            print(f"Model not found: {model_name}")
                            continue

                        print(f"Evaluating ANN model: {model_name}")
                        ann_model = load_model(model_path)

                        # Test seti tahminleri
                        y_pred = (ann_model.predict(self.X_test) > 0.5).astype(int).flatten()

                        # Metrik hesaplama
                        evaluator = MetricsEvaluator(self.y_test, y_pred)
                        test_metrics = evaluator.get_metrics()

                        # Metrik ekleme
                        self.metrics_list.append([
                            "ANN", lr, epochs, optimizer_name, layers, test_metrics
                        ])

                        # Confusion Matrix çizdirme
                        self.visualizer.plot_confusion_matrix(
                            y_true=self.y_test,
                            y_pred=y_pred,
                            set_type="test",
                            model_name="ANN",
                            params={"LR": lr, "Epochs": epochs, "Layers": layers, "Optimizer": optimizer_name},
                            save_dir=self.confusion_dir
                        )


                        # Decision Boundary çizdirme
                        self.visualizer.plot_decision_boundary(
                            model=ann_model,
                            X=self.X_test,
                            y=self.y_test,
                            save_path=os.path.join(self.boundary_dir, model_name.replace('.keras', '_boundary.png')),
                            model_type="ANN",
                            learning_rate=lr,
                            epochs=epochs,
                            optimizer_name=optimizer_name,
                            hidden_layers=layers
                        )

    def evaluate_svm(self, kernel_params):
        """
        Kayıtlı SVM modellerini değerlendirir.

        Args:
            kernel_params (dict): Kernel türlerini ve hiperparametrelerini temsil eden bir sözlük.
        """
        print("\nEvaluating SVM models...")
        model_dir = "train_results/models"

        for kernel, params in kernel_params.items():
            for C in params.get("C", [1]):
                for degree in params.get("degree", [3]):
                    for gamma in params.get("gamma", ["scale"]):
                        model_name = f"svm_model_{kernel}_C{C}_Degree{degree}_Gamma{gamma}.pkl"
                        model_path = os.path.join(model_dir, model_name)

                        if not os.path.exists(model_path):
                            print(f"Model not found: {model_name}")
                            continue

                        print(f"Evaluating SVM model: {model_name}")
                        svm_model = joblib.load(model_path)

                        # Test seti tahminleri
                        y_pred = svm_model.predict(self.X_test)

                        # Metrik hesaplama
                        evaluator = MetricsEvaluator(self.y_test, y_pred)
                        test_metrics = evaluator.get_metrics()

                        # Metrik ekleme
                        self.metrics_list.append([
                            "SVM", kernel, C, degree, gamma, test_metrics
                        ])

                        # Confusion Matrix çizdirme
                        self.visualizer.plot_confusion_matrix(
                            y_true=self.y_test,
                            y_pred=y_pred,
                            set_type="test",
                            model_name="SVM",
                            params={"Kernel": kernel, "C": C, "Degree": degree, "Gamma": gamma},
                            save_dir=self.confusion_dir
                        )
                        
                        # Decision Boundary çizdirme
                        self.visualizer.plot_decision_boundary(
                            model=svm_model,
                            X=self.X_test,
                            y=self.y_test,
                            save_path=os.path.join(self.boundary_dir, model_name.replace('.pkl', '_boundary.png')),
                            model_type="SVM",
                            kernel=kernel,
                            C=C,
                            degree=degree,
                            gamma=gamma
                        )




    def save_test_metrics(self):
        """
        Test değerlendirmesinden elde edilen metrikleri bir metin dosyasına yazar.
        """
        print("\nSaving test metrics...")
        with open(self.metrics_path, "w") as file:
            header = "{:<8} {:<12} {:<8} {:<8} {:<14} {:<10} {:<10} {:<10} {:<10}\n"
            file.write(header.format(
                "Model", "Param1", "Param2", "Param3", "Param4",
                "Accuracy", "Precision", "Recall", "F1-Score"
            ))

            for entry in self.metrics_list:
                model_type, p1, p2, p3, p4, test_metrics = entry
                row = "{:<8} {:<12} {:<8} {:<8} {:<14} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}\n"
                file.write(row.format(
                    model_type, p1, p2, p3, p4,
                    test_metrics['accuracy'], test_metrics['precision'], 
                    test_metrics['recall'], test_metrics['f1_score']
                ))

        print(f"Test metrics saved to {self.metrics_path}.")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.prepare_test_data()

    # ANN Değerlendirme Ayarları
    # learning_rates = [0.01]
    learning_rates = [0.0001,0.001,0.01, 0.1]

    # epochs_list = [50]
    epochs_list = [50, 250, 500,1000]
    optimizers = {"SGD": 1, "BGD": len(evaluator.X_test), "MBGD": 32}
    layer_configurations = [1, 2, 3]

    evaluator.evaluate_ann(
        learning_rates=learning_rates,
        epochs_list=epochs_list,
        optimizers=optimizers,
        layer_configurations=layer_configurations
    )

    # SVM Değerlendirme Ayarları
    # kernel_params = {
    #     "linear": {"C": [0.1, 1]},
    #     "poly": {"C": [0.1], "degree": [2]},
    #     "rbf": {"C": [0.1], "gamma": ["scale"]}
    # }

    kernel_params = {
        "linear": {"C": [0.01, 0.1, 1, 10, 100]},  
        "poly": {
            "C": [0.01, 0.1, 1, 10], 
            "degree": [2, 3, 4], 
            "gamma": ["scale", "auto"]  
        },
        "rbf": {
            "C": [0.01, 0.1, 1, 10, 100], 
            "gamma": ["scale", "auto", 0.01, 0.1, 1]  
        }
    }

    evaluator.evaluate_svm(kernel_params=kernel_params)

    # Test metriklerini kaydetme
    evaluator.save_test_metrics()

    print("Evaluation completed.")
