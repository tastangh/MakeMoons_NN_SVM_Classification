import os
import sys
import joblib
import numpy as np
from dataset import DatasetProcessor
from ann_model import ANNModel
from svm_model import SVMModel
from metrics import MetricsEvaluator
from visualize import Visualizer
from tensorflow.keras.callbacks import Callback

class LogEpoch(Callback):
    """
    Sadece ilk ve son epoch'taki train/val metriklerini loglamak için callback.
    """
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or epoch == self.params['epochs'] - 1:
            print(f"Epoch {epoch+1}/{self.params['epochs']} - "
                  f"loss: {logs['loss']:.4f}, accuracy: {logs['accuracy']:.4f}, "
                  f"val_loss: {logs['val_loss']:.4f}, val_accuracy: {logs['val_accuracy']:.4f}")

class Tee:
    """
    Logları hem konsola hem dosyaya yönlendiren sınıf.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            if not f.closed:
                f.write(obj)
                f.flush()

    def flush(self):
        for f in self.files:
            try:
                if not f.closed:
                    f.flush()
            except ValueError:
                pass
class Trainer:
    """
    Veri hazırlama, model eğitimi, metrik hesaplama ve görselleştirme işlemlerini yöneten Trainer sınıfı.
    """
    def __init__(self):
        self._initialize_directories()
        self._initialize_logger()
        print("Trainer initialized.")

    def _initialize_directories(self):
        """
        Model, sonuçlar ve grafiklerin kaydedileceği dizinleri oluşturur.
        """
        self.save_dir = "train_results"
        self.model_dir = os.path.join(self.save_dir, "models")
        self.plots_dir = os.path.join(self.save_dir, "plots")
        self.loss_plots_dir = os.path.join(self.plots_dir, "loss")
        self.decision_plots_dir = os.path.join(self.plots_dir, "decision_boundaries")
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.loss_plots_dir, exist_ok=True)
        os.makedirs(self.decision_plots_dir, exist_ok=True)

    def _initialize_logger(self):
        """
        Loglama işlemlerini başlatır.
        """
        log_file_path = os.path.join(self.save_dir, "training_log.txt")
        log_file = open(log_file_path, "w")
        sys.stdout = Tee(sys.stdout, log_file)

    def prepare_data(self):
        """
        Veriyi oluşturur, bölüştürür ve görselleştirir.
        """
        print("Creating and splitting the dataset...")
        dataset_processor = DatasetProcessor()
        dataset = dataset_processor.create_dataset()
        splits = dataset_processor.split_dataset()

        self.X_train, self.y_train = splits["train"]
        self.X_val, self.y_val = splits["validation"]
        self.X_test, self.y_test = splits["test"]

        print("Visualizing the dataset...")
        self.visualizer = Visualizer(save_dir=self.plots_dir)
        self.visualizer.plot_all_data(dataset)
        self.visualizer.plot_splits(splits)

        print("Data preparation complete.")

    def train_ann(self, learning_rates, epochs_list, optimizers, layer_configurations):
        """
        Yapay Sinir Ağları (ANN) için hiperparametre taraması ve eğitim.

        Parametreler:
        learning_rates -- Öğrenme oranları listesi
        epochs_list -- Epoch sayıları listesi
        optimizers -- Optimizasyon yöntemleri (örneğin, SGD, MBGD)
        layer_configurations -- Gizli katman sayıları listesi
        """
        metrics_list = []

        for lr in learning_rates:
            for epochs in epochs_list:
                for optimizer_name, batch_size in optimizers.items():
                    for hidden_layers in layer_configurations:
                        print(f"Training ANN: LR={lr}, Epochs={epochs}, Optimizer={optimizer_name}, Layers={hidden_layers}...")
                        
                        # ANN Modelini oluştur ve eğit
                        ann_builder = ANNModel(input_dim=self.X_train.shape[1], hidden_layers=hidden_layers, learning_rate=lr)
                        ann_model = ann_builder.build_model()
                        history = ann_model.fit(
                            self.X_train, self.y_train,
                            validation_data=(self.X_val, self.y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0,
                            callbacks=[LogEpoch()]  # İlk ve son epoch loglama
                        )

                        # Tahmin yap ve metrikleri değerlendir
                        y_pred = (ann_model.predict(self.X_test) > 0.5).astype(int).flatten()
                        evaluator = MetricsEvaluator(self.y_test, y_pred)
                        metrics = evaluator.get_metrics()
                        metrics_list.append(["ANN", lr, epochs, hidden_layers, optimizer_name, metrics])

                        # Kayıp grafiğini kaydet
                        self.visualizer.plot_loss(
                            history=history,
                            learning_rate=lr,
                            epochs=epochs,
                            optimizer_name=optimizer_name,
                            hidden_layers=hidden_layers,
                            save_dir=self.loss_plots_dir
                        )

                        # Karar sınırlarını kaydet
                        decision_boundary_path = os.path.join(
                            self.decision_plots_dir,
                            f"ann_decision_boundary_LR{lr}_Epoch{epochs}_Opt{optimizer_name}_Layers{hidden_layers}.png"
                        )
                        self.visualizer.plot_decision_boundary(
                            model=ann_model,
                            X=self.X_val,
                            y=self.y_val,
                            save_path=decision_boundary_path,
                            model_type="ANN",
                            learning_rate=lr,
                            epochs=epochs,
                            optimizer_name=optimizer_name,
                            hidden_layers=hidden_layers
                        )

                        # Modeli kaydet
                        model_path = os.path.join(self.model_dir, f"ann_model_LR{lr}_Epoch{epochs}_Opt{optimizer_name}_Layers{hidden_layers}.keras")
                        ann_model.save(model_path)

        print("ANN training complete.")

    def train_svm(self, kernel_params):
        """
        SVM modellerini eğitim için hiperparametre taraması yapar.

        Parametreler:
        kernel_params -- Çekirdek türleri ve parametrelerinin sözlüğü
        """
        metrics_list = []

        for kernel, params in kernel_params.items():
            for C in params.get("C", [1]):
                for degree in params.get("degree", [3]):
                    for gamma in params.get("gamma", ["scale"]):
                        print(f"Training SVM: Kernel={kernel}, C={C}, Degree={degree}, Gamma={gamma}...")
                        
                        # SVM Modelini oluştur ve eğit
                        svm_builder = SVMModel(kernel=kernel, C=C, degree=degree, gamma=gamma)
                        svm_model = svm_builder.build_model()
                        svm_model.fit(self.X_train, self.y_train)

                        # Tahmin yap ve metrikleri değerlendir
                        y_pred = svm_model.predict(self.X_test)
                        evaluator = MetricsEvaluator(self.y_test, y_pred)
                        metrics = evaluator.get_metrics()
                        metrics_list.append(["SVM", kernel, C, degree, gamma, metrics])

                        # Karar sınırlarını kaydet
                        decision_boundary_path = os.path.join(
                            self.decision_plots_dir,
                            f"svm_decision_boundary_{kernel}_C{C}_Degree{degree}_Gamma{gamma}.png"
                        )
                        self.visualizer.plot_decision_boundary(
                            model=svm_model,
                            X=self.X_val,
                            y=self.y_val,
                            save_path=decision_boundary_path,
                            model_type="SVM"
                        )

                        # Modeli kaydet
                        model_path = os.path.join(
                            self.model_dir,
                            f"svm_model_{kernel}_C{C}_Degree{degree}_Gamma{gamma}.pkl"
                        )
                        joblib.dump(svm_model, model_path)

        print("SVM training complete.")
    def save_combined_metrics(self):
            """
            Tüm eğitim metriklerini birleştirir ve dosyaya kaydeder.
            """
            print("\nSaving combined metrics...")
            with open(self.metrics_path, "w") as file:
                header = "{:<8} {:<12} {:<8} {:<8} {:<14} {:<10} {:<10} {:<10} {:<10}\n"
                file.write(header.format("Model", "Param1", "Param2", "Param3", "Param4", "Accuracy", "Precision", "Recall", "F1-Score"))

                for entry in self.metrics_list:
                    model_type, p1, p2, p3, p4, metrics = entry
                    row = "{:<8} {:<12} {:<8} {:<8} {:<14} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}\n"
                    file.write(row.format(model_type, p1, p2, p3, p4, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']))

            print(f"Combined metrics saved to {self.metrics_path}.")

    def finalize(self):
            """
            Log dosyasını düzgün şekilde kapatır.
            """
            sys.stdout = sys.__stdout__
            self.log_file.close()
            print("Log file closed. Training complete.")
if __name__ == "__main__":
    trainer = Trainer()
    trainer.prepare_data()

    # ANN Eğitim Ayarları
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    epochs_list = [50, 100, 500]
    optimizers = {"SGD": 1, "BGD": len(trainer.X_train), "MBGD": 32}
    layer_configurations = [1, 2, 3]

    trainer.train_ann(
        learning_rates=learning_rates,
        epochs_list=epochs_list,
        optimizers=optimizers,
        layer_configurations=layer_configurations
    )

    # SVM Eğitim Ayarları
    kernel_params = {
        "linear": {"C": [0.1, 1]},
        "poly": {"C": [0.1], "degree": [2]},
        "rbf": {"C": [0.1], "gamma": ["scale"]}
    }

    trainer.train_svm(kernel_params=kernel_params)
    trainer.save_combined_metrics()
    trainer.finalize()