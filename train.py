import os
import sys
import joblib
import numpy as np
from dataset import DatasetProcessor
from ann_model import ANNModel
from svm_model import SVMModel
from metrics import MetricsEvaluator
from visualize import Visualizer
from logger import Logger 
from epoch_logger import EpochLogger  
from logger import Logger 

class Trainer:
    """
    Veri hazırlama, model eğitimi, metrik hesaplama ve görselleştirme işlemlerini yöneten Trainer sınıfı.
    """
    def __init__(self):
        self._initialize_directories()
        self.logger = Logger(save_dir=self.save_dir,log_filename="train.log").logger     
        self.metrics_list = []  
        self.logger.info("Trainer initialized.")

    def _initialize_directories(self):
        """
        Model, sonuçlar ve grafiklerin kaydedileceği dizinleri oluşturur.
        """
        self.save_dir = "train_results"
        self.dataset_dir="dataset"
        self.model_dir = os.path.join(self.save_dir, "models")
        self.plots_dir = os.path.join(self.save_dir, "plots")
        self.loss_plots_dir = os.path.join(self.plots_dir, "loss")
        self.decision_plots_dir = os.path.join(self.plots_dir, "val_decision_boundaries")
        self.confusion_matrix_dir = os.path.join(self.plots_dir, "train_val_confusion_matrix")
        self.metrics_path = os.path.join(self.save_dir, "combined_metrics.txt") 

        # Dizinleri oluştur
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.loss_plots_dir, exist_ok=True)
        os.makedirs(self.decision_plots_dir, exist_ok=True)
        os.makedirs(self.confusion_matrix_dir, exist_ok=True)
        

    def prepare_data(self):
        """
        Veriyi oluşturur, bölüştürür ve görselleştirir.
        """
        self.logger.info("Creating and splitting the dataset...")
        dataset_processor = DatasetProcessor()
        dataset = dataset_processor.create_dataset()
        splits = dataset_processor.split_dataset()

        self.X_train, self.y_train = splits["train"]
        self.X_val, self.y_val = splits["validation"]

        self.logger.info("Visualizing the dataset...")
        self.visualizer = Visualizer(save_dir=self.dataset_dir)
        self.visualizer.plot_all_data(dataset)
        self.visualizer.plot_splits(splits)

        self.logger.info("Data preparation complete.")

    def train_ann(self, learning_rates, epochs_list, optimizers, layer_configurations):
        """
        Yapay Sinir Ağları (ANN) için hiperparametre taraması ve eğitim.

        Parametreler:
        learning_rates -- Öğrenme oranları listesi
        epochs_list -- Epoch sayıları listesi
        optimizers -- Optimizasyon yöntemleri (örneğin, SGD, MBGD)
        layer_configurations -- Gizli katman sayıları listesi
        """

        for lr in learning_rates:
            for epochs in epochs_list:
                for optimizer_name, batch_size in optimizers.items():
                    for hidden_layers in layer_configurations:
                        self.logger.info(f"Training ANN: LR={lr}, Epochs={epochs}, Optimizer={optimizer_name}, Layers={hidden_layers}...")
                        
                        # ANN Modelini oluştur ve eğit
                        ann_builder = ANNModel(input_dim=self.X_train.shape[1], hidden_layers=hidden_layers, learning_rate=lr)
                        ann_model = ann_builder.build_model()
                        history = ann_model.fit(
                            self.X_train, self.y_train,
                            validation_data=(self.X_val, self.y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0, 
                            callbacks=[EpochLogger(self.logger)]  
                        )

                        # Eğitim seti için tahmin ve metrikler
                        y_pred_train = (ann_model.predict(self.X_train) > 0.5).astype(int).flatten()
                        train_evaluator = MetricsEvaluator(self.y_train, y_pred_train)
                        train_metrics = train_evaluator.get_metrics()

                        # Validation seti için tahmin ve metrikler
                        y_pred_val = (ann_model.predict(self.X_val) > 0.5).astype(int).flatten()
                        val_evaluator = MetricsEvaluator(self.y_val, y_pred_val)
                        val_metrics = val_evaluator.get_metrics()

                        # Eğitim ve Validation için Confusion Matrix çiz
                        self.visualizer.plot_confusion_matrix(
                            y_true=self.y_train,
                            y_pred=y_pred_train,
                            set_type="train",
                            model_name="ANN",
                            params={"LR": lr, "Epochs": epochs, "Layers": hidden_layers, "Optimizer": optimizer_name},
                            save_dir=self.confusion_matrix_dir
                        )

                        self.visualizer.plot_confusion_matrix(
                            y_true=self.y_val,
                            y_pred=y_pred_val,
                            set_type="validation",
                            model_name="ANN",
                            params={"LR": lr, "Epochs": epochs, "Layers": hidden_layers, "Optimizer": optimizer_name},
                            save_dir=self.confusion_matrix_dir
                        )

                        # Metrikleri listeye ekle
                        self.metrics_list.append([
                            "ANN", lr, epochs, hidden_layers, optimizer_name,
                            train_metrics, val_metrics
                        ])

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

        self.logger.info("ANN training complete.")

    def train_svm(self, kernel_params):
        """
        SVM modellerini eğitim için hiperparametre taraması yapar.

        Parametreler:
        kernel_params -- Çekirdek türleri ve parametrelerinin sözlüğü
        """

        for kernel, params in kernel_params.items():
            for C in params.get("C", [1]):
                for degree in params.get("degree", [3]):
                    for gamma in params.get("gamma", ["scale"]):
                        self.logger.info(f"Training SVM: Kernel={kernel}, C={C}, Degree={degree}, Gamma={gamma}...")
                        
                        # SVM Modelini oluştur ve eğit
                        svm_builder = SVMModel(kernel=kernel, C=C, degree=degree, gamma=gamma)
                        svm_model = svm_builder.build_model()
                        svm_model.fit(self.X_train, self.y_train)

                        # Eğitim seti için tahmin ve metrikler
                        y_pred_train = svm_model.predict(self.X_train)
                        train_evaluator = MetricsEvaluator(self.y_train, y_pred_train)
                        train_metrics = train_evaluator.get_metrics()

                        # Validation seti için tahmin ve metrikler
                        y_pred_val = svm_model.predict(self.X_val)
                        val_evaluator = MetricsEvaluator(self.y_val, y_pred_val)
                        val_metrics = val_evaluator.get_metrics()

                        # Eğitim ve Validation için Confusion Matrix çiz
                        self.visualizer.plot_confusion_matrix(
                            y_true=self.y_train,
                            y_pred=y_pred_train,
                            set_type="train",
                            model_name="SVM",
                            params={"Kernel": kernel, "C": C, "Degree": degree, "Gamma": gamma},
                            save_dir=self.confusion_matrix_dir
                        )

                        self.visualizer.plot_confusion_matrix(
                            y_true=self.y_val,
                            y_pred=y_pred_val,
                            set_type="validation",
                            model_name="SVM",
                            params={"Kernel": kernel, "C": C, "Degree": degree, "Gamma": gamma},
                            save_dir=self.confusion_matrix_dir
                        )


                        # Metrikleri listeye ekle
                        self.metrics_list.append([
                            "SVM", kernel, C, degree, gamma,
                            train_metrics, val_metrics
                        ])

                        decision_boundary_path = os.path.join(
                            self.decision_plots_dir,
                            f"svm_decision_boundary_{kernel}_C{C}_Degree{degree}_Gamma{gamma}.png"
                        )
                        self.visualizer.plot_decision_boundary(
                            model=svm_model,
                            X=self.X_val,
                            y=self.y_val,
                            save_path=decision_boundary_path,
                            model_type="SVM",
                            kernel=kernel,
                            C=C,
                            degree=degree,
                            gamma=gamma
                        )

                        # Modeli kaydet
                        model_path = os.path.join(
                            self.model_dir,
                            f"svm_model_{kernel}_C{C}_Degree{degree}_Gamma{gamma}.pkl"
                        )
                        joblib.dump(svm_model, model_path)

        self.logger.info("SVM training complete.")

    def save_combined_metrics(self):
        """
        Tüm eğitim ve doğrulama metriklerini birleştirir ve dosyaya kaydeder.
        """
        self.logger.info("\nSaving combined metrics...")
        with open(self.metrics_path, "w") as file:
            header = "{:<8} {:<12} {:<8} {:<8} {:<14} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n"
            file.write(header.format(
                "Model", "Param1", "Param2", "Param3", "Param4",
                "T_Acc", "T_Prec", "T_Recall", "T_F1",
                "V_Acc", "V_Prec", "V_Recall", "V_F1"
            ))

            for entry in self.metrics_list:
                model_type, p1, p2, p3, p4, train_metrics, val_metrics = entry
                row = "{:<8} {:<12} {:<8} {:<8} {:<14} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}\n"
                file.write(row.format(
                    model_type, p1, p2, p3, p4,
                    train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['f1_score'],
                    val_metrics['accuracy'], val_metrics['precision'], val_metrics['recall'], val_metrics['f1_score']
                ))

        self.logger.info(f"Combined metrics saved to {self.metrics_path}.")

            
if __name__ == "__main__":
    trainer = Trainer()
    trainer.prepare_data()

    # ANN Eğitim Ayarları
    # learning_rates = [0.01]
    learning_rates = [0.0001,0.001,0.01, 0.1]

    # epochs_list = [50]
    epochs_list = [50, 250, 500,1000]

    optimizers = {"SGD": 1, "BGD": len(trainer.X_train), "MBGD": 32}
    layer_configurations = [1, 2, 3]

    trainer.train_ann(
        learning_rates=learning_rates,
        epochs_list=epochs_list,
        optimizers=optimizers,
        layer_configurations=layer_configurations
    )

    # SVM Eğitim Ayarları
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

    trainer.train_svm(kernel_params=kernel_params)
    trainer.save_combined_metrics()