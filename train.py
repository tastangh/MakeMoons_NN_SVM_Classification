from dataset import DatasetProcessor
from ann_model import ANNModel
from svm_model import SVMModel
from visualize import Visualizer
from sklearn.model_selection import GridSearchCV
import os
import pickle

class Trainer:
    def __init__(self, n_samples=400, noise=0.2, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.processor = DatasetProcessor(n_samples=n_samples, noise=noise)
        self.visualizer = Visualizer(save_dir="results/train")
        self.splits = None

    def prepare_data(self):
        """
        Veriyi oluşturur ve bölme işlemini yapar.
        """
        dataset = self.processor.create_dataset()
        self.splits = self.processor.split_dataset()
        self.visualizer.plot_all_data(dataset)
        self.visualizer.plot_splits(self.splits)

    def train_ann_models(self):
        """
        Yapay sinir ağlarını eğitir ve görselleştirir.
        """
        X_train, y_train = self.splits["train"]
        X_val, y_val = self.splits["validation"]
        batch_sizes = {"SGD": 1, "BGD": len(X_train), "MBGD": 16}
        hidden_layer_configs = [1, 2, 3]

        for hidden_layers in hidden_layer_configs:
            for optimizer_name, batch_size in batch_sizes.items():
                ann = ANNModel(input_dim=X_train.shape[1], hidden_layers=hidden_layers, learning_rate=self.learning_rate)
                history = ann.train(X_train, y_train, X_val, y_val, epochs=self.epochs, batch_size=batch_size)
                self.visualizer.plot_loss(history, optimizer_name, hidden_layers)

                # Modeli kaydet
                os.makedirs("models", exist_ok=True)
                model_path = f"models/ann_{optimizer_name}_{hidden_layers}_hidden_layers.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(ann, f)

                # Karar sınırları
                self.visualizer.plot_decision_boundary(
                    ann, X_train, y_train,
                    save_path=f"results/train/ann_{optimizer_name}_{hidden_layers}_decision_boundary.png",
                    model_type="ANN"
                )

    def train_svm_models(self):
        """
        SVM modellerini eğitir ve görselleştirir.
        """
        X_train, y_train = self.splits["train"]
        kernels = ["linear", "poly", "rbf"]
        best_models = {}

        for kernel in kernels:
            print(f"Kernel: {kernel}")

            # Parametre aralıkları
            param_grid = {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"] if kernel != "linear" else ["scale"],
                "degree": [2, 3] if kernel == "poly" else [3]  # Polinomsal kernel için derece
            }
            svm = SVMModel(kernel=kernel)
            grid_search = GridSearchCV(svm.model, param_grid, scoring="accuracy", cv=3)
            grid_search.fit(X_train, y_train)

            # En iyi model ve parametreleri kaydet
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            print(f"En iyi parametreler: {best_params}")

            best_models[kernel] = best_model

            # Modeli kaydet
            os.makedirs("models", exist_ok=True)
            model_path = f"models/svm_{kernel}_best_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)

            # Karar sınırlarını çizdir
            self.visualizer.plot_decision_boundary(
                best_model, X_train, y_train,
                save_path=f"results/train/svm_{kernel}_decision_boundary.png",
                model_type="SVM"
            )

        return best_models

if __name__ == "__main__":
    trainer = Trainer()
    trainer.prepare_data()
    trainer.train_ann_models()  # ANN modellerini eğit
    trainer.train_svm_models()  # SVM modellerini eğit
