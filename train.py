from dataset import DatasetProcessor
from ann_model import ANNModel
from visualize import Visualizer
import os
import pickle

class Trainer:
    """
    Eğitim sürecini yöneten sınıf.
    """

    def __init__(self, n_samples=400, noise=0.2, learning_rate=0.01, epochs=100):
        """
        Trainer sınıfını başlatır.

        Args:
        - n_samples: Veri kümesi örnek sayısı
        - noise: Gürültü seviyesi
        - learning_rate: Öğrenme hızı
        - epochs: Epoch sayısı
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.processor = DatasetProcessor(n_samples=n_samples, noise=noise)
        self.visualizer = Visualizer(save_dir="results/train")
        self.batch_sizes = {
            "Stochastic Gradient Descent": 1,
            "Batch Gradient Descent": None,  # Tüm veri (Batch GD)
            "Mini Batch Gradient Descent": 16
        }
        self.hidden_layer_configs = [1, 2, 3]
        self.splits = None

    def prepare_data(self):
        """
        Veriyi oluşturur ve bölme işlemini yapar.
        """
        dataset = self.processor.create_dataset()
        self.splits = self.processor.split_dataset()
        self.visualizer.plot_all_data(dataset)
        self.visualizer.plot_splits(self.splits)

    def train_models(self):
        """
        Tüm ANN modellerini eğitir.
        """
        X_train, y_train = self.splits["train"]
        X_val, y_val = self.splits["validation"]

        for hidden_layers in self.hidden_layer_configs:
            for optimizer_name, batch_size in self.batch_sizes.items():
                batch_size = batch_size or len(X_train)  # Batch GD için tüm veri
                ann = ANNModel(input_dim=X_train.shape[1], hidden_layers=hidden_layers, learning_rate=self.learning_rate)
                history = ann.train(X_train, y_train, X_val, y_val, epochs=self.epochs, batch_size=batch_size)

                # Kayıp grafikleri
                self.visualizer.plot_loss(history, optimizer_name, hidden_layers)

                # Modeli kaydet
                os.makedirs("models", exist_ok=True)
                model_path = f"models/{optimizer_name}_{hidden_layers}_hidden_layers.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(ann, f)

                # Karar sınırlarını çizdir
                self.visualizer.plot_decision_boundary(
                    ann, X_train, y_train,
                    save_path=f"results/train/{optimizer_name}_{hidden_layers}_decision_boundary.png",
                    model_type="ANN"
                )

if __name__ == "__main__":
    trainer = Trainer()
    trainer.prepare_data()
    trainer.train_models()
