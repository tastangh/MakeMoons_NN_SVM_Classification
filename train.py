import os
import sys
import joblib
import tensorflow as tf
from dataset import DatasetProcessor
from visualize import Visualizer
from ann_model import ANNModel
from svm_model import SVMModel
from metrics import MetricsEvaluator

class LogEpoch(tf.keras.callbacks.Callback):
    """
    Sadece ilk ve son epoch'taki train/val metriklerini loglamak için callback.
    """
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or epoch == self.params['epochs'] - 1:  # İlk ve son epoch
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
    Veri hazırlama, model eğitimi ve görselleştirme işlemlerini yöneten Trainer sınıfı.
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
        visualizer = Visualizer(save_dir=self.plots_dir)
        visualizer.plot_all_data(dataset)
        visualizer.plot_splits(splits)

        print("Data preparation complete.")

    def train_model(self, model, epochs=10, batch_size=32):
        """
        Modeli eğitir.
        """
        print("Starting model training...")
        log_epoch = LogEpoch()
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[log_epoch]
        )
        print("Training complete.")
        return history

# Ana çalıştırma kodu
if __name__ == "__main__":
    trainer = Trainer()
    trainer.prepare_data()

    # ANN Modelini kullanarak eğitme
    print("Initializing ANN Model...")
    ann_model = ANNModel(input_shape=trainer.X_train.shape[1:])
    history = trainer.train_model(ann_model, epochs=5, batch_size=32)
