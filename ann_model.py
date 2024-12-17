import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

class ANNTrainer:
    """
    Yapay Sinir Ağı Modeli: Binary Cross-Entropy ve sigmoid aktivasyon fonksiyonu ile eğitir.
    SGD, Batch Gradient Descent ve Mini-Batch seçeneklerini destekler.
    """
    def __init__(self, input_dim, hidden_layers, learning_rate=0.01, epochs=100, batch_size=None, results_dir="results"):
        """
        Args:
            input_dim (int): Giriş katmanının boyutu.
            hidden_layers (list): Gizli katmanlardaki nöron sayısı.
            learning_rate (float): Öğrenme oranı.
            epochs (int): Eğitim epoch sayısı.
            batch_size (int, optional): Batch boyutu. None -> Batch GD, 1 -> SGD.
            results_dir (str): Eğitim sonuçlarının kaydedileceği ana klasör.
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.results_dir = os.path.join(results_dir, f"lr_{self.learning_rate}_epochs_{self.epochs}")
        os.makedirs(self.results_dir, exist_ok=True)
        self.model = self._build_model()

    def _build_model(self):
        """Keras Sequential modelini oluşturur."""
        model = Sequential()
        model.add(Dense(self.hidden_layers[0], activation='sigmoid', input_dim=self.input_dim))
        for neurons in self.hidden_layers[1:]:
            model.add(Dense(neurons, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))  # Binary sınıflandırma için çıkış
        model.compile(optimizer=SGD(learning_rate=self.learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val):
        """Modeli eğitir ve kayıp grafiğini kaydeder."""
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            verbose=0
        )
        self._plot_and_save_loss(history)

    def _plot_and_save_loss(self, history):
        """Eğitim ve doğrulama kaybını kaydeder."""
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], label='Eğitim Loss')
        plt.plot(history.history['val_loss'], label='Doğrulama Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss Grafiği - lr: {self.learning_rate}, epochs: {self.epochs}')
        plt.legend()
        save_path = os.path.join(self.results_dir, "loss_plot.png")
        plt.savefig(save_path)
        plt.close()

    def evaluate(self, X_test, y_test):
        """Modeli test seti üzerinde değerlendirir."""
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        with open(os.path.join(self.results_dir, "evaluation.txt"), "w") as f:
            f.write(f"Test Loss: {loss:.4f}\nTest Accuracy: {accuracy:.4f}\n")
