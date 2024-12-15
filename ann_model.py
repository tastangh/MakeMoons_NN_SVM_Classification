from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

class ANNModel:
    """
    Yapay Sinir Ağı modeli oluşturma ve eğitim işlemleri.
    """

    def __init__(self, input_dim, hidden_layers, learning_rate=0.01):
        """
        Model sınıfını başlatır.

        Args:
        - input_dim: Giriş boyutu (özellik sayısı)
        - hidden_layers: Gizli katman sayısı
        - learning_rate: Öğrenme hızı
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """
        Yapay Sinir Ağı modelini oluşturur.

        Returns:
        - model: Derlenmiş Keras modeli
        """
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=self.input_dim))
        for _ in range(self.hidden_layers - 1):
            model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=SGD(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """
        Modeli eğitir.

        Args:
        - X_train: Eğitim özellikleri
        - y_train: Eğitim etiketleri
        - X_val: Doğrulama özellikleri
        - y_val: Doğrulama etiketleri
        - epochs: Epoch sayısı
        - batch_size: Batch boyutu

        Returns:
        - history: Eğitim geçmişi (loss ve accuracy)
        """
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history

    def predict(self, X):
        """
        Model ile tahmin yapar.

        Args:
        - X: Özellikler

        Returns:
        - np.ndarray: Tahminler (0 veya 1)
        """
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype("int32")
