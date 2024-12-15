from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

class ANNModel:
    def __init__(self, input_dim, hidden_layers, learning_rate=0.01):
        """
        Yapay Sinir Ağı Modeli.
        - input_dim: Giriş boyutu
        - hidden_layers: Gizli katman sayısı
        - learning_rate: Öğrenme hızı
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """
        Sigmoid aktivasyon fonksiyonu ile model inşa eder.
        """
        model = Sequential()
        model.add(Dense(16, activation='sigmoid', input_dim=self.input_dim))  # İlk gizli katman
        for _ in range(self.hidden_layers - 1):
            model.add(Dense(16, activation='sigmoid'))  # Ek gizli katmanlar
        model.add(Dense(1, activation='sigmoid'))  # Çıkış katmanı
        model.compile(optimizer=SGD(learning_rate=self.learning_rate), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """
        Modeli eğitir.
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
        """
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype("int32")
