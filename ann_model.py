from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

class ANNModel:
    """
    Yapay Sinir Ağları (ANN) modelini oluşturmak ve yönetmek için bir sınıf.
    """

    def __init__(self, input_dim, hidden_layers=1, hidden_units=32, learning_rate=0.01):
        """
        ANNModel sınıfını başlatır.

        Args:
        - input_dim (int): Giriş katmanındaki özellik sayısı.
        - hidden_layers (int): Gizli katman sayısı (default 1).
        - hidden_units (int): Her gizli katmandaki nöron sayısı (default 32).
        - learning_rate (float): Öğrenme oranı (default 0.01).
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model = None  # Model henüz oluşturulmadı

    def build_model(self):
        """
        Yapay Sinir Ağı modelini oluşturur ve derler.

        Returns:
        - model (tensorflow.keras.Model): Derlenmiş ANN modeli.
        """
        # Modeli oluştur
        model = Sequential()

        # Giriş katmanını Input layer ile tanımla
        model.add(Input(shape=(self.input_dim,)))

        # İlk gizli katman
        model.add(Dense(self.hidden_units, activation='sigmoid'))

        # Ek gizli katmanlar
        for _ in range(self.hidden_layers - 1):
            model.add(Dense(self.hidden_units, activation='sigmoid'))

        # Çıkış katmanı
        model.add(Dense(1, activation='sigmoid'))

        # Optimizasyonu tanımla ve modeli derle
        optimizer = SGD(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Oluşturulan modeli sınıf değişkenine ata
        self.model = model
        return model

