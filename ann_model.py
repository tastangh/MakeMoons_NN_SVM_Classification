from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

class ANNModel:
    """
    Yapay Sinir Ağları (ANN) modelini oluşturma ve yönetme sınıfı.

    Bu sınıf, belirli sayıda giriş özelliği, gizli katman ve öğrenme oranı ile
    bir Yapay Sinir Ağı (ANN) modeli oluşturmak için kullanılır.
    """

    def __init__(self, input_dim, hidden_layers=1, hidden_units=32, learning_rate=0.01):
        """
        ANNModel sınıfının yapıcı fonksiyonu.

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
        self.model = None  

    def build_model(self):
        """
        Yapay Sinir Ağı modelini oluşturur ve derler.

        Bu yöntem, verilen parametrelere göre bir Yapay Sinir Ağı (ANN) modeli oluşturur.
        Modelin gizli katman sayısı ve nöron sayısı, sınıfın özelliklerinden alınır. 
        Model, sigmoid aktivasyon fonksiyonu ile derlenir.

        Returns:
            tensorflow.keras.Model: Derlenmiş ANN modeli.
        """
        # Modeli oluştur
        model = Sequential()

        # Giriş katmanını Input layer ile tanımla
        model.add(Input(shape=(self.input_dim,)))

        # İlk gizli katmanı ekle
        model.add(Dense(self.hidden_units, activation='sigmoid'))

        # Ek gizli katmanlar ekle
        for _ in range(self.hidden_layers - 1):
            model.add(Dense(self.hidden_units, activation='sigmoid'))

        # Çıkış katmanı ekle
        model.add(Dense(1, activation='sigmoid'))

        # Optimizasyonu tanımla ve modeli derle
        optimizer = SGD(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Oluşturulan modeli sınıf değişkenine ata
        self.model = model
        return model

