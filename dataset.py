import os
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

class DatasetProcessor:
    """
    Veri setini oluşturma, bölme, kaydetme ve yükleme işlemleri için sınıf.
    """

    def __init__(self, n_samples=400, noise=None, random_state=42, save_dir="dataset"):
        """
        DatasetProcessor sınıfını başlatır.

        Args:
        - n_samples (int): Oluşturulacak örnek sayısı (Ödevde belirtilen: 400). 
        - noise (float): Veriye eklenecek gürültü seviyesi (Ödevde belirtilmemiş: None).
        - random_state (int): Rastgelelik kontrolü için seed (default: 42).
        - save_dir (str): Veri setinin kaydedileceği klasör (default: 'dataset').
        """
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.save_dir = save_dir
        self.dataset = None
        self.splits = {}

        # Dataset klasörünü oluştur
        os.makedirs(self.save_dir, exist_ok=True)

    def create_dataset(self):
        """
        make_moons fonksiyonu ile veri seti oluşturur, kaydeder ve pandas DataFrame döndürür.

        Returns:
        - dataset (pd.DataFrame): Veri seti (Feature1, Feature2, Target sütunlarını içerir).
        """
        X, y = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        self.dataset = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
        self.dataset['Target'] = y
        
        # Veri setini kaydet
        save_path = os.path.join(self.save_dir, "dataset.csv")
        self.dataset.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")

        return self.dataset

    def load_dataset(self):
        """
        Kaydedilmiş veri setini yükler.

        Returns:
        - dataset (pd.DataFrame): Veri seti (Feature1, Feature2, Target sütunlarını içerir).
        """
        load_path = os.path.join(self.save_dir, "dataset.csv")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Dataset file not found at {load_path}. Please create the dataset first.")
        
        self.dataset = pd.read_csv(load_path)
        print(f"Dataset loaded from {load_path}")
        return self.dataset

    def split_dataset(self):
        """
        Veri setini eğitim (%60), doğrulama (%20) ve test (%20) olarak ayırır.

        Returns:
        - splits (dict): Eğitim, doğrulama ve test setlerini içeren sözlük.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not created or loaded. Call create_dataset() or load_dataset() first.")
        
        X = self.dataset[['Feature1', 'Feature2']].values
        y = self.dataset['Target'].values

        # Eğitim ve kalan veri ayırımı
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=self.random_state)
        # Doğrulama ve test veri ayırımı
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.random_state)

        self.splits = {
            "train": (X_train, y_train),
            "validation": (X_val, y_val),
            "test": (X_test, y_test)
        }
        return self.splits
