import os
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

class DatasetProcessor:
    """
    Veri setini oluşturma, bölme ve kaydetme işlemleri için sınıf.
    """

    def __init__(self, n_samples=400, noise=0.2, random_state=42):
        """
        DatasetProcessor sınıfını başlatır.

        Args:
        - n_samples (int): Oluşturulacak örnek sayısı (default: 400).
        - noise (float): Veriye eklenecek gürültü seviyesi (default: 0.2).
        - random_state (int): Rastgelelik kontrolü için seed (default: 42).
        """
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.dataset = None
        self.splits = {}

    def create_dataset(self):
        """
        make_moons fonksiyonu ile veri seti oluşturur ve pandas DataFrame döndürür.

        Returns:
        - dataset (pd.DataFrame): Veri seti (Feature1, Feature2, Target sütunlarını içerir).
        """
        X, y = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        self.dataset = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
        self.dataset['Target'] = y
        return self.dataset

    def split_dataset(self):
        """
        Veri setini eğitim (%60), doğrulama (%20) ve test (%20) olarak ayırır.

        Returns:
        - splits (dict): Eğitim, doğrulama ve test setlerini içeren sözlük.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not created yet. Call create_dataset() first.")
        
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

    def save_dataset(self, path="dataset.csv"):
        """
        Veri setini CSV formatında kaydeder.

        Args:
        - path (str): Kaydedilecek dosya yolu (default: "dataset.csv").
        """
        if self.dataset is None:
            raise ValueError("Dataset is not created yet. Call create_dataset() first.")
        self.dataset.to_csv(path, index=False)
        print(f"Dataset saved to {path}.")

    def load_dataset(self, path="dataset.csv"):
        """
        CSV formatındaki veri setini yükler.

        Args:
        - path (str): Yüklenecek dosya yolu (default: "dataset.csv").

        Returns:
        - dataset (pd.DataFrame): Yüklenen veri seti.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}.")
        self.dataset = pd.read_csv(path)
        print(f"Dataset loaded from {path}.")
        return self.dataset
