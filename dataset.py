import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pandas as pd

class DatasetProcessor:
    """
    Veri kümesi işlemleri için sınıf.
    Veri oluşturma, görselleştirme ve eğitim, doğrulama, test setlerine bölme işlemlerini içerir.
    """

    def __init__(self, n_samples=400, noise=0.2, random_state=42, save_dir="results"):
        """
        DatasetProcessor sınıfını başlatır.

        Args:
        - n_samples (int): Üretilecek toplam veri sayısı. Varsayılan 400.
        - noise (float): Verilere eklenecek gürültü miktarı. Varsayılan 0.2.
        - random_state (int): Rastgelelik için seed değeri. Varsayılan 42.
        - save_dir (str): Grafiklerin kaydedileceği dizin. Varsayılan 'results'.
        """
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True) 
        self.dataset = None
        self.splits = {}

    def create_dataset(self):
        """
        make_moons fonksiyonu ile veri kümesi oluşturur.

        Returns:
        - pd.DataFrame: Veri kümesini DataFrame formatında döner.
        """
        X, y = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        self.dataset = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
        self.dataset['Target'] = y
        return self.dataset

    def split_dataset(self):
        """
        Veri kümesini eğitim, doğrulama ve test setlerine böler.

        Returns:
        - dict: Eğitim, doğrulama ve test setlerini içeren bir sözlük.
        """
        if self.dataset is None:
            raise ValueError("Veri kümesi oluşturulmadı. Önce create_dataset() fonksiyonunu çağırın.")
        
        X = self.dataset[['Feature1', 'Feature2']].values
        y = self.dataset['Target'].values

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=self.random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.random_state)

        self.splits = {
            "train": (X_train, y_train),
            "validation": (X_val, y_val),
            "test": (X_test, y_test)
        }
        return self.splits

    def plot_all_data(self):
        """
        Tüm veri setinin dağılımını çiz ve kaydet.
        """
        if self.dataset is None:
            raise ValueError("Veri kümesi oluşturulmadı. Önce create_dataset() fonksiyonunu çağırın.")
        
        X = self.dataset[['Feature1', 'Feature2']].values
        y = self.dataset['Target'].values
        plt.figure(figsize=(8, 6))
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', label='Sınıf 0 (Kırmızı)', alpha=0.6)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='green', marker='o', label='Sınıf 1 (Yeşil)', alpha=0.6)
        plt.xlabel("Özellik 1")
        plt.ylabel("Özellik 2")
        plt.title("Tüm Veri Seti")
        plt.legend()
        save_path = os.path.join(self.save_dir, "tum_veri_seti.png")
        plt.savefig(save_path)
        plt.show()

    def plot_individual(self, X, y, title, save_file):
        """
        Bireysel setin grafiğini çizer (eğitim, doğrulama veya test seti).

        Args:
        - X (ndarray): Özellikler.
        - y (ndarray): Etiketler.
        - title (str): Grafik başlığı.
        - save_file (str): Kaydedilecek dosya adı.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', label='Sınıf 0 (Kırmızı)', alpha=0.6)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='green', marker='o', label='Sınıf 1 (Yeşil)', alpha=0.6)
        plt.xlabel("Özellik 1")
        plt.ylabel("Özellik 2")
        plt.title(title)
        plt.legend()
        save_path = os.path.join(self.save_dir, save_file)
        plt.savefig(save_path)
        plt.show()

    def plot_splits(self):
        """
        Eğitim, doğrulama ve test setlerinin dağılımlarını ayrı ayrı çiz ve kaydet.
        """
        if not self.splits:
            raise ValueError("Veri kümesi bölünmedi. Önce split_dataset() fonksiyonunu çağırın.")
        
        train_data = self.splits.get("train")
        val_data = self.splits.get("validation")
        test_data = self.splits.get("test")

        if train_data:
            self.plot_individual(*train_data, "Eğitim Seti", "egitim_seti.png")
        if val_data:
            self.plot_individual(*val_data, "Doğrulama Seti", "dogrulama_seti.png")
        if test_data:
            self.plot_individual(*test_data, "Test Seti", "test_seti.png")


# Ana çalışma bloğu
if __name__ == "__main__":
    processor = DatasetProcessor()
    processor.create_dataset()
    processor.plot_all_data()
    processor.split_dataset()
    processor.plot_splits()
