import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pandas as pd

# Sınıf veri işleme işleri için
class DatasetProcessor:
    """
    Veri işlemleri: veri oluşturma, görselleştirme ve veri setlerini bölme gibi işlemler.
    Bu sınıfın genel yapısı oldukça esnek; ileride gerekirse geliştirilebilir.
    """

    def __init__(self, n_samples=400, noise=0.2, random_state=42, save_dir="results"):
        """
        DatasetProcessor sınıfını başlatır.

        Args:
        - n_samples: Kaç tane veri üretileceği (Varsayılan 400) (Ödevde 400 olarak belirtlimiş.)
        - noise: Verilere eklenecek gürültü seviyesi (Varsayılan 0.2)
        - random_state: Rastgelelik için seed değeri (Varsayılan 42)
        - save_dir: Grafiklerin saklanacağı klasör (Varsayılan 'results')
        """
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.save_dir = save_dir

        # Dizin zaten varsa hata vermesin, oluşturalım
        os.makedirs(self.save_dir, exist_ok=True)  
        self.dataset = None  # Veri kümesi için placeholder
        self.splits = {}  # Eğitim, doğrulama, test verilerini tutar

    def create_dataset(self):
        """
        make_moons fonksiyonu ile veri kümesi oluşturur.

        Returns:
        - pd.DataFrame: Özellikler ve etiketlerden oluşan veri çerçevesi
        """
        X, y = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        # Pandas DataFrame'e dönüştürüp sütun isimleri ekliyoruz
        self.dataset = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
        self.dataset['Target'] = y
        return self.dataset

    def split_dataset(self):
        """
        Veri kümesini eğitim, doğrulama ve test setlerine böler.

        Returns:
        - dict: Eğitim, doğrulama ve test setlerini içeren bir sözlük
        """
        if self.dataset is None:
            raise ValueError("Önce create_dataset() fonksiyonunu çağırın!")

        # Özellikler ve hedef sütununu ayıralım
        X = self.dataset[['Feature1', 'Feature2']].values
        y = self.dataset['Target'].values

        # Eğitim ve geçici (temp) seti ayırıyoruz
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=self.random_state)

        # Temp setini doğrulama ve test olarak ayırıyoruz
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.random_state)

        # Tüm setleri saklayalım
        self.splits = {
            "train": (X_train, y_train),
            "validation": (X_val, y_val),
            "test": (X_test, y_test)
        }
        return self.splits

    def plot_all_data(self):
        """
        Tüm veri kümesinin genel görünümünü çizer ve kaydeder.
        """
        if self.dataset is None:
            raise ValueError("Önce create_dataset() fonksiyonunu çağırın!")

        X = self.dataset[['Feature1', 'Feature2']].values
        y = self.dataset['Target'].values

        plt.figure(figsize=(8, 6))
        # Sınıf 0 için kırmızı x, sınıf 1 için yeşil o
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', label='Sınıf 0', alpha=0.6)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='green', marker='o', label='Sınıf 1', alpha=0.6)
        plt.xlabel("Özellik 1")
        plt.ylabel("Özellik 2")
        plt.title("Tüm Veri Kümesi Dağılımı")
        plt.legend()

        # Grafik kaydetme
        save_path = os.path.join(self.save_dir, "tum_veri_seti.png")
        plt.savefig(save_path)
        plt.show()

    def plot_individual(self, X, y, title, save_file):
        """
        Belirli bir setin (eğitim/validasyon/test) dağılımını çizer ve kaydeder.

        Args:
        - X: Özellikler
        - y: Etiketler
        - title: Grafik başlığı
        - save_file: Kaydedilecek dosya ismi
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', label='Sınıf 0', alpha=0.6)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='green', marker='o', label='Sınıf 1', alpha=0.6)
        plt.xlabel("Özellik 1")
        plt.ylabel("Özellik 2")
        plt.title(title)
        plt.legend()

        save_path = os.path.join(self.save_dir, save_file)
        plt.savefig(save_path)
        plt.show()

    def plot_splits(self):
        """
        Eğitim, doğrulama ve test setlerini ayrı ayrı çizer.
        """
        if not self.splits:
            raise ValueError("Önce split_dataset() fonksiyonunu çağırın!")

        # Eğitim, doğrulama ve test setlerinin grafiğini çiziyoruz
        if "train" in self.splits:
            self.plot_individual(*self.splits["train"], "Eğitim Seti", "egitim_seti.png")
        if "validation" in self.splits:
            self.plot_individual(*self.splits["validation"], "Doğrulama Seti", "dogrulama_seti.png")
        if "test" in self.splits:
            self.plot_individual(*self.splits["test"], "Test Seti", "test_seti.png")


# Bu dosya doğrudan çalıştırıldığında burası devreye girer
if __name__ == "__main__":
    # DatasetProcessor'ı varsayılan değerlerle başlatıyoruz
    processor = DatasetProcessor()

    # Veri setini oluştur ve çiz
    processor.create_dataset()
    processor.plot_all_data()

    # Veri setini böl ve her parçayı çiz
    processor.split_dataset()
    processor.plot_splits()
