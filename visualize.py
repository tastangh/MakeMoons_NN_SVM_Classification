import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Visualizer:
    """
    Görselleştirme işlemlerini yönetir: tüm veri seti, eğitim, doğrulama ve test setleri, kayıp grafikleri ve karar sınırları.
    """

    def __init__(self, save_dir="results"):
        """
        Visualizer sınıfını başlatır.

        Args:
        - save_dir: Grafiklerin kaydedileceği dizin
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_all_data(self, dataset):
        """
        Tüm veri kümesinin genel görünümünü çizer ve kaydeder.

        Args:
        - dataset: DatasetProcessor tarafından üretilmiş veri çerçevesi
        """
        X = dataset[['Feature1', 'Feature2']].values
        y = dataset['Target'].values

        plt.figure(figsize=(8, 6))
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', label='Sınıf 0', alpha=0.6)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='green', marker='o', label='Sınıf 1', alpha=0.6)
        plt.xlabel("Özellik 1")
        plt.ylabel("Özellik 2")
        plt.title("Tüm Veri Kümesi Dağılımı")
        plt.legend()

        save_path = os.path.join(self.save_dir, "tum_veri_seti.png")
        plt.savefig(save_path)
        plt.close()

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
        plt.close()

    def plot_splits(self, splits):
        """
        Eğitim, doğrulama ve test setlerini ayrı ayrı çizer.

        Args:
        - splits: DatasetProcessor tarafından döndürülen veri bölmeleri
        """
        for set_name, (X, y) in splits.items():
            title = f"{set_name.capitalize()} Seti"
            save_file = f"{set_name}_seti.png"
            self.plot_individual(X, y, title, save_file)


