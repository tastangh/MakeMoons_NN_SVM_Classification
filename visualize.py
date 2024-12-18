import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

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


    def plot_loss(self, history, learning_rate, epochs, optimizer_name, hidden_layers, save_dir):
        """
        Eğitim ve doğrulama kayıplarını epoch bazında çizer.

        Args:
        - history: Model eğitim geçmişi.
        - learning_rate (float): Öğrenme oranı.
        - epochs (int): Epoch sayısı.
        - optimizer_name (str): Optimizasyon algoritması.
        - hidden_layers (int): Gizli katman sayısı.
        - save_dir (str): Kayıt dizini.
        """
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"ann_loss_LR{learning_rate}_Epochs{epochs}_Opt{optimizer_name}_Layers{hidden_layers}.png"
        title = (f"Kayıp Grafiği - LR: {learning_rate}, Epochs: {epochs}, "
                 f"Optimizer: {optimizer_name}, Layers: {hidden_layers}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["loss"], label="Eğitim Kaybı", marker='o')
        plt.plot(history.history["val_loss"], label="Doğrulama Kaybı", marker='x')
        plt.xlabel("Epoch")
        plt.ylabel("Kayıp (Loss)")
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()

    def plot_decision_boundary(self, model, X, y, save_path, model_type="ANN", **kwargs):
        """
        Modelin karar sınırını çizer ve belirtilen dosya yoluna kaydeder.

        Args:
            model: Eğitimli model (ANN veya SVM).
            X: Veri kümesi (özellikler).
            y: Veri kümesi (etiketler).
            save_path: Karar sınırının kaydedileceği dosya yolu.
            model_type: Model tipi ("ANN" veya "SVM").
            **kwargs: Ek parametreler (LR, Epoch, Kernel, Degree, Gamma gibi).
        """
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Model tahminleri
        if model_type == "ANN":
            Z = (model.predict(grid).ravel() > 0.5).astype(int)
        elif model_type == "SVM":
            Z = model.predict(grid)
        else:
            raise ValueError("Geçersiz model tipi: 'ANN' veya 'SVM' olmalı.")

        Z = Z.reshape(xx.shape)

        # Grafik çizimi
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors='k', marker='o')

        # Başlık düzenlemesi: Model parametrelerini ekleyelim
        title_details = "\n".join([f"{key}={value}" for key, value in kwargs.items()])
        plt.title(f"Decision Boundary ({model_type})\n{title_details}", fontsize=10, wrap=True)
        
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, set_type, model_name, params, save_dir):
            """
            Confusion matrix'i çizip kaydeder.

            Args:
                y_true: Gerçek etiketler.
                y_pred: Model tahminleri.
                set_type: 'train' veya 'validation' gibi veri kümesi tipi.
                model_name: Model adı ('ANN', 'SVM').
                params: Model parametrelerini içeren sözlük.
                save_dir: Kaydedilecek dizin yolu.
            """
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

            # Başlık
            title = f"{model_name} Confusion Matrix ({set_type})\n" + ", ".join([f"{k}={v}" for k, v in params.items()])
            plt.title(title)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')

            # Detaylı dosya ismi
            param_str = "_".join([f"{k}{v}" for k, v in params.items()])
            file_name = f"{model_name}_CM_{set_type}_{param_str}.png"
            file_path = os.path.join(save_dir, file_name)

            plt.tight_layout()
            plt.savefig(file_path)
            plt.close()
            print(f"Confusion matrix saved: {file_path}")

    def plot_model_comparison(self, ann_metrics, svm_metrics, save_path):
        """
        ANN ve SVM modellerinin metriklerini bar grafiği ile karşılaştırır.

        Args:
        - ann_metrics: ANN için metrikler
        - svm_metrics: SVM için metrikler
        - save_path: Kaydedilecek dosya yolu
        """
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        ann_values = [ann_metrics[m.lower()] for m in metrics]
        svm_values = [svm_metrics[m.lower()] for m in metrics]

        x = range(len(metrics))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x, ann_values, width, label="ANN", color="blue")
        plt.bar([p + width for p in x], svm_values, width, label="SVM", color="orange")

        plt.xlabel("Metrikler")
        plt.ylabel("Değerler")
        plt.title("ANN ve SVM Karşılaştırması")
        plt.xticks([p + width / 2 for p in x], metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        # plt.show()