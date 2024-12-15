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

    def plot_loss(self, history, optimizer_name, hidden_layers):
        """
        Eğitim ve doğrulama kayıplarını epoch bazında çizer ve kaydeder.

        Args:
        - history: Modelin eğitim geçmişi (Keras History nesnesi)
        - optimizer_name: Optimizasyon yöntemi (örn. "SGD", "Batch GD")
        - hidden_layers: Gizli katman sayısı
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["loss"], label="Eğitim Kaybı", marker='o')
        plt.plot(history.history["val_loss"], label="Doğrulama Kaybı", marker='x')
        plt.xlabel("Epoch")
        plt.ylabel("Kayıp (Loss)")
        plt.title(f"{optimizer_name} - {hidden_layers} Gizli Katman")
        plt.legend()
        save_path = os.path.join(self.save_dir, f"ann_{optimizer_name}_{hidden_layers}_loss.png")
        plt.savefig(save_path)
        plt.show()

    def plot_decision_boundary(self, model, X, y, save_path, model_type="ANN"):
        """
        Karar sınırlarını çizer ve kaydeder.

        Args:
        - model: Eğitimli model (ANN veya SVM)
        - X: Özellikler
        - y: Etiketler
        - save_path: Kaydedilecek dosya yolu
        - model_type: Model türü ("ANN" veya "SVM")
        """
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        if model_type == "ANN":
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).flatten()
        elif model_type == "SVM":
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        plt.title(f"{model_type} Decision Boundary")
        plt.savefig(save_path)
        plt.show()

    def plot_confusion_matrix(self, cm, class_labels, title, save_path):
        """
        Confusion matrix'i görselleştirir.

        Args:
        - cm: Confusion matrix (sklearn.metrics.confusion_matrix çıktısı)
        - class_labels: Sınıf etiketleri (örn. ["Class 0", "Class 1"])
        - title: Grafik başlığı
        - save_path: Kaydedilecek dosya yolu
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Tahmin Edilen")
        plt.ylabel("Gerçek")
        plt.title(title)
        plt.savefig(save_path)
        plt.show()

    def plot_combined_metrics_table(self, metrics_list, optimizer_names, hidden_layers_list, save_path):
        """
        Birden fazla metrik setini düzenli bir tablo halinde çizer ve kaydeder.

        Args:
        - metrics_list: Her bir optimizasyon yöntemi ve gizli katman sayısı için metrik değerlerini içeren liste.
        - optimizer_names: Optimizasyon yöntemlerinin isimleri (örn. ["SGD", "MBGD", "BGD"]).
        - hidden_layers_list: Her metrik setine karşılık gelen gizli katman sayıları.
        - save_path: Kaydedilecek dosya yolu.
        """
        # Tablo başlıkları
        headers = ["Optimizer", "Hidden Layers", "Accuracy", "Precision", "Recall", "F1-Score"]

        # Gruplandırılmış tablo verisi
        rows = []
        optimizers = sorted(set(optimizer_names))  # Benzersiz optimizer isimleri

        for optimizer in optimizers:
            for hidden_layers in [1, 2, 3]:  # 1, 2, 3 katmanlı modeller
                for i, opt_name in enumerate(optimizer_names):
                    if opt_name == optimizer and hidden_layers_list[i] == hidden_layers:
                        metrics = metrics_list[i]
                        rows.append([
                            optimizer if hidden_layers == 1 else "",  # İlk katman için optimizer ismini ekle
                            hidden_layers,
                            f"{metrics['accuracy']:.2f}",
                            f"{metrics['precision']:.2f}",
                            f"{metrics['recall']:.2f}",
                            f"{metrics['f1_score']:.2f}"
                        ])
                        break

        # Grafiksel tablo oluşturma
        plt.figure(figsize=(12, 8))
        plt.axis("off")
        plt.title("Combined Metrics Table", fontsize=16, fontweight="bold")

        table = plt.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center", colLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(headers))))

        # Kaydet ve göster
        plt.savefig(save_path)
        plt.show()

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
        plt.show()