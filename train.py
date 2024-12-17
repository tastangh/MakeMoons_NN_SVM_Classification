import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from ann_model import ANNTrainer
import matplotlib.pyplot as plt

# Veri kümesini oluştur
X, y = make_moons(n_samples=400, noise=0.2, random_state=42)
y = y.reshape(-1, 1)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Parametre kombinasyonları
learning_rates = [0.1, 0.01, 0.001]
epochs_list = [100, 500, 1000]
hidden_layers_list = [[8], [8, 8], [8, 8, 8]]
batch_sizes = {"SGD": 1, "BGD": None, "MBGD": 32}

def plot_loss(history, title, save_path):
    """Eğitim ve doğrulama loss grafiğini çizer ve kaydeder."""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label="Eğitim Loss")
    plt.plot(history.history['val_loss'], label="Doğrulama Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_decision_boundary(model, X, y, save_path):
    """Modelin karar sınırlarını çizer."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolor="k")
    plt.title("Karar Sınırları")
    plt.savefig(save_path)
    plt.close()

# Eğitim döngüsü
for learning_rate in learning_rates:
    for epochs in epochs_list:
        for hidden_layers in hidden_layers_list:
            for gd_type, batch_size in batch_sizes.items():
                print(f"### Eğitim Başlıyor - lr: {learning_rate}, epochs: {epochs}, Katmanlar: {hidden_layers}, Yöntem: {gd_type} ###")
                
                results_dir = f"results/lr_{learning_rate}_epochs_{epochs}_{gd_type}_layers_{len(hidden_layers)}"
                trainer = ANNTrainer(
                    input_dim=2, 
                    hidden_layers=hidden_layers, 
                    learning_rate=learning_rate, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    results_dir=results_dir
                )
                
                # Modeli eğit
                history = trainer.model.fit(
                    X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_data=(X_val, y_val), 
                    verbose=0
                )
                
                # Loss grafiğini çiz
                plot_loss(history, 
                          f"Loss Grafiği - lr: {learning_rate}, epochs: {epochs}, {gd_type}, {len(hidden_layers)} Katman",
                          f"{results_dir}/loss_plot.png")
                
                # Modelin karar sınırlarını çiz
                plot_decision_boundary(trainer.model, X_train, y_train, f"{results_dir}/decision_boundary.png")

                print(f"### Eğitim Tamamlandı - lr: {learning_rate}, epochs: {epochs}, Katmanlar: {hidden_layers}, Yöntem: {gd_type} ###\n")
