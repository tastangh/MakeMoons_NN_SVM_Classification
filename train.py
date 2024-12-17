import os
import sys
import joblib
from dataset import DatasetProcessor
from visualize import Visualizer
from ann_model import ANNModel
from svm_model import SVMModel
from metrics import MetricsEvaluator
import tensorflow as tf

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            if not f.closed:  # Dosya kapalı değilse yaz
                f.write(obj)
                f.flush()

    def flush(self):
        for f in self.files:
            try:
                if not f.closed:
                    f.flush()
            except ValueError:
                pass  # Dosya zaten kapalıysa hata görmezden gel

# Redirect output
log_file = open(log_file_path, "w")
sys.stdout = Tee(sys.stdout, log_file)

# Constants
SAVE_DIR = "results"
MODEL_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Step 1: Dataset Preparation
print("Creating and splitting the dataset...")
dataset_processor = DatasetProcessor()
dataset = dataset_processor.create_dataset()
splits = dataset_processor.split_dataset()

X_train, y_train = splits["train"]
X_val, y_val = splits["validation"]
X_test, y_test = splits["test"]

# Visualize dataset
visualizer = Visualizer(save_dir=SAVE_DIR)
visualizer.plot_all_data(dataset)
visualizer.plot_splits(splits)

# Step 2: ANN Training with SGD, BGD, MBGD
learning_rates = [0.001, 0.01, 0.05, 0.1]
epochs_list = [50]
# epochs_list = [50, 100, 500, 1000]
optimizers = {"SGD": 1, "BGD": len(X_train), "MBGD": 32}
layer_configurations = [1, 2, 3]

best_ann_model = None
best_ann_val_loss = float("inf")

# Collect data for the metrics table
metrics_list = []
optimizer_names = []
hidden_layers_list = []
lrs = []
epochs_list_final = []

for lr in learning_rates:
    for epochs in epochs_list:
        for optimizer_name, batch_size in optimizers.items():
            for hidden_layers in layer_configurations:
                print(f"Training ANN: LR={lr}, Epochs={epochs}, Optimizer={optimizer_name}, Layers={hidden_layers}...")

                # Initialize and train the ANN model
                ann_builder = ANNModel(input_dim=X_train.shape[1], hidden_layers=hidden_layers, learning_rate=lr)
                ann_model = ann_builder.build_model()
                
                history = ann_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )

                # Calculate metrics
                val_accuracy = max(history.history["val_accuracy"])
                val_loss = min(history.history["val_loss"])
                y_pred = (ann_model.predict(X_test) > 0.5).astype(int).flatten()
                evaluator = MetricsEvaluator(y_test, y_pred)
                metrics = evaluator.get_metrics()

                # Append metrics and corresponding info
                metrics_list.append(metrics)
                optimizer_names.append(optimizer_name)
                hidden_layers_list.append(hidden_layers)
                lrs.append(lr)
                epochs_list_final.append(epochs)

                # Save model and visualize results
                result_path = os.path.join(SAVE_DIR, f"LR_{lr}_Epoch_{epochs}/Optimizer_{optimizer_name}/Layers_{hidden_layers}")
                os.makedirs(result_path, exist_ok=True)

                visualizer.plot_loss(history, optimizer_name=optimizer_name, hidden_layers=hidden_layers, save_dir=result_path)

                # Modeli .keras formatında kaydet
                ann_model.save(os.path.join(result_path, f"ann_model.keras"))

                # Update best model based on validation loss
                if val_loss < best_ann_val_loss:
                    best_ann_val_loss = val_loss
                    best_ann_model = ann_model

# Step 3: SVM Training with Kernel Variations
print("\nTraining SVM models with different kernels and parameters...")
best_svm_model = None
best_svm_accuracy = 0

kernel_params = {
    "linear": {"C": [0.1, 1, 10]},
    "poly": {"C": [0.1, 1], "degree": [2, 3]},
    "rbf": {"C": [0.1, 1], "gamma": ["scale", "auto"]}
}

for kernel, params in kernel_params.items():
    for C in params.get("C", [1]):
        for degree in params.get("degree", [3]):
            for gamma in params.get("gamma", ["scale"]):
                print(f"Training SVM: Kernel={kernel}, C={C}, Degree={degree}, Gamma={gamma}...")
                
                svm_builder = SVMModel(kernel=kernel, C=C, degree=degree, gamma=gamma)
                svm_model = svm_builder.build_model()
                svm_model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_accuracy = svm_model.score(X_val, y_val)
                print(f"Validation Accuracy: {val_accuracy:.4f}")
                
                result_path = os.path.join(SAVE_DIR, f"SVM_{kernel}_C_{C}_Degree_{degree}_Gamma_{gamma}")
                os.makedirs(result_path, exist_ok=True)

                # Save model and update best model
                joblib.dump(svm_model, os.path.join(result_path, f"svm_model.pkl"))
                if val_accuracy > best_svm_accuracy:
                    best_svm_accuracy = val_accuracy
                    best_svm_model = svm_model

# Step 4: Save Metrics to Text File
metrics_txt_path = os.path.join(SAVE_DIR, "combined_metrics.txt")
with open(metrics_txt_path, "w") as file:
    # Sabit sütun genişliklerini belirle
    header_format = "{:<12} {:<8} {:<8} {:<14} {:<10} {:<10} {:<10} {:<10}\n"
    row_format = "{:<12} {:<8.4f} {:<8} {:<14} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}\n"

    # Başlıkları yaz
    file.write(header_format.format("Optimizer", "LR", "Epochs", "Hidden Layers", "Accuracy", "Precision", "Recall", "F1-Score"))

    # Her bir satır için veriyi yaz
    for i, metrics in enumerate(metrics_list):
        file.write(row_format.format(
            optimizer_names[i],
            lrs[i],
            epochs_list_final[i],
            hidden_layers_list[i],
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ))

print("\nTraining complete! Combined metrics saved to text file.")

#  Reset stdout and close log file
sys.stdout = sys.__stdout__
log_file.close()