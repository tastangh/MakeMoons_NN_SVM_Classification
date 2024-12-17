import os
import sys
import joblib
from dataset import DatasetProcessor
from visualize import Visualizer
from ann_model import ANNModel
from svm_model import SVMModel
from metrics import MetricsEvaluator
import tensorflow as tf

class LogEpoch(tf.keras.callbacks.Callback):
    """Sadece ilk ve son epoch'taki train/val metriklerini loglamak için callback."""
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or epoch == self.params['epochs'] - 1:  # İlk ve son epoch
            print(f"Epoch {epoch+1}/{self.params['epochs']} - "
                  f"loss: {logs['loss']:.4f}, accuracy: {logs['accuracy']:.4f}, "
                  f"val_loss: {logs['val_loss']:.4f}, val_accuracy: {logs['val_accuracy']:.4f}")

# Redirect stdout to a file as well as console
log_file_path = "training_log.txt"
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            if not f.closed:
                f.write(obj)
                f.flush()

    def flush(self):
        for f in self.files:
            try:
                if not f.closed:
                    f.flush()
            except ValueError:
                pass

log_file = open(log_file_path, "w")
sys.stdout = Tee(sys.stdout, log_file)

# Constants
SAVE_DIR = "results"
MODEL_DIR = os.path.join(SAVE_DIR, "models")
PLOTS_DIR = os.path.join(SAVE_DIR, "plots")
LOSS_PLOTS_DIR = os.path.join(PLOTS_DIR, "loss")
DECISION_PLOTS_DIR = os.path.join(PLOTS_DIR, "decision_boundaries")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOSS_PLOTS_DIR, exist_ok=True)
os.makedirs(DECISION_PLOTS_DIR, exist_ok=True)


# Step 1: Dataset Preparation
print("Creating and splitting the dataset...")
dataset_processor = DatasetProcessor()
dataset = dataset_processor.create_dataset()
splits = dataset_processor.split_dataset()

X_train, y_train = splits["train"]
X_val, y_val = splits["validation"]
X_test, y_test = splits["test"]

# Visualize dataset
visualizer = Visualizer(save_dir=PLOTS_DIR)
visualizer.plot_all_data(dataset)
visualizer.plot_splits(splits)

# Step 2: ANN Training
learning_rates = [0.001, 0.01, 0.05, 0.1]
epochs_list = [50]
optimizers = {"SGD": 1, "BGD": len(X_train), "MBGD": 32}
layer_configurations = [1, 2, 3]

metrics_list = []
model_types = []

for lr in learning_rates:
    for epochs in epochs_list:
        for optimizer_name, batch_size in optimizers.items():
            for hidden_layers in layer_configurations:
                print(f"Training ANN: LR={lr}, Epochs={epochs}, Optimizer={optimizer_name}, Layers={hidden_layers}...")
                ann_builder = ANNModel(input_dim=X_train.shape[1], hidden_layers=hidden_layers, learning_rate=lr)
                ann_model = ann_builder.build_model()

                history = ann_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,  # Ara logları kapat
                    callbacks=[LogEpoch()]  # İlk ve son epoch loglama
                )


                y_pred = (ann_model.predict(X_test) > 0.5).astype(int).flatten()
                evaluator = MetricsEvaluator(y_test, y_pred)
                metrics = evaluator.get_metrics()
                metrics_list.append(["ANN", lr, epochs, hidden_layers, optimizer_name, metrics])
                model_types.append("ANN")

                # Save loss plot
                loss_plot_path = os.path.join(LOSS_PLOTS_DIR, f"ann_loss_LR{lr}_Epoch{epochs}_Opt{optimizer_name}_Layers{hidden_layers}.png")
                visualizer.plot_loss(history, optimizer_name, hidden_layers, LOSS_PLOTS_DIR, os.path.basename(loss_plot_path))
                
                # Save decision boundary
                decision_boundary_path = os.path.join(DECISION_PLOTS_DIR, f"ann_decision_boundary_LR{lr}_Epoch{epochs}_Opt{optimizer_name}_Layers{hidden_layers}.png")
                visualizer.plot_decision_boundary(
                    model=ann_model,
                    X=X_test,
                    y=y_test,
                    save_path=decision_boundary_path,
                    model_type="ANN"
                )

                # Save model
                model_path = os.path.join(MODEL_DIR, f"ann_model_LR{lr}_Epoch{epochs}_Opt{optimizer_name}_Layers{hidden_layers}.keras")
                ann_model.save(model_path)

# Step 3: SVM Training
kernel_params = {
    "linear": {"C": [0.1, 1]},
    "poly": {"C": [0.1], "degree": [2]},
    "rbf": {"C": [0.1], "gamma": ["scale"]}
}

for kernel, params in kernel_params.items():
    for C in params.get("C", [1]):
        for degree in params.get("degree", [3]):
            for gamma in params.get("gamma", ["scale"]):
                print(f"Training SVM: Kernel={kernel}, C={C}, Degree={degree}, Gamma={gamma}...")
                svm_builder = SVMModel(kernel=kernel, C=C, degree=degree, gamma=gamma)
                svm_model = svm_builder.build_model()
                svm_model.fit(X_train, y_train)

                y_pred = svm_model.predict(X_test)
                evaluator = MetricsEvaluator(y_test, y_pred)
                metrics = evaluator.get_metrics()
                metrics_list.append(["SVM", kernel, C, degree, gamma, metrics])
                model_types.append("SVM")

                # Save decision boundary
                decision_boundary_path = os.path.join(DECISION_PLOTS_DIR, f"svm_decision_boundary_{kernel}_C{C}_Degree{degree}_Gamma{gamma}.png")
                visualizer.plot_decision_boundary(
                    model=svm_model,
                    X=X_test,
                    y=y_test,
                    save_path=decision_boundary_path,
                    model_type="SVM"
                )

                # Save model
                model_path = os.path.join(MODEL_DIR, f"svm_model_{kernel}_C{C}_Degree{degree}_Gamma{gamma}.pkl")
                joblib.dump(svm_model, model_path)

# Step 4: Save Combined Metrics
metrics_txt_path = os.path.join(SAVE_DIR, "combined_metrics.txt")
with open(metrics_txt_path, "w") as file:
    header = "{:<8} {:<12} {:<8} {:<8} {:<14} {:<10} {:<10} {:<10} {:<10}\n"
    file.write(header.format("Model", "Param1", "Param2", "Param3", "Param4", "Accuracy", "Precision", "Recall", "F1-Score"))

    for entry in metrics_list:
        model_type, p1, p2, p3, p4, metrics = entry
        row = "{:<8} {:<12} {:<8} {:<8} {:<14} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}\n"
        file.write(row.format(model_type , p1, p2, p3, p4, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']))

print("\nTraining complete! Combined metrics saved to text file.")

# Reset stdout and close log file
sys.stdout = sys.__stdout__
log_file.close()
