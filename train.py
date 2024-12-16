import os
import sys
import joblib
from dataset import DatasetProcessor
from visualize import Visualizer
from ann_model import ANNModel
from svm_model import SVMModel
from metrics import MetricsEvaluator
import tensorflow as tf

# Redirect stdout to a file as well as console
log_file_path = "training_log.txt"
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

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
epochs_list = [50, 100, 500, 1000]
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
                ann_model.save(os.path.join(result_path, f"ann_model.h5"))

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

# Step 4: Evaluate the Best Models
print("\nEvaluating the best models...")

# ANN Evaluation
if best_ann_model:
    print("Evaluating the best ANN model...")
    ann_y_pred = (best_ann_model.predict(X_test) > 0.5).astype(int).flatten()
    ann_evaluator = MetricsEvaluator(y_test, ann_y_pred)
    ann_metrics = ann_evaluator.get_metrics()
    ann_evaluator.pretty_print()
    visualizer.plot_confusion_matrix(
        cm=ann_metrics["confusion_matrix"],
        class_labels=["Class 0", "Class 1"],
        title="Best ANN Confusion Matrix",
        save_path=os.path.join(SAVE_DIR, "best_ann_confusion_matrix.png")
    )
    visualizer.plot_decision_boundary(
        model=best_ann_model,
        X=X_test,
        y=y_test,
        save_path=os.path.join(SAVE_DIR, "best_ann_decision_boundary.png"),
        model_type="ANN"
    )

# SVM Evaluation
if best_svm_model:
    print("Evaluating the best SVM model...")
    svm_y_pred = best_svm_model.predict(X_test)
    svm_evaluator = MetricsEvaluator(y_test, svm_y_pred)
    svm_metrics = svm_evaluator.get_metrics()
    svm_evaluator.pretty_print()
    visualizer.plot_confusion_matrix(
        cm=svm_metrics["confusion_matrix"],
        class_labels=["Class 0", "Class 1"],
        title="Best SVM Confusion Matrix",
        save_path=os.path.join(SAVE_DIR, "best_svm_confusion_matrix.png")
    )
    visualizer.plot_decision_boundary(
        model=best_svm_model,
        X=X_test,
        y=y_test,
        save_path=os.path.join(SAVE_DIR, "best_svm_decision_boundary.png"),
        model_type="SVM"
    )

# Step 5: Combined Metrics Table
print("Generating combined metrics table...")
metrics_table_path = os.path.join(SAVE_DIR, "combined_metrics_table.png")
visualizer.plot_combined_metrics_table(
    metrics_list=metrics_list,
    optimizer_names=optimizer_names,
    hidden_layers_list=hidden_layers_list,
    lrs=lrs,
    epochs_list=epochs_list_final,
    save_path=metrics_table_path
)

print("\nTraining and evaluation complete! All results saved in the 'results' and 'models' directories.")

# Close log file
log_file.close()
