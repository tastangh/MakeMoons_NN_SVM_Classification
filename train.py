
import os
import joblib
from dataset import DatasetProcessor
from visualize import Visualizer
from ann_model import ANNModel
from svm_model import SVMModel
from metrics import MetricsEvaluator
import tensorflow as tf

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

# Step 2: ANN Training
optimized_combinations = [
    (50, 0.0001), (50, 0.001), (50, 0.01), (50, 0.1),
    (100, 0.0001), (100, 0.001), (100, 0.01), (100, 0.1),
    (500, 0.0001), (500, 0.001), (500, 0.01), (500, 0.1),
    (1000, 0.0001), (1000, 0.001), (1000, 0.01), (1000, 0.1),
    (5000, 0.0001), (5000, 0.001), (5000, 0.01), (5000, 0.1)
]

best_ann_model = None
best_ann_val_loss = float("inf")

for epochs, learning_rate in optimized_combinations:
    print(f"Training ANN with epochs={epochs} and learning_rate={learning_rate}...")
    ann_builder = ANNModel(input_dim=X_train.shape[1], hidden_layers=2, learning_rate=learning_rate)
    ann_model = ann_builder.build_model()

    history = ann_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )

    # Visualize training/validation loss
    visualizer.plot_loss(history, optimizer_name=f"LR={learning_rate}_Epochs={epochs}", hidden_layers=2)

    # Save the best model based on validation loss
    val_loss = min(history.history["val_loss"])
    if val_loss < best_ann_val_loss:
        best_ann_val_loss = val_loss
        best_ann_model = ann_model
        ann_model.save(os.path.join(MODEL_DIR, f"best_ann_epochs{epochs}_lr{learning_rate}.h5"))

    # Save all models for analysis
    ann_model.save(os.path.join(MODEL_DIR, f"ann_epochs{epochs}_lr{learning_rate}.h5"))

# Step 3: SVM Training with Kernel Variations
print("Training SVM models with different kernels and parameters...")
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
                print(f"Training SVM with kernel={kernel}, C={C}, degree={degree}, gamma={gamma}...")
                svm_builder = SVMModel(kernel=kernel, C=C, degree=degree, gamma=gamma)
                svm_model = svm_builder.build_model()
                svm_model.fit(X_train, y_train)

                # Evaluate on validation set
                val_accuracy = svm_model.score(X_val, y_val)
                print(f"Validation accuracy for SVM (kernel={kernel}, C={C}, degree={degree}, gamma={gamma}): {val_accuracy:.2f}")

                # Save the best model
                if val_accuracy > best_svm_accuracy:
                    best_svm_accuracy = val_accuracy
                    best_svm_model = svm_model
                    joblib.dump(svm_model, os.path.join(MODEL_DIR, f"best_svm_{kernel}_C{C}_degree{degree}_gamma{gamma}.pkl"))

                # Save all models for analysis
                joblib.dump(svm_model, os.path.join(MODEL_DIR, f"svm_{kernel}_C{C}_degree{degree}_gamma{gamma}.pkl"))

# Step 4: Evaluate the Best Models
print("Evaluating the best models...")

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

print("\nTraining and evaluation complete! All results are saved in the 'results' and 'models' directories.")
