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
