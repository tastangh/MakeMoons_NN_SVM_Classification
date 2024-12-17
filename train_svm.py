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
                decision_boundary_path = os.path.join(
                    DECISION_PLOTS_DIR, 
                    f"svm_decision_boundary_{kernel}_C{C}_Degree{degree}_Gamma{gamma}.png"
                )
                visualizer.plot_decision_boundary(
                    model=svm_model,
                    X=X_val,
                    y=y_val,
                    save_path=decision_boundary_path,
                    model_type="SVM"
                )
                # Save model
                model_path = os.path.join(MODEL_DIR, f"svm_model_{kernel}_C{C}_Degree{degree}_Gamma{gamma}.pkl")
                joblib.dump(svm_model, model_path)
