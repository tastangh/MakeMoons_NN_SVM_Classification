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
                visualizer.plot_loss(
                    history=history,
                    learning_rate=lr,
                    epochs=epochs,
                    optimizer_name=optimizer_name,
                    hidden_layers=hidden_layers,
                    save_dir=LOSS_PLOTS_DIR
                )

                # Save decision boundary
                decision_boundary_path = os.path.join(
                    DECISION_PLOTS_DIR, 
                    f"ann_decision_boundary_LR{lr}_Epoch{epochs}_Opt{optimizer_name}_Layers{hidden_layers}.png"
                )
                visualizer.plot_decision_boundary(
                    model=ann_model,
                    X=X_val,
                    y=y_val,
                    save_path=decision_boundary_path,
                    model_type="ANN",
                    learning_rate=lr,
                    epochs=epochs,
                    optimizer_name=optimizer_name,
                    hidden_layers=hidden_layers
                )

                # Save model
                model_path = os.path.join(MODEL_DIR, f"ann_model_LR{lr}_Epoch{epochs}_Opt{optimizer_name}_Layers{hidden_layers}.keras")
                ann_model.save(model_path)