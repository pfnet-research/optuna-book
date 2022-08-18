import optuna


study = optuna.load_study(
    storage="sqlite:///optuna-storage.db",
    study_name="chapter2-conditional",
)

print(f"Best objective value: {study.best_value}")
print(f"Best parameter: {study.best_params}")

