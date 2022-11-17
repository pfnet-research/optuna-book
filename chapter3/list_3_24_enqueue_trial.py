import optuna

def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_float("y", -1, 1)
    return x * y

study = optuna.create_study()

study.enqueue_trial({"x": 0.5, "y": -0.3})
study.enqueue_trial({"x": 0.9})

study.optimize(objective, n_trials=3)
for trial in study.trials:
    print(f"[{trial.number}] params={trial.params}, value={trial.value}")
