import optuna

def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)

    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

study = optuna.create_study()

study.enqueue_trial({"x": 2.8, "y": 2.2})  # 評価値は1.3312
study.enqueue_trial({"x": 3.1, "y": 1.7})  # 評価値は1.1161

study.optimize(objective, n_trials=100)

optuna.visualization.plot_contour(study).show()
