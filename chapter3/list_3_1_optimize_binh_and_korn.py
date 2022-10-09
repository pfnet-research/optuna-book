import optuna

def f1(x, y):
    return 4 * x**2 + 4 * y**2

def f2(x, y):
    return (x - 5)**2 + (y - 5)**2

def objective(trial):
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    v1 = f1(x, y)
    v2 = f2(x, y)

    return v1, v2

study = optuna.create_study(
    directions=["minimize", "minimize"]
)

study.optimize(objective, n_trials=100)
