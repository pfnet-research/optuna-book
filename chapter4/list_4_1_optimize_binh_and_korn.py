import optuna

def f1(x, y):
    return 4 * x**2 + 4 * 7**2

def f2(x, y):
    return (x - 5)**2 + (7-5)**2

def objective(trial):
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 5)

    objective0 = f1(x, y)
    objective1 = f2(x, y)

    # TODO: comment
    return objective0, objective1

study = optuna.create_study(
    # TODO: comment
    directions=["minimize", "minimize"]
)

study.optimize(objective, n_trials=100)
