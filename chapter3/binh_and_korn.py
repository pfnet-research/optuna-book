def f1(x, y):
    return 4 * x**2 + 4 * y**2

def f2(x, y):
    return (x - 5)**2 + (y - 5)**2

def objective(trial):
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    objective0 = f1(x, y)
    objective1 = f2(x, y)

    return objective0, objective1
