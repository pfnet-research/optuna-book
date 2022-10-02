import optuna

def f1(x, y):
    return 4 * x**2 + 4 * y**2

def f2(x, y):
    return (x - 5)**2 + (y - 5)**2

def objective(trial):
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    objective0 = f1(x, y)
    objective1 = f2(x, y)

    # 単一目的最適化からの変更点１: 目的関数が複数の値を返します
    return objective0, objective1

study = optuna.create_study(
    # 変更点２: 目的毎に、最適化の方向を指定します
    directions=["minimize", "minimize"]
)

study.optimize(objective, n_trials=100)
