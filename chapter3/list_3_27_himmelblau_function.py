import optuna

def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)

    # Himmelblau関数
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

study = optuna.create_study()

# x=3、y=2付近の探索点を追加し、最適化を実施します
study.enqueue_trial({"x": 2.8, "y": 2.2})  # 評価値は1.3312
study.enqueue_trial({"x": 3.1, "y": 1.7})  # 評価値は1.1161

study.optimize(objective, n_trials=100)

# 最適化結果を可視化します
optuna.visualization.plot_contour(study).show()
