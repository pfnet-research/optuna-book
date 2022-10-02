import math
import optuna

def f1(X):
    return X[0]

def f2(X):
    g = 1 + 9 * sum(X[1:]) / (len(X) - 1)
    h = 1 - math.sqrt(X[0] / g)
    return g * h

def objective(trial):
    X = [trial.suggest_float(f"x{i}", 0, 1) for i in range(30)]
    v1 = f1(X)
    v2 = f2(X)
    return v1, v2

sampler = optuna.samplers.NSGAIISampler(
    # 制約付き最適化とは直接関係ないですが、
    # デフォルトよりも UNDX というクロスオーバ手法の方が、
    # 可視化結果が分かりやすいので、今回はこちらを指定しています
    crossover="undx"
)

study = optuna.create_study(
    directions=["minimize", "minimize"]
    sampler=sampler
)

study.optimize(objective, n_trials=1000)
