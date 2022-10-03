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

    # 制約が満たされない場合には、トライアルを枝刈りしてしまいます
    if v1 > 0.5:
        raise optuna.TrialPruned(f"Too large value: {v1}")

    v2 = f2(X)

    # 執筆時点では、多目的最適化での枝刈りは未対応のため、
    # 「v1が0.5未満という制約下でv2を最小化する」という
    # 単一目的最適化として扱っています
    return v2

sampler = optuna.samplers.NSGAIISampler(
    crossover=optuna.samplers.nsgaii.UNDXCrossover(),
)

study = optuna.create_study(sampler=sampler)

study.optimize(objective, n_trials=1000)
