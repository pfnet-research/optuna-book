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

    # TODO: comment
    trial.set_user_attr('constraints', [v1 - 0.5])

    return v1, v2

sampler = optuna.samplers.NSGAIISampler(
    crossover="undx",

    # TODO: comment
    constraints_func=lambda trial: trial.user_attrs['constraints']
)

study = optuna.create_study(
    directions=["minimize", "minimize"]
    sampler=sampler
)

study.optimize(objective, n_trials=1000)
