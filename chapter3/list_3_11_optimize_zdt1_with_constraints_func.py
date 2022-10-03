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

    # [変更点１]
    # 「f1 の値が 0.5 以下」という制約を違反しているかどうかを、ユーザ属性に覚えておきます。
    trial.set_user_attr("constraints", [v1 - 0.5])

    return v1, v2

sampler = optuna.samplers.NSGAIISampler(
    crossover=optuna.samplers.nsgaii.UNDXCrossover(),

    # [変更点２]
    # サンプラーに制約を表現する関数を指定します。
    constraints_func=lambda trial: trial.user_attrs["constraints"]
)

study = optuna.create_study(
    study_name="ch3-zdt1-with-constraints",
    storage="sqlite:///optuna.db",
    directions=["minimize", "minimize"],
    sampler=sampler,
    load_if_exists=True,
)

if len(study.trials) == 0:
    study.optimize(objective, n_trials=1000)

optuna.visualization.plot_pareto_front(
    study,
    constraints_func=lambda trial: [trial.values[0] - 0.5]
).show()
