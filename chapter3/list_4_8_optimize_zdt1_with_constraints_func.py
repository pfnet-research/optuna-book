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
    # constraints の値が 0 より大きい場合には、制約を違反していることを表しています。
    # また、値が大きいほど、違反度合いも大きいと判断されます。
    #
    # なお、制約は複数個指定可能ですが、今回は一つだけ指定しています。
    trial.set_user_attr('constraints', [v1 - 0.5])

    return v1, v2

sampler = optuna.samplers.NSGAIISampler(
    crossover="undx",

    # [変更点２]
    # サンプラーに制約を表現する関数を指定します。
    # 通常は、目的関数の中で設定した値を単に返すだけで十分です。
    constraints_func=lambda trial: trial.user_attrs['constraints']
)

study = optuna.create_study(
    directions=["minimize", "minimize"]
    sampler=sampler
)

study.optimize(objective, n_trials=1000)
