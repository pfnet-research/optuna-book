import optuna

def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_float("y", -1, 1)
    return x * y

study = optuna.create_study()

# add_trialメソッドの場合には、追加時に探索空間の指定が必要となります
search_space = {
    "x": optuna.distributions.FloatDistribution(-1, 1),
    "y": optuna.distributions.FloatDistribution(-1, 1)
}

# x=0.5、y=-0.3の評価済みトライアルを追加します
study.add_trial(optuna.trial.create_trial(
    params={"x": 0.5, "y": -0.3},
    distributions=search_space,
    value=-0.15
))

# x=0.1、y=0.1の評価済みトライアルを間違った値で追加します
study.add_trial(optuna.trial.create_trial(
    params={"x": 0.1, "y": 0.1},
    distributions=search_space,
    value=0.0  # 本当は0.001になるべきですが、わざと不正な値を指定しています
))

# 最適化の実行し、結果を表示します
study.optimize(objective, n_trials=3)
for trial in study.trials:
    print(f"[{trial.number}] params={trial.params}, value={trial.value}")