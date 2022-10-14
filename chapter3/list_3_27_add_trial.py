import optuna

def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_float("y", -1, 1)
    return x * y

study = optuna.create_study()

search_space = {
    "x": optuna.distributions.FloatDistribution(-1, 1),
    "y": optuna.distributions.FloatDistribution(-1, 1)
}

study.add_trial(optuna.trial.create_trial(
    params={"x": 0.5, "y": -0.3},
    distributions=search_space,
    value=-0.15
))

study.add_trial(optuna.trial.create_trial(
    params={"x": 0.1, "y": 0.1},
    distributions=search_space,
    value=0.0
))

study.optimize(objective, n_trials=3)
for trial in study.trials:
    print(f"[{trial.number}] params={trial.params}, value={trial.value}")
