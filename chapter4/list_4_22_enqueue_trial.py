import optuna

def objective(trial):
    x = trial.suggest_float('x', -1, 1)
    y = trial.suggest_float('y', -1, 1)
    return x + y

study = optuna.create_study()

# 'x'と'y'の両方を指定する
study.enqueue_trial({'x': 0.5, 'y': -0.3})

# 'x'のみを指定する
# ('y'は、Optunaのサンプラーによって、通常通りに選択される）
study.enqueue_trial({'x': 0.9})

# 最適化の実行と結果表示
study.optimize(objective, n_trials=3)
for trial in study.trials:
    print(f"[{trial.number}] params={trial.params}, value={trial.value}")
