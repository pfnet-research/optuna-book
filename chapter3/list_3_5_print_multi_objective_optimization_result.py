import optuna
from binh_and_korn import objective

study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=100)

print("[Best Trials]")

# 変更点３: Study.best_trial の代わりに Study.best_trials を使用します
for trial in study.best_trials:
    # 変更点４: FrozenTrial.value の代わりに FrozenTrial.values を使用します
    print(f"- [{trial.number}] params={trial.params}, values={trial.values}")
