import optuna
from binh_and_korn import objective

study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=100)

print("[Best Trials]")

# 変更点３: Study.best_trial の代わりに Study.best_trials を使用する
for trial in study.best_trials:
    # 変更点４: FrozenTrial.value の代わりに FrozenTrial.values を使用する
    print("- [{}] params={}, values={}".format(
        trial.number,
        trial.params,
        trial.values,
    ))
