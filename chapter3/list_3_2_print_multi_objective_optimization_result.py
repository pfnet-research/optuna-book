import optuna
from binh_and_korn import objective

study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=100)

print("[Best Trials]")

for trial in study.best_trials:
    print(f"- [{trial.number}] params={trial.params}, values={trial.values}")
