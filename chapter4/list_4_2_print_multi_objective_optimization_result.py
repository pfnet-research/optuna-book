import optuna
from binh_and_korn import objective

study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=100)

print("[Best Trials]")

# TODO: comment
for trial in study.best_trials:
    # TODO: comment
    print("- [{}] params={}, values={}".format(
        trial.number,
        trial.params,
        trial.values,
    ))
