import optuna
from binh_and_korn import objective

study = optuna.create_study(
    # TODO: comment
    study_name="ch4-multi-objective-example",
    storage="sqlite:///optuna.db",
    directions=["minimize", "minimize"]
)
study.optimize(objective, n_trials=1000)
