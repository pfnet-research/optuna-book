import optuna
from binh_and_korn import objective

study = optuna.create_study(
    study_name="ch3-multi-objective-example",
    storage="sqlite:///optuna.db",
    directions=["minimize", "minimize"]
)
study.optimize(objective, n_trials=1000)
