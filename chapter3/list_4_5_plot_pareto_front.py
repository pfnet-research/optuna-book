import optuna

study = optuna.load_study(
    study_name="ch4-multi-objective-example",
    storage="sqlite:///optuna.db"
)

optuna.visualization.plot_pareto_front(study)
