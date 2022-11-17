import optuna

study = optuna.load_study(
    study_name="ch3-multi-objective-example",
    storage="sqlite:///optuna.db"
)

optuna.visualization.plot_slice(
    study,
    target=lambda t: t.values[0],
    target_name="Objective value 0",
).show()

optuna.visualization.plot_slice(
    study,
    target=lambda t: t.values[1],
    target_name="Objective value 1",
).show()
