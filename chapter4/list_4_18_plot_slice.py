import optuna

study = optuna.load_study(
    storage="sqlite:///optuna.db",
    study_name="ch4-lightgbm-random-v1"
)

optuna.visualization.plot_slice(
    study,
    ["extra_trees", "feature_fraction"]
).show()
