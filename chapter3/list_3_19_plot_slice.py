import optuna

study = optuna.load_study(
    study_name="ch3-lightgbm-search-space-v1",
    storage="sqlite:///optuna.db",
)

optuna.visualization.plot_slice(
    study,
    params=["extra_trees", "feature_fraction"]
).show()
