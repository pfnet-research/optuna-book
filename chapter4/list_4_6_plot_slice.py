import optuna
from binh_and_korn import objective

study = optuna.load_study(
    study_name="ch4-multi-objective-example",
    storage="sqlite:///optuna.db"
)

# TODO: comment
optuna.visualization.plot_slice(
    study,
    target=lambda t: t.values[0]
).show()

# TODO: comment
optuna.visualization.plot_slice(
    study,
    target=lambda t: t.values[1]
).show()
