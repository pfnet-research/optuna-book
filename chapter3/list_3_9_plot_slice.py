import optuna

study = optuna.load_study(
    study_name="ch3-multi-objective-example",
    storage="sqlite:///optuna.db"
)

# f1 関数の結果を使って可視化します
optuna.visualization.plot_slice(
    study,
    target=lambda t: t.values[0]
).show()

# f2 関数の結果を使って可視化します
optuna.visualization.plot_slice(
    study,
    target=lambda t: t.values[1]
).show()
