import optuna

study = optuna.load_study(
    study_name="ch3-multi-objective-example",
    storage="sqlite:///optuna.db"
)

# すべてのトライアルをプロット（デフォルト挙動）
optuna.visualization.plot_pareto_front(
	study,
	include_dominated_trials=True
).show()

# Study.best_trials だけをプロット
optuna.visualization.plot_pareto_front(
	study,
	include_dominated_trials=False
).show()

