import optuna

# 最適化結果をロード
study = optuna.load_study(
    storage="sqlite:///optuna.db",
    study_name="ch4-lightgbm-random-v1"
)

# 各ハイパーパラメータの重要度を可視化
optuna.visualization.plot_param_importances(study).show()
