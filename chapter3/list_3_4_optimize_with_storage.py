import optuna
from binh_and_korn import objective

# optuna.dbには既にこのコードを実行した結果が格納されているので、実行の必要はありません。
study = optuna.create_study(
    study_name="ch3-multi-objective-example",
	storage="sqlite:///optuna.db",
    directions=["minimize", "minimize"],
	load_if_exists=True,
)

if len(study.trials) == 0:
	study.optimize(objective, n_trials=1000)
