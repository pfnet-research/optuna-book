import optuna
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


data = fetch_openml(name="adult")
X = pd.get_dummies(data["data"])
y = [1 if d == ">50K" else 0 for d in data["target"]]


def objective(trial):
    clf_name = trial.suggest_categorical("clf", ("RF", "GB"))

    # clf_name の値によってハイパーパラメータを分岐させる
    if clf_name == "RF":
        clf = RandomForestClassifier(
            max_depth=trial.suggest_int(
                "rf_max_depth", 2, 32,
            ),
            min_samples_split=trial.suggest_float(
                "rf_min_samples_split", 0, 1,
            ),
        )
    else:
        clf = GradientBoostingClassifier(
            max_depth=trial.suggest_int(
                "gb_max_depth", 2, 32,
            ),
            min_samples_split=trial.suggest_float(
                "gb_min_samples_split", 0, 1,
            ),
        )

    score = cross_val_score(clf, X, y, cv=3)
    accuracy = score.mean()
    return accuracy


# study_name と storage を指定することで study を保存できる
study = optuna.create_study(
    direction="maximize",
    study_name="ch2-conditional",
    storage="sqlite:///optuna.db",
)
study.optimize(objective, n_trials=100)

print(f"Best objective value: {study.best_value}")
print(f"Best parameter: {study.best_params}")

