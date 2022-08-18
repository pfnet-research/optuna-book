import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# データのダウンロードと前処理
data = fetch_openml(name="adult")
X = pd.get_dummies(data["data"])
y = [1 if d == ">50K" else 0 for d in data["target"]]

# 機械学習モデルの初期化
clf = RandomForestClassifier(
    max_depth=8,  # ハイパーパラメータ
    min_samples_split=0.5,  # ハイパーパラメータ
)

# 交差検証による評価
score = cross_val_score(clf, X, y, cv=3)
accuracy = score.mean()
print(f"Accuracy: {accuracy}")

