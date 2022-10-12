import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

data, target = sklearn.datasets.fetch_covtype(return_X_y=True)
target = list(map(lambda label: int(label) - 1, target))
train_x, valid_x, train_y, valid_y = train_test_split(
    data,
    target,
    test_size=0.25,
    random_state=0
)
dtrain = lgb.Dataset(train_x, label=train_y)

params = {
    "verbosity": -1,
}
gbm = lgb.train(params, dtrain)

preds = gbm.predict(valid_x)
pred_labels = np.rint(preds)
accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)

print(f"Accuracy: {accuracy}")
