import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)


# (X, y) = fetch_openml("iris", version=1, as_frame=True, return_X_y=True)
# X = fetch_openml("iris", version=1, as_frame=True)

# df = pd.DataFrame(data=X.data, columns=X.feature_names)
df = pd.read_csv("datasets/urban/test_data.csv")
# df["target"] = X.target

# X.drop(["boat", "body", "home.dest"], axis=1, inplace=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# boston_dataset = load_breast_cancer()

# boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
# boston["target"] = boston_dataset.target

print(df)

df.to_csv(f"datasets/iris/data.csv", index=False, sep=",", header=True)
df.to_csv(f"datasets/iris/data_xg.csv", index=False, sep=" ", header=False)
