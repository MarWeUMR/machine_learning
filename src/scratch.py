import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)


# X = fetch_openml("titanic", version=1, as_frame=True, return_X_y=False)
X = fetch_openml("titanic", version=1, as_frame=True)

df = pd.DataFrame(data=X.data, columns=X.feature_names)

df["target"] = X.target

# X.drop(["boat", "body", "home.dest"], axis=1, inplace=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# boston_dataset = load_breast_cancer()

# boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
# boston["target"] = boston_dataset.target

print(df)

df.to_csv(f"datasets/titanic/data.csv", index=False, sep=",", header=True)
df.to_csv(f"datasets/titanic/data_xg.csv", index=False, sep=" ", header=False)
