import pandas as pd
from sklearn.datasets import load_breast_cancer

boston_dataset = load_breast_cancer()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston["target"] = boston_dataset.target

print(boston)

boston.to_csv(f"datasets/cancer/data.csv", index=False)
