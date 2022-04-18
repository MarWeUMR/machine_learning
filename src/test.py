import pandas as pd

from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv('datasets/urban/train_data.csv')
df_test = pd.read_csv('datasets/urban/test_data.csv')

label_encoder = LabelEncoder()
df_train["class_enc"] = label_encoder.fit_transform(df_train["class"])
df_train.drop(["class"], axis=1, inplace=True)
df_train.to_csv('datasets/urban/train_data_enc.csv')

label_encoder = LabelEncoder()
df_test["class_enc"] = label_encoder.fit_transform(df_test["class"])
df_test.drop(["class"], axis=1, inplace=True)
df_test.to_csv('datasets/urban/test_data_enc.csv')


print("python done")
