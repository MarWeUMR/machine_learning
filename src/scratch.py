import pandas as pd

dataset = "landcover"

df_train = pd.read_csv('datasets/{}/train_data.csv'.format(dataset))
df_test = pd.read_csv('datasets/{}/test_data.csv'.format(dataset))

df_train.drop(columns=['Landuse'], inplace=True)
df_test.drop(columns=['Landuse'], inplace=True)


df_train.to_csv('datasets/{}/train_data.csv'.format(dataset), index=False)
df_test.to_csv('datasets/{}/test_data.csv'.format(dataset), index=False)


print(df_test)
