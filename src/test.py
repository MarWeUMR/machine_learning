import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


def load_data(dataset):

    print(f"got {dataset} from rust")

    df_train = pd.read_csv(f'datasets/{dataset}/train_data.csv')
    df_test = pd.read_csv(f'datasets/{dataset}/test_data.csv')

    # get categorical columns
    categorical_columns = df_train.select_dtypes(include=['object']).columns

    # iterate over categorical columns and apply encoding
    for col in categorical_columns:

        label_encoder = LabelEncoder()
        df_train[f"{col}"] = label_encoder.fit_transform(
            df_train[col])

        df_test[f"{col}"] = label_encoder.fit_transform(
            df_test[col])

    df_train.to_csv(
        f'datasets/{dataset}/train_data_enc.csv', index=False)
    df_test.to_csv(
        f'datasets/{dataset}/test_data_enc.csv', index=False)

    print("python done")
