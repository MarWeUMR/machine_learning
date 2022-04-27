import pandas as pd
import numpy as np
import os.path
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import LabelEncoder


def load_data(dataset):

    # check if files are present
    train_file = os.path.isfile(f"datasets/{dataset}/train_data.csv")
    test_file = os.path.isfile(f"datasets/{dataset}/test_data.csv")

    # generate split dataset for training and testing if files are not yet present
    if (train_file and test_file) is False:

        print("generating files...")
        data_frame = pd.read_csv(f"datasets/{dataset}/data.csv")
        target_col = data_frame.iloc[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(
            data_frame, target_col, test_size=0.2, random_state=5
        )

        x_train.to_csv(f"datasets/{dataset}/train_data.csv", index=False)

        x_test.to_csv(
            f"datasets/{dataset}/test_data.csv",
            index=False,
        )

    # load data and preprocess
    df_train = pd.read_csv(f"datasets/{dataset}/train_data.csv")
    df_test = pd.read_csv(f"datasets/{dataset}/test_data.csv")

    # get categorical columns
    categorical_columns = df_train.select_dtypes(include=["object"]).columns

    # iterate over categorical columns and apply encoding
    for col in categorical_columns:

        label_encoder = LabelEncoder()
        df_train[f"{col}"] = label_encoder.fit_transform(df_train[col])

        df_test[f"{col}"] = label_encoder.fit_transform(df_test[col])

    df_train.to_csv(f"datasets/{dataset}/train_data_enc.csv", index=False)
    df_test.to_csv(f"datasets/{dataset}/test_data_enc.csv", index=False)
