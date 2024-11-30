import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.constants import SEED


def process_data():
    df = pd.read_csv('data/training_data_fall2024.csv')

    ## 1. Convert to numerical
    processed_df = df.copy()

    # map the target column to 1/0 (if not already)
    if processed_df['increase_stock'].dtype is not np.int64:
        processed_df['increase_stock'] = processed_df['increase_stock'].map({"high_bike_demand":1, "low_bike_demand":0})

    
    ## 2. Create and drop features

    # Drop snow (it only has 0 values -> no information)
    processed_df.drop('snow', axis=1, inplace=True)

    # Add new (derived/composite) features
    # TODO


    ## 3. Shuffle and split
    X = processed_df.iloc[:, :-1]
    Y = processed_df.iloc[:, -1]

    X_train, X_test = train_test_split(X, test_size=0.15, random_state=SEED, shuffle=True)
    Y_train, Y_test = train_test_split(Y, test_size=0.15, random_state=SEED, shuffle=True)

    print('X train shape:', X_train.shape)
    print('Y train shape:', Y_train.shape)
    print('X test shape:', X_test.shape)
    print('Y test shape:', Y_test.shape)

    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    splits = process_data()
    print(splits[0].head())