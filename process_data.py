import numpy as np
import pandas as pd
from utils.constants import SEED


def __split(df:pd.DataFrame, prec:dict, info=False):
    # setting and checking parameters
    if sum(prec.values()) != 1.0:
        raise ValueError(f'Split precentages do not add up to 1.0! Precentages: {prec.values()}')
    # shuffling
    data = df.sample(frac=1, random_state=SEED, ignore_index=True)
    
    splits = []
    split_start_index = 0
    data_size = data.shape[0]
    for split_name, p in prec.items():
        # calculate current split's end index
        split_size = data_size * p
        split_end_index = int(split_start_index + split_size if split_start_index + split_size < data_size else data_size)
        
        if info: print(f'{split_name} -> range=[{split_start_index}-{split_end_index-1}] size={split_end_index-split_start_index}')
        
        # get data for current split
        split_data = data.iloc[split_start_index:split_end_index]
        splits.append(split_data)

        # set the next split's start
        split_start_index = split_end_index
    return splits


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
    prec = {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    }
    splits = __split(df, prec=prec, info=True)

    # separate the target and the data
    X_train, Y_train = splits[0].iloc[:,:-1], splits[0].iloc[:,-1]
    X_val, Y_val = splits[1].iloc[:,:-1], splits[1].iloc[:,-1]
    X_test, Y_test = splits[2].iloc[:,:-1], splits[2].iloc[:,-1]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


if __name__ == '__main__':
    splits = process_data()
    print(splits[0].head())


    