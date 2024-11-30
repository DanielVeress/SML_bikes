import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.constants import SEED


def create_new_features(df:pd.DataFrame, info=False) -> pd.DataFrame:
    '''Drops and adds/creates new features'''

    extended_df = df.copy()
    original_columns = df.columns

    # drop snow (it only has 0 values -> no information)
    extended_df.drop('snow', axis=1, inplace=True)

    # add new (derived/composite) features
    # TODO
    # e.g. create separate columns for each day of the week
    
    extended_columns = extended_df.columns
    if info:
        dropped_columns = np.setdiff1d(original_columns, extended_columns)
        new_columns = np.setdiff1d(extended_columns, original_columns)
        print('Dropped columns:', dropped_columns)
        print('New columns:', new_columns)
    return extended_df


def create_splits(df:pd.DataFrame, split_prec:dict, info=False) -> list[pd.DataFrame]:
    '''Creates splits from a dataframe'''

    # check if they add up to 1
    if sum(split_prec.values()) != 1.0:
        raise ValueError(f'The split precentages do not add up to 1! ({split_prec.values()}={sum(split_prec.values())})')
    # check the split names
    for split_name in split_prec.keys():
        correct_split_names = ['train', 'valid', 'test']
        if split_name not in correct_split_names:
            raise ValueError(f'The split name "{split_name}" is not correct! Only the following values are accepted: {correct_split_names}')

    X = df.iloc[:, df.columns!='increase_stock']
    Y = df.loc[:, 'increase_stock']

    # split into sets
    splits = None
    if len(split_prec.keys()) == 3:         # train, valid, test
        corrected_valid_prec = split_prec['valid'] / (1-split_prec['train'])
        X_train, remainder = train_test_split(X, train_size=split_prec['train'], random_state=SEED, shuffle=True)
        X_valid, X_test = train_test_split(remainder, train_size=corrected_valid_prec, random_state=SEED, shuffle=True)
        Y_train, remainder = train_test_split(Y, train_size=split_prec['train'], random_state=SEED, shuffle=True)
        Y_valid, Y_test = train_test_split(remainder, train_size=corrected_valid_prec, random_state=SEED, shuffle=True)
        splits = [X_train, X_valid, X_test, Y_train, Y_valid, Y_test]
    elif len(split_prec.keys()) == 2:       # train, test
        X_train, X_test = train_test_split(X, train_size=split_prec['train'], random_state=SEED, shuffle=True)
        Y_train, Y_test = train_test_split(Y, train_size=split_prec['train'], random_state=SEED, shuffle=True)
        splits = [X_train, X_test, Y_train, Y_test]
    elif len(split_prec.keys()) == 1:       # train
        X_train = X.sample(frac=1.0, replace=False, random_state=SEED)
        Y_train = Y.sample(frac=1.0, replace=False, random_state=SEED)
        splits = [X_train, Y_train]

    # print some info about the splits
    if info:
        X_Y_split_point = int(len(splits)/2)
        for split_name, X_split, Y_split in zip(split_prec.keys(), splits[:X_Y_split_point], splits[X_Y_split_point:]):
            print(f'Split: "{split_name}" \t[Size: {X_split.shape[0]}] \t[Prec: {X_split.shape[0]/df.shape[0]}]')
            print(f'\tX: {X_split.shape}')
            print(f'\tY: {Y_split.shape}')

    return splits


def process_data():
    # find the project directory and load the data
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_dir, 'data', 'training_data_fall2024.csv')
    df = pd.read_csv(data_path)

    ## 1. Convert to numerical
    processed_df = df.copy()
    # map the target column to 1/0 (if not already)
    if processed_df['increase_stock'].dtype is not np.int64:
        processed_df['increase_stock'] = processed_df['increase_stock'].map({"high_bike_demand":1, "low_bike_demand":0})

    
    ## 2. Create and drop features
    processed_df = create_new_features(processed_df, info=True)

    ## 3. Shuffle and split
    split_prec = {
        'train': 0.7, 
        'valid': 0.15, 
        'test': 0.15
    }
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = create_splits(processed_df, split_prec, info=True)
    #X_train, X_test, Y_train, Y_test = create_splits(processed_df, split_prec, info=True)
    #X_train, Y_train = create_splits(processed_df, split_prec, info=True)
    
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test
    #return X_train, X_test, Y_train, Y_test
    #return X_train, Y_train


if __name__ == '__main__':
    splits = process_data()
    print(splits[0].head())