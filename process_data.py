import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
import sklearn.preprocessing as skl_pre
from utils.constants import SEED

#an artifical score of if the weather is good or not
def get_good_weather_score(df:pd.DataFrame):
    scores = []
    #look at every row in the dataframe
    for (i,v) in df.iterrows():
        #if the snowdepth is greater than 0 we know demand is low from the training data
        if v["snowdepth"] > 0:
            scores.append(0)
        else:
            #score is a combination of if it is summertime
            #a factor of the temperature
            #a factor of the dewpoint
            #an inverse factor of humidity
            #an inverse factor of windspeed
            num = v["summertime"] + v["temp"]/27 + v["dew"]/15 + 1/(v['humidity']+20) + 1/(v["windspeed"]+10)
            scores.append(num)
    return np.array(scores)

#returns 1 if it is day time and 0 if it is nighttime
#rough estimate
def get_is_day(df:pd.DataFrame):
    days = []
    #loop through every data point
    for (i,v) in df.iterrows():
        # if the time is between 8 and 18 we are in daytime
        if v["hour_of_day"] > 8 and v["hour_of_day"] < 18:
            days.append(1)
        #otherwise it is nighttime
        else:
            days.append(0)
    return days

def create_new_features(df:pd.DataFrame, info=False, dropped_columns = ["snow"]) -> pd.DataFrame:
    '''Drops and adds/creates new features'''

    extended_df = df.copy()
    original_columns = df.columns

    # drop snow (it only has 0 values -> no information)
    for column in dropped_columns:
        extended_df.drop(column, axis=1, inplace=True)

    # add new (derived/composite) features
    # TODO
    # e.g. create separate columns for each day of the week
    
    #making column for Farenheit since it is the superior temperature measure
    extended_df["temp_fahrenheit"] = round((extended_df["temp"] * 9/5) + 32)
    extended_df["good_weather"] = get_good_weather_score(extended_df)
    extended_df["is_day"] = get_is_day(extended_df)
    extended_columns = extended_df.columns
    if info:
        dropped_columns = np.setdiff1d(original_columns, extended_columns)
        new_columns = np.setdiff1d(extended_columns, original_columns)
        print('Dropped columns:', dropped_columns)
        print('New columns:', new_columns)
    return extended_df


def create_splits(df:pd.DataFrame, split_prec:dict, info=False, is_random = False) -> list[pd.DataFrame]:
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
    if is_random == False:
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
    else:
        if len(split_prec.keys()) == 3:         # train, valid, test
            corrected_valid_prec = split_prec['valid'] / (1-split_prec['train'])
            X_train, remainder = train_test_split(X, train_size=split_prec['train'], shuffle=True)
            X_valid, X_test = train_test_split(remainder, train_size=corrected_valid_prec, shuffle=True)
            Y_train, remainder = train_test_split(Y, train_size=split_prec['train'], shuffle=True)
            Y_valid, Y_test = train_test_split(remainder, train_size=corrected_valid_prec, shuffle=True)
            splits = [X_train, X_valid, X_test, Y_train, Y_valid, Y_test]
        elif len(split_prec.keys()) == 2:       # train, test
            X_train, X_test = train_test_split(X, train_size=split_prec['train'], shuffle=True)
            Y_train, Y_test = train_test_split(Y, train_size=split_prec['train'], shuffle=True)
            splits = [X_train, X_test, Y_train, Y_test]
        elif len(split_prec.keys()) == 1:       # train
            X_train = X.sample(frac=1.0, replace=False)
            Y_train = Y.sample(frac=1.0, replace=False)
            splits = [X_train, Y_train]

    # print some info about the splits
    if info:
        X_Y_split_point = int(len(splits)/2)
        for split_name, X_split, Y_split in zip(split_prec.keys(), splits[:X_Y_split_point], splits[X_Y_split_point:]):
            print(f'Split: "{split_name}" \t[Size: {X_split.shape[0]}] \t[Prec: {X_split.shape[0]/df.shape[0]}]')
            print(f'\tX: {X_split.shape}')
            print(f'\tY: {Y_split.shape}')

    return splits


def process_data(split_prec: dict, scaler = None,dropped_columns = ["snow"], is_random = False):
    # find the project directory and load the data
    project_dir = os.path.abspath(os.path.join('..'))
    data_path = os.path.join(project_dir,'data', 'training_data_fall2024.csv')
    df = pd.read_csv(data_path)

    ## 1. Convert to numerical
    processed_df = df.copy()
    # map the target column to 1/0 (if not already)
    if processed_df['increase_stock'].dtype is not np.int64:
        processed_df['increase_stock'] = processed_df['increase_stock'].map({"high_bike_demand":1, "low_bike_demand":0})
    
    ## 2. Create and drop features
    processed_df = create_new_features(processed_df, info=True,dropped_columns=dropped_columns)
    #print(processed_df)
    x_cols = processed_df.columns.tolist()
    x_cols.remove("increase_stock")

    ## 3. Shuffle and split
    splits = create_splits(processed_df, split_prec, info=True, is_random=is_random)
    # print(splits)
    #if the scaler needs to be fitted to the data that is done here
    #the scaler is fitted to the training data and then the validation and testing X data is fitted 
    #with the scaler that the training data was fitted with
    if type(scaler) == skl_pre._data.StandardScaler or type(scaler) == skl_pre._data.MinMaxScaler:
        scaler = scaler.fit(splits[0])
        splits[0] = pd.DataFrame(scaler.transform(splits[0]),columns=x_cols)
        if len(splits) >= 4:
            splits[1] = pd.DataFrame(scaler.transform(splits[1]),columns=x_cols)
        if len(splits) > 4:
            splits[2] = pd.DataFrame(scaler.transform(splits[2]),columns=x_cols)
    # print(splits)

    return splits


if __name__ == '__main__':
    splits = process_data()
    print(splits[0].head())