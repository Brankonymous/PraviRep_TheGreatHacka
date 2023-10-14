import pandas as pd
import numpy as np

def reshape_data(df, target = 'AMBROSIA', stride = 1, train = True):
    df = df.drop(['location', 'Unnamed: 0'], axis = 1)
    upper = 14
    if not train:
       upper = 10
    X = []
    y = []
    for i in range(0, df.shape[0] - (upper - 1), stride):
        lst = []

        ran = range(i, i + 10)
        X.append(np.array(df.iloc[ran]).flatten())
        if train:
            y.append((
                float(df.iloc[[i+11]][target]),
                float(df.iloc[[i+12]][target]),
                float(df.iloc[[i+13]][target]),
            ))
    return X, y

def load_train_test(train_path, test_path, weather_path = None, stride = 1, target = 'AMBROSIA'):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_test = df_test.drop('batch_id', axis = 1)
    
    if weather_path:
        df_weather = pd.read_csv(weather_path)
        df_train.merge(right = df_weather, how = 'left', on = ['location', 'date'])
        df_train = df_train.dropna()

        

    locations = ['БЕОГРАД - НОВИ БЕОГРАД', 'ВРШАЦ', 'НИШ', 'ПОЖАРЕВАЦ', 'СУБОТИЦА','КРАГУЈЕВАЦ', 'КРАЉЕВО']

    train_data = {}
    train_targets = {}
    test_data = {}

    for location in locations:
        train_data[location] = df_train[df_train['location'] == location]
        train_data[location], train_targets[location] = reshape_data(train_data[location], target, stride)
        test_data[location] = df_test[df_test['location'] == location]
        test_data[location], _ = reshape_data(test_data[location], target = target, stride = 10, train = False)
    
    return train_data, train_targets, test_data

if __name__ == "__main__":
    train_path = './data/pollen_train.csv'
    test_path = './data/pollen_test.csv'
    dataset_path = './data/'

    X_train, y, X_test = load_train_test(train_path, test_path)
    print(X_train['ВРШАЦ'][0])