import xgboost as xgb
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_to_json, model_from_json

from utils import load_train_test
from datetime import datetime, timedelta


def get_model(model_name):

    if model_name == 'logistic_regression':
        return LogisticRegression()
    
    elif model_name == 'svm':
        return SVC()
    
    elif model_name == 'random_forest':
        return RandomForestClassifier()
    
    elif model_name == 'xgboost':
        return xgb.XGBRegressor()
    
    elif model_name == 'naive_bayes':
        return GaussianNB()
    
    else:
        raise ValueError(f'Unknown model name: {name}')
    
def get_pred_from_prophet(dates):
    dataset_path = "data/"

    with open('models/prophet.json', 'r') as fin:
        m = model_from_json(fin.read())  # Load model

    future = pd.DataFrame({'ds': dates})
    future['floor'] = 0
    future['cap'] = 2500

    forecast = m.predict(future)

    ans = []
    for forecast_row in forecast.itertuples():
        val = int(forecast_row.yhat)
        val = max(0, val)
        ans.append(val)

    return ans

def add_prophet_features(df):
    date_columns = [i*27 for i in range(0, 10)]

    last_dates = list(df[date_columns[-1]])
    last_dates = [datetime.strptime(last_date, '%Y-%m-%d') for last_date in last_dates]

    day11 = [last_date + timedelta(days=1) for last_date in last_dates]
    day11 = [d.strftime('%Y-%m-%d') for d in day11]

    day12 = [last_date + timedelta(days=2) for last_date in last_dates]
    day12 = [d.strftime('%Y-%m-%d') for d in day12]

    day13 = [last_date + timedelta(days=3) for last_date in last_dates]
    day13 = [d.strftime('%Y-%m-%d') for d in day13]

    df['prophet_1'] = get_pred_from_prophet(day11)
    df['prophet_2'] = get_pred_from_prophet(day12)
    df['prophet_3'] = get_pred_from_prophet(day13)

    return df
    
def handle_dates(df, keep_date=False):
    date_columns = [i*27 for i in range(0, 10)]
    if keep_date:
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
            # df['day'] = df[col].dt.day
            df['month'] = df[col].dt.month
            # df['year'] = df['date'].dt.year
            df.drop([col], axis=1, inplace=True)
    else:
        for col in date_columns:
            df.drop([col], axis=1, inplace=True)

    return df

def return_data_location(location):
    X_train_loc = pd.DataFrame(X_train[location])
    y_loc = y[location]
    X_test_loc = pd.DataFrame(X_test[location])

    return X_train_loc, y_loc, X_test_loc

def return_targets_day(y_loc, day):
    y_loc_day = [x[day] for x in y_loc]
    
    return y_loc_day


name = "xgboost"
dataset_path = "data/"

X_train, y, X_test = load_train_test(dataset_path + "pollen_train.csv", dataset_path + "pollen_test.csv")
X_train_loc, y_loc, X_test_loc = return_data_location('БЕОГРАД - НОВИ БЕОГРАД')
X_train_loc = add_prophet_features(X_train_loc)
X_test_loc = add_prophet_features(X_test_loc)

X_train_loc = handle_dates(X_train_loc)
X_test_loc = handle_dates(X_test_loc)

X = X_train_loc
y = return_targets_day(y_loc, 0)

model = get_model(name)
pipeline = Pipeline([
            ('model', model)
            ])

param_grid = {} # podesiti po zelji; prima i distribucije 
scoring = {
        'mse': make_scorer(mean_squared_error),
        'mae': make_scorer(mean_absolute_error),
}
clf = GridSearchCV(pipeline, param_grid, cv = 10, scoring=scoring, refit='mae', error_score="raise")

grid_search = clf.fit(X, y)

print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.cv_results_)