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
    
def handle_dates(df, keep_date=False):
    date_columns = ['2', '31', '60', '89', '118', '147', '176', '205', '234', '263']

    # for col in date_columns:
    #     dates = list(df[col])
    #     df['prophet_' + col] = get_pred_from_prophet(dates)

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

name = "xgboost"
dataset_path = "data/"
X = pd.read_csv(dataset_path + "bg.csv")
y = pd.read_csv(dataset_path + "bg_target.csv")

X.drop(['Unnamed: 0', '0', '1'], axis=1, inplace=True) # IZBACIO SAM DATUME OVDE
X = handle_dates(X, keep_date=True)
y.drop(['Unnamed: 0'], axis=1, inplace=True)
y = y['0'] # SAMO JEDAN DAN PREDVIDJAMO

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