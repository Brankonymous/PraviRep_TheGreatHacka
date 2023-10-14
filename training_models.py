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
    

name = "xgboost"
dataset_path = "data/"
X = pd.read_csv(dataset_path + "bg.csv")
y = pd.read_csv(dataset_path + "bg_target.csv")

X.drop(['Unnamed: 0', '0', '1', '2', '31', '60', '89', '118', '147', '176', '205', '234', '263'], axis=1, inplace=True) # IZBACIO SAM DATUME OVDE
y.drop(['Unnamed: 0'], axis=1, inplace=True)
y = y['0'] # SAMO JEDAN DAN PREDVIDJAMO

model = get_model(name)
pipeline = Pipeline([
            ('scaler', StandardScaler()),
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