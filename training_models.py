import xgboost as xgb
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV

def get_model(model_name):

    if model_name == 'logistic_regression':
        return LogisticRegression()
    
    elif model_name == 'knn':
        return KNeighborsClassifier()
    
    elif model_name == 'svm':
        return SVC()
    
    elif model_name == 'random_forest':
        return RandomForestClassifier()
    
    elif model_name == 'xgboost':
        return xgb.XGBClassifier()
    
    elif model_name == 'naive_bayes':
        return GaussianNB()
    
    else:
        raise ValueError(f'Unknown model name: {name}')
    

name = "xgboost"
dataset_path = ""
X = pd.read_csv(dataset_path + "train.csv")
y = pd.read_csv(dataset_path + "train_labels.csv")

model = get_model(name)
pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
            ])

param_grid = {"model__min_depth":[4, 5, 6]} # podesiti po zelji; prima i distribucije 
scoring = {
        'accuracy': make_scorer(accuracy_score)
}
clf = GridSearchCV(pipeline, param_grid, cv = None, scoring=scoring, n_iter=1, refit='accuracy', error_score="raise")

grid_search = clf.fit(X, y)

print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.cv_results_)