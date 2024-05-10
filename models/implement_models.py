import pickle, time
from sklearn import svm, tree, ensemble
from sklearn.model_selection import GridSearchCV
from typing import Union
import numpy as np
import numpy.typing as npt


def train_SVM(train_data: list | np.ndarray, targets: list, grid: bool) -> svm.LinearSVC:
    # train an SVM classifier using scikit-learn

    if grid:
        # use grid search to find the best parameter combination
        print('Using grid search to find the best parameter combination.')
        print('Paramter options: {"max_iter":[1000, 5000], "C":[0.001, 0.01, 0.1, 1, 10, 100], "loss": ("hinge", "squared_hinge")}')
        parameters = {'max_iter':[1000, 5000], 'C':[0.001, 0.01, 0.1, 1, 10, 100], 'loss': ('hinge', 'squared_hinge')}
        svc = svm.LinearSVC(dual="auto")
        lin_clf = GridSearchCV(svc, parameters, scoring='accuracy')
        lin_clf.fit(train_data, targets)

    else:
        # default hyperparameters: {max_iter=5000, regularizer(C)=10}
        lin_clf = svm.LinearSVC(dual="auto")
        lin_clf.fit(train_data, targets)


    return lin_clf


def train_DT(train_data: list | np.ndarray, targets: list, grid: bool) -> tree.DecisionTreeClassifier:
    # train an DT classifier using scikit-learn

    if grid:
        print('Using grid search to find the best parameter combination.')
        print('Paramter options: {"max_depth":[30, 50], "max_features":("sqrt", "log2", None)}')
        parameters = {'max_depth':[30, 50], 'max_features':('sqrt', 'log2', None)}
        dt = tree.DecisionTreeClassifier()
        dt_clf = GridSearchCV(dt, parameters, scoring='accuracy')
        dt_clf.fit(train_data, targets)
    else:
        # default hyperparameters: {max_depth=50, max_features=sqrt}
        dt_clf = tree.DecisionTreeClassifier()
        dt_clf.fit(train_data, targets)

    return dt_clf


def train_RF(train_data: list | np.ndarray, targets: list, grid: bool) -> ensemble.RandomForestClassifier:
    # train an RF classifier using scikit-learn

    if grid:
        print('Using grid search to find the best parameter combination.')
        print('Paramter options: {"max_depth":[30, 50], "max_features":("sqrt", "log2", None), "n_estimators":[50, 100]}')
        parameters = {'max_depth':[30, 50], 'max_features':('sqrt', 'log2', None), 'n_estimators':[50, 100]}
        rf =  ensemble.RandomForestClassifier()
        rf_clf = GridSearchCV(rf, parameters, scoring='accuracy')
        rf_clf.fit(train_data, targets)
    else:
        # default hyperparameters: {max_depth=50, max_features=sqrt, n_estimators=100}
        rf_clf = ensemble.RandomForestClassifier()
        rf_clf.fit(train_data, targets)

    return rf_clf


def save_model(model: Union[svm.LinearSVC, tree.DecisionTreeClassifier, ensemble.RandomForestClassifier, GridSearchCV], filename: str) -> None:

    # save the model to disk
    pickle.dump(model, open(filename, 'wb')) 


# def save_grid_search_object(grid: GridSearchCV, filename: str) -> None:
# 
#     # save the grid search model
#     joblib.dump(grid, filename)