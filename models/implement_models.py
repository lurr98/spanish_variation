#!/usr/bin/env python3
"""
Author: Laura Zeidler
Last changed: 14.08.2024

This module provides functions to train and save machine learning models using scikit-learn, 
focusing on SVM and Decision Tree classifiers. Additionally, it offers functionality for 
hyperparameter tuning using GridSearchCV and model serialization with pickle.

"""

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
        print('Paramter options: {"max_iter":[1000], "C":[0.001, 0.01, 0.1, 10]}')
        parameters = {'C':[0.001, 0.01, 0.1, 10]}
        # parameters = {'C':[0.001, 10]}
        svc = svm.LinearSVC(dual="auto", max_iter=1000)
        lin_clf = GridSearchCV(svc, parameters, scoring='accuracy', n_jobs=-1, verbose=2)
        lin_clf.fit(train_data, targets)

    else:
        # default hyperparameters for count features: regularizer(C)=0.001
        # default hyperparameters for tf features: regularizer(C)=10
        lin_clf = svm.LinearSVC(dual="auto", C=10, max_iter=1000)
        # lin_clf = svm.LinearSVC(dual="auto", C=0.001, max_iter=1000)
        lin_clf.fit(train_data, targets)


    return lin_clf


def train_DT(train_data: list | np.ndarray, targets: list, grid: bool) -> tree.DecisionTreeClassifier:
    # train an DT classifier using scikit-learn

    if grid:
        print('Using grid search to find the best parameter combination.')
        print('Paramter options: {"max_depth":[30, 50], "max_features":("sqrt", "log2", None)}')
        parameters = {'max_depth':[30, 50], 'max_features':('sqrt', 'log2', None)}
        dt = tree.DecisionTreeClassifier()
        dt_clf = GridSearchCV(dt, parameters, scoring='accuracy', n_jobs=-1, verbose=2)
        dt_clf.fit(train_data, targets)
    else:
        # default hyperparameters: {max_depth=50, max_features=None}
        dt_clf = tree.DecisionTreeClassifier(max_depth=50, max_features=None)
        dt_clf.fit(train_data, targets)

    return dt_clf


def save_model(model: Union[svm.LinearSVC, tree.DecisionTreeClassifier, ensemble.RandomForestClassifier, GridSearchCV], filename: str) -> None:

    # save the model to disk
    pickle.dump(model, open(filename, 'wb')) 
