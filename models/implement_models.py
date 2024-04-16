from sklearn import svm, tree
import numpy as np
import numpy.typing as npt


def train_SVM(train_data: list | np.ndarray, targets: list) -> svm.LinearSVC:

    lin_clf = svm.LinearSVC(dual="auto")
    lin_clf.fit(train_data, targets)

    return lin_clf


def train_DT(train_data: list | np.ndarray, targets: list) -> tree.DecisionTreeClassifier:

    dt_clf = tree.DecisionTreeClassifier()
    dt_clf = dt_clf.fit(train_data, targets)

    return dt_clf


def train_RF(train_data: list | np.ndarray, targets: list) -> tree.DecisionTreeClassifier:

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    return dt_clf