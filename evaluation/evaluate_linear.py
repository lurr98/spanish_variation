import pickle
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
from sklearn import svm, tree, ensemble, metrics
from sklearn.model_selection import GridSearchCV
from scipy.sparse import spmatrix
from typing import Union


def load_linear_model(model_path: str) -> Union[svm.LinearSVC, tree.DecisionTreeClassifier, ensemble.RandomForestClassifier]:

    # load classifier
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

    return classifier


def predict_labels(classifier: Union[svm.LinearSVC, tree.DecisionTreeClassifier, ensemble.RandomForestClassifier], features: spmatrix) -> list:
    # let the classifier predict the labels and return the predictions

    predictions = classifier.predict(features)

    return predictions


def evaluate_predictions(which_evaluation: list, predictions: list, true_labels: list, model: str) -> str:

    evaluation_str = '----------------------------------\nEVALUATION REPORT\n----------------------------------\n'

    if 'confusion_matrix' in which_evaluation:
        cm = metrics.confusion_matrix(true_labels, predictions)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot()
        plt.savefig('plots/confusion_matrix_{}_plot{}.png'.format(model, date.today()))

    if 'class_report' in which_evaluation:
        report = metrics.classification_report(true_labels, predictions)
        evaluation_str += 'CLASSIFICATION REPORT:\n------------------------\n'
        evaluation_str += report

    if 'accuracy' in which_evaluation:
        acc = metrics.accuracy_score(true_labels, predictions)
        evaluation_str += 'ACCURACY:\n---------------\n'
        evaluation_str += acc

    if 'f1' in which_evaluation:
        f1 = metrics.f1_score(true_labels, predictions, average='macro')
        evaluation_str += 'MACRO F1:\n------------\n'
        evaluation_str += f1


    return evaluation_str


def evaluate_grid_search(gridsearch_object: GridSearchCV, axis: str) -> str:

    results_df = pd.DataFrame(gridsearch_object.cv_results_)
    results_df = results_df.sort_values(by=['rank_test_score'])
    results_df = results_df.set_index(
        results_df['params'].apply(lambda x: '_'.join(str(val) for val in x.values()))
    ).rename_axis(axis)
    results_df[['params', 'rank_test_score', 'mean_test_score', 'std_test_score']]