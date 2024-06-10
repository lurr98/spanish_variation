import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

import pickle, torch, re
torch.cuda.empty_cache()

import matplotlib.pyplot as plt
import pandas as pd
from typing import Sequence
from datetime import date
from sklearn import svm, tree, ensemble, metrics
from sklearn.model_selection import GridSearchCV
from transformers import AutoModelForSequenceClassification, BertTokenizer
from scipy.sparse import spmatrix
from typing import Union


#################################
#        Linear models          #
#################################

def load_linear_model(model_path: str) -> Union[svm.LinearSVC, tree.DecisionTreeClassifier, ensemble.RandomForestClassifier]:

    # load classifier
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

    return classifier


def predict_labels(classifier: Union[svm.LinearSVC, tree.DecisionTreeClassifier, ensemble.RandomForestClassifier], features: spmatrix) -> list:
    # let the classifier predict the labels and return the predictions

    predictions = classifier.predict(features)

    return predictions


######################################
#        Transformer models          #
######################################

def load_fine_tuned_model(model_path: str) -> AutoModelForSequenceClassification:

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return model


def predict_labels_transformer(model: AutoModelForSequenceClassification, tokenised_text: Sequence) -> list:

    outputs = model(**tokenised_text)
    predicted_labels = outputs.logits.argmax(-1)

    return predicted_labels


########################
#        Both          #
########################

def evaluate_predictions(which_evaluation: list, predictions: list, true_labels: list, model: str, labels: list) -> str:

    evaluation_str = '----------------------------------\nEVALUATION REPORT\n----------------------------------\n'

    if 'confusion_matrix' in which_evaluation:
        model = re.sub('/', '_', model)
        cm = metrics.confusion_matrix(true_labels, predictions, labels=labels)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(10,10))
        disp.plot(ax=ax)
        plt.savefig('plots/confusion_matrices/confusion_matrix_{}_plot{}.png'.format(model, date.today()))

    if 'class_report' in which_evaluation:
        report = metrics.classification_report(true_labels, predictions)
        evaluation_str += 'CLASSIFICATION REPORT:\n------------------------\n'
        evaluation_str += report

    if 'accuracy' in which_evaluation:
        acc = metrics.accuracy_score(true_labels, predictions)
        evaluation_str += '\n\nACCURACY:\n---------------\n'
        evaluation_str += str(acc)

    if 'f1' in which_evaluation:
        f1 = metrics.f1_score(true_labels, predictions, average='macro')
        evaluation_str += '\n\nMACRO F1:\n------------\n'
        evaluation_str += str(f1)


    return evaluation_str


def evaluate_grid_search(gridsearch_object: GridSearchCV, axis: str) -> str:

    results_df = pd.DataFrame(gridsearch_object.cv_results_)
    results_df = results_df.sort_values(by=['rank_test_score'])
    results_df = results_df.set_index(
        results_df['params'].apply(lambda x: '_'.join(str(val) for val in x.values()))
    ).rename_axis(axis)
    result_string = results_df[['params', 'rank_test_score', 'mean_test_score', 'std_test_score']].to_string()

    return result_string


