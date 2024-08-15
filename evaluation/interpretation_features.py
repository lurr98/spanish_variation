#!/usr/bin/env python3
"""
Author: Laura Zeidler
Last changed: 14.08.2024

This script is used for analyzing and interpreting the importance of features in linear models (SVM, Decision Trees). 
It provides functionality to extract and plot feature importances, generate decision tree rules and visualize the first layers of a decision tree model. 

### Functionality

1. **Model Coefficients Extraction**:
   - For SVM models, extracts and sorts coefficients to determine feature importance.
   - For Decision Trees, extracts and sorts feature importances.

2. **Feature Importance Visualization**:
   - Plots feature importances for SVM and Decision Trees.
   - Handles different cases including zero importance values and saves the plots with appropriate titles.

3. **Decision Tree Interpretation**:
   - Extracts and prints decision tree rules for 'dt' models.
   - Visualizes and saves the decision tree plot.

4. **Most Informative Features for Class**:
   - For SVM models, identifies and visualizes the most informative features for each class.

"""

import argparse, json, time, sys, re, matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from scipy.sparse import spmatrix, vstack
from sklearn import svm, tree, ensemble
from sklearn.tree import export_text, plot_tree
from evaluate_all_models import load_linear_model

sys.path.append("..")
from basics import load_sparse_csr

feature_names_tailored = ['VOSEO_vos', 'VOSEO_tú', 'VOSEO_usted', 'VOSEO_vosotros', 'VOSEO_ustedes', 'VOSEO_áis', 'VOSEO_éis', 'VOSEO_ís', 'VOSEO_ás', 'VOSEO_és', 'VOSEO_as', 'VOSEO_es', 'OVSUBJ_yo', 'OVSUBJ_tú', 'OVSUBJ_vos', 'OVSUBJ_él', 'OVSUBJ_nosotros', 'OVSUBJ_vosotros', 'OVSUBJ_ellos', 'SUBJINF_vr', 'SUBJINF_vpp', 'SUBJINF_vpp-00', 'ARTPOSS', 'TENSE_vm', 'TENSE_vc', 'TENSE_vif', 'TENSE_vii', 'TENSE_vip', 'TENSE_vis', 'TENSE_vimp', 'TENSE_vpp', 'TENSE_vps', 'TENSE_vr', 'TENSE_vsf', 'TENSE_vsi', 'TENSE_vsj', 'TENSE_vsp', 'INVQUEST', 'DIM_ico', 'DIM_ito', 'DIM_illo', 'DIM_ingo', 'MASNEG', 'MUYISIMO', 'ADA', 'CLITIC_lo', 'CLITIC_le', 'CLITIC_les', 'SER', 'ESTAR']


def get_feature_names(feature_names_type: str) -> list:

    if feature_names_type == 'tailored':
        feature_names = feature_names_tailored
    else:
        with open('/projekte/semrel/WORK-AREA/Users/laura/ngram_features/ngram_frequencies_indices_feature_names.json', 'r') as jsn:
            feature_names_dict = json.load(jsn)
            if feature_names_type == 'ngrams':
                feature_names = feature_names_dict['feature_names']
            elif feature_names_type == 'nofeat':
                feature_names = feature_names_dict['feature_names_nofeat']
            elif feature_names_type == 'nones':
                feature_names = feature_names_dict['feature_names_nones']
            elif feature_names_type == 'both':
                feature_names = feature_names_dict['feature_names_nofeat'] + feature_names_tailored
            elif feature_names_type == 'bothn':
                feature_names = feature_names_dict['feature_names_nones'] + feature_names_tailored

    return feature_names


def get_coefficients(model: Union[svm.LinearSVC, tree.DecisionTreeClassifier], model_type: str, feature_names_type: str) -> pd.DataFrame:

    feature_names = get_feature_names(feature_names_type)

    if model_type == 'svm':
        importances = model.coef_[0]
        importance = {f:i for f, i in zip(feature_names, importances)}
        sorted_list = sorted(importance.items(), key = lambda item:item[1], reverse=True)
        n_first, n_last = sorted_list[:25], sorted_list[-25:]
        sorted_importance = {f:i for f, i in n_first+n_last}
        cols = ['Importances']
    if model_type == 'dt':
        importances = model.feature_importances_
        importance = {f:i for f, i in zip(feature_names, importances)}
        sorted_importance = {f: i for f, i in sorted(importance.items(), key = lambda item:item[1])[-50:]}
        cols = ['Importances']

    coefs = pd.DataFrame.from_dict(sorted_importance, orient='index', columns=cols)

    return coefs


def most_informative_feature_for_class(classifier: svm.LinearSVC, label: str, feature_names: list) -> pd.DataFrame:
    label_id = list(classifier.classes_).index(label)
    importance = {f:i for f, i in zip(feature_names, classifier.coef_[label_id])}
    sorted_importances = {f: i for f, i in sorted(importance.items(), key = lambda item:item[1])[-20:]}
    
    df = pd.DataFrame.from_dict(sorted_importances, orient='index', columns=['Importances'])    

    return df


def get_dt_rules(classifier: tree.DecisionTreeClassifier, feature_names: list) -> str:

    rules = export_text(classifier, feature_names=feature_names)

    return rules


def plot_dt(model: tree.DecisionTreeClassifier, feature_names: list, model_path: str) -> None:

    def replace_text(obj):
        if type(obj) == matplotlib.text.Annotation:
            txt = obj.get_text()
            txt = re.sub("\nsamples[^$]*]","",txt)
            obj.set_text(txt)
        return obj
    
    # plt.figure(figsize=(30,12))
    fig, ax = plt.subplots(figsize=(16,8))
    plot_tree(model, ax=ax, feature_names=feature_names, max_depth=3, fontsize=10)
    print(ax.properties())
    ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]
    print(ax.properties())
    plt.savefig('plots/feature_importances/{}_dt_rules.png'.format(model_path.split('/')[1]))


def plot_feature_importance(df: pd.DataFrame, model_path: str, perm: str='none') -> None:

    # filter zeros for better readability
    zeros = False
    i = 0
    for imp_value in df.iloc:
        if df.iloc[i]['Importances'] == 0.0:
            if not zeros:
                print('not zeros')
                df = df.rename(index={imp_value.name: '…'})
                zeros = True
            else:
                print('zeros')
                df = df.drop([imp_value.name])
                i -= 1
        i += 1

    df.plot.barh(figsize=(7, 10))
    # SVM_models/SVM_model_tailored__2024-05-13
    if args.model_type == 'svm':
        add = 'coefficients'
    if args.model_type == 'dt':
        add = 'weights'
    if perm != 'none':
        plt.title('{} feature {} for class {}'.format(' '.join(model_path.split('/')[-1].split('_')[:-1]), add, perm))
    else:
        title_name = re.sub(' nofeat tf ', ' ',' '.join( model_path.split('/')[-1].split('_')[:-1]))
        title_name = re.sub(' nofeat ', ' ', title_name)
        if 'nones' in title_name:
            plt.title('{} feature {} without NEs'.format(re.sub('nones', '', title_name), add))
        elif 'grouped' in title_name:
            plt.title('{} feature {} on grouped classes'.format(re.sub('grouped', '', title_name), add))
        else:
            plt.title('{} feature {}'.format(title_name, add))
    plt.axvline(x=0, color=".5")
    plt.xlabel('Raw {} values'.format(add))
    plt.subplots_adjust(left=0.3)
    
    if perm != 'none':
        plt.savefig('plots/feature_importances/importances_by_country/{}/{}_feature_importances_{}.png'.format(perm, model_path.split('/')[-1], perm))
    else:
        plt.savefig('plots/feature_importances/{}_feature_importances.png'.format(model_path.split('/')[-1]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the pipeline in order to evaluate a specified linear model using the specified metrics.')
    parser.add_argument('model_path', type=str,
                        help='pass the path to the model to be interpreted')
    parser.add_argument('model_type', type=str,
                        help='pass the type of model (svm|dt)')
    parser.add_argument('feature_names', type=str,
                        help='pass the type of feature names (tailored|ngrams|nofeat|both|nones)')
    parser.add_argument('-ftp','--features_targets_preds', nargs='+', 
                    help='specify the path to the features and then the targets')
    
    args = parser.parse_args()
    print('The script is running with the following arguments: {}'.format(args))

    estimator = load_linear_model('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.model_path))
    print('loaded model')
    try:
        model = estimator.best_estimator_
    except AttributeError:
        model = estimator

    start = time.time()
    print('getting coefficients')
    coefs = get_coefficients(model, args.model_type, args.feature_names)
    end = time.time()
    print('Getting coefficients took {} seconds.'.format(end-start))

    plot_feature_importance(coefs, args.model_path)

    if args.features_targets_preds:

        feature_names = get_feature_names(args.feature_names)   

        with open ('/projekte/semrel/WORK-AREA/Users/laura/evaluation/predictions.json', 'r') as jsn:
            pred_dict = json.load(jsn)
        predictions = pred_dict[args.model_path.split('/')[-1]]

        if args.model_type == 'svm':

            for label in list(set(predictions)):
                most_informative_features = most_informative_feature_for_class(model, label, feature_names)

                print('now plotting')
                plot_feature_importance(most_informative_features, args.model_path, label)

        elif args.model_type == 'dt':
            rules = get_dt_rules(model, feature_names)
            plot_dt(model, feature_names, args.model_path)

            with open('/projekte/semrel/WORK-AREA/Users/laura/evaluation/DT_models/{}'.format(args.model_path.split('/')[-1]), 'w') as r:
                r.write(rules)