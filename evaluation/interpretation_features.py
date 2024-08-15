import argparse, json, time, sys, re, matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from scipy.sparse import spmatrix, vstack
from sklearn import svm, tree, ensemble
from sklearn.tree import export_text, plot_tree
from sklearn.inspection import permutation_importance
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


def get_coefficients(model: Union[svm.LinearSVC, tree.DecisionTreeClassifier, ensemble.RandomForestClassifier], model_type: str, feature_names_type: str) -> pd.DataFrame:

    feature_names = get_feature_names(feature_names_type)

    if model_type == 'svm':
        # if feature_type == 'both':
        #     importances = model.coef_[0][-50:]
        # else:
        importances = model.coef_[0]
        importance = {f:i for f, i in zip(feature_names, importances)}
        sorted_list = sorted(importance.items(), key = lambda item:item[1], reverse=True)
        n_first, n_last = sorted_list[:25], sorted_list[-25:]
        sorted_importance = {f:i for f, i in n_first+n_last}
        cols = ['Importances']
    if model_type == 'dt':
        # if feature_type == 'both':
        #     importances = model.feature_importances_[-50:]
        # else:
        importances = model.feature_importances_
        importance = {f:i for f, i in zip(feature_names, importances)}
        sorted_importance = {f: i for f, i in sorted(importance.items(), key = lambda item:item[1])[-50:]}
        cols = ['Importances']
    # for RF we also need to compute the standard deviation
    if model_type == 'rf':
        # if feature_type == 'both':
        #     importances = model.feature_importances_[-50:]
        # else:
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        importance = {f: [i, j] for f, i, j in zip(feature_names, importances, std)}
        sorted_importance = {f: i for f, i in sorted(importance.items(), key = lambda item:item[1][0])[-50:]}
        cols = ['Weights', 'Standard Deviation']

    # make sure dict is not too large

    coefs = pd.DataFrame.from_dict(sorted_importance, orient='index', columns=cols)

    return coefs


# code from https://datascience.stackexchange.com/questions/107580/is-there-a-way-to-output-feature-importance-based-on-the-outputted-class
# TODO: be critical with this, I am not sure, whether this actually makes sense
def get_feature_importances_permutation(model: Union[tree.DecisionTreeClassifier, ensemble.RandomForestClassifier], features: spmatrix, feature_names: list, targets: list) -> pd.DataFrame:
    # in order to get the importances per class, just pass features and targets where the prediction was the desired class
    result = permutation_importance(model, features.toarray(), targets, n_repeats=2, random_state=0)
    importance = {f:i for f, i in zip(feature_names, result.importances_mean)}
    sorted_importances = {f: i for f, i in sorted(importance.items(), key = lambda item:item[1])[-50:]}
    df = pd.DataFrame.from_dict(sorted_importances, orient='index', columns=['Importances'])    
    return df


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



def plot_feature_importance(df: pd.DataFrame, model_type: str, model_path: str, perm: str='none') -> None:

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

    if model_type in ['svm', 'dt']:
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

    if model_type == 'rf':
        add = 'weights'

        fig, ax = plt.subplots()
        if perm != 'none':
            df.plot.barh(figsize=(7, 10))
            plt.title('{} feature importances for class {}'.format(' '.join(model_path.split('/')[-1].split('_')[:-1]), perm))
            plt.axvline(x=0, color=".5")
            plt.xlabel('Raw importance values')
            plt.subplots_adjust(left=0.3)
        else:
            forest_importances = pd.Series(list(df['Weights']), index=list(df.index))
            std = list(df['Standard Deviation'])
            forest_importances.plot.bar(yerr=std, ax=ax, figsize=(10, 7))
            ax.set_title('Feature importances for {} using MDI'.format(' '.join(model_path.split('/')[-1].split('_')[:-1]), add))
            ax.set_ylabel('Mean decrease in impurity')
            fig.tight_layout()
    
    if perm != 'none':
        plt.savefig('plots/feature_importances/importances_by_country/{}/{}_feature_importances_{}.png'.format(perm, model_path.split('/')[-1], perm))
    else:
        plt.savefig('plots/feature_importances/{}_feature_importances.png'.format(model_path.split('/')[-1]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the pipeline in order to evaluate a specified linear model using the specified metrics.')
    parser.add_argument('model_path', type=str,
                        help='pass the path to the model to be interpreted')
    parser.add_argument('model_type', type=str,
                        help='pass the type of model (svm|dt|rf)')
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

    plot_feature_importance(coefs, args.model_type, args.model_path)

    if args.features_targets_preds:

        feature_names = get_feature_names(args.feature_names)   

        with open ('/projekte/semrel/WORK-AREA/Users/laura/evaluation/predictions.json', 'r') as jsn:
            pred_dict = json.load(jsn)
        predictions = pred_dict[args.model_path.split('/')[-1]]

        if args.model_type == 'svm':

            for label in list(set(predictions)):
                most_informative_features = most_informative_feature_for_class(model, label, feature_names)

                print('now plotting')
                plot_feature_importance(most_informative_features, args.model_type, args.model_path, label)

        elif args.model_type == 'dt':
            rules = get_dt_rules(model, feature_names)
            plot_dt(model, feature_names, args.model_path)

            with open('/projekte/semrel/WORK-AREA/Users/laura/evaluation/DT_models/{}'.format(args.model_path.split('/')[-1]), 'w') as r:
                r.write(rules)
                
        # else:
# 
        #     features = load_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.features_targets_preds[0]))
        #     with open ('/projekte/semrel/WORK-AREA/Users/laura/indices_targets_tdt_split_080101_balanced.json', 'r') as jsn:
        #         ind_n_tars = json.load(jsn)
# 
        #     targets = ind_n_tars[args.features_targets_preds[0].split('_')[-1]]['targets']
# 
        #     for label in list(set(predictions)):
        #         print('getting coefficients for label {}'.format(label))
        #         start = time.time()
        #         indices = [i for i, des_label in enumerate(predictions) if des_label == label]
        #         print(len(indices))
        #         label_targets = [targets[i] for i in indices]
        #         # initialise array
        #         all_features_array = []
# 
        #         for idx in indices:
        #             if not isinstance(all_features_array, list):
        #                 # add the ngram feature vector to the other feature vectors
        #                 all_features_array = vstack([all_features_array, features[idx, :]])
        #             else:
        #                 # initialise all_feature_array with the first ngram feature vector
        #                 all_features_array = features[idx, :]
# 
        #         print(len(label_targets))
        #         print(all_features_array.toarray().shape)
# 
        #         fip = get_feature_importances_permutation(model, all_features_array, feature_names, label_targets)
        #         end = time.time()
        #         print('getting coefficients took {} seconds'.format(end-start))
# 
        #         print('now plotting')
        #         plot_feature_importance(fip, args.model_type, args.model_path, label)
# 
