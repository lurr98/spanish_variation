import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from scipy.sparse import spmatrix
from sklearn import svm, tree, ensemble
from sklearn.inspection import permutation_importance
from evaluate_linear import load_linear_model


feature_names = ['VOSEO_vos', 'VOSEO_tú', 'VOSEO_usted', 'VOSEO_vosotros', 'VOSEO_ustedes', 'VOSEO_áis', 'VOSEO_éis', 'VOSEO_ís', 'VOSEO_ás', 'VOSEO_és', 'VOSEO_as', 'VOSEO_es', 'OVSUBJ_yo', 'OVSUBJ_tú', 'OVSUBJ_vos', 'OVSUBJ_él', 'OVSUBJ_nosotros', 'OVSUBJ_vosotros', 'OVSUBJ_ellos', 'SUBJINF_vr', 'SUBJINF_vpp', 'SUBJINF_vpp-00', 'ARTPOSS', 'TENSE_vm', 'TENSE_vc', 'TENSE_vif', 'TENSE_vii', 'TENSE_vip', 'TENSE_vis', 'TENSE_vimp', 'TENSE_vpp', 'TENSE_vps', 'TENSE_vr', 'TENSE_vsf', 'TENSE_vsi', 'TENSE_vsj', 'TENSE_vsp', 'INVQUEST', 'DIM_ico', 'DIM_ito', 'DIM_illo', 'DIM_ingo', 'MASNEG', 'MUYISIMO', 'ADA', 'CLITIC_lo', 'CLITIC_le', 'CLITIC_les', 'SER', 'ESTAR']

def get_coefficients(model: Union[svm.LinearSVC, tree.DecisionTreeClassifier, ensemble.RandomForestClassifier], model_type: str, feature_type: str) -> pd.DataFrame:

    # feature_names = model.get_feature_names_out()

    if model_type == 'svm':
        if feature_type == 'both':
            importances = model.coef_[0][-50:]
        else:
            importances = model.coef_[0]
        importance = {f:i for f, i in zip(feature_names, importances)}
        sorted_importance = {f:i for f, i in sorted(importance.items(), key = lambda item:item[1], reverse=True)}
        cols = ['Coefficients']
    if model_type == 'dt':
        if feature_type == 'both':
            importances = model.feature_importances_[-50:]
        else:
            importances = model.feature_importances_
        importance = {f:i for f, i in zip(feature_names, importances)}
        sorted_importance = {f:i for f, i in sorted(importance.items(), key = lambda item:item[1])}
        cols = ['Weights']
    # for RF we also need to compute the standard deviation
    if model_type == 'rf':
        if feature_type == 'both':
            importances = model.feature_importances_[-50:]
        else:
            importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        importance = {f: [i, j] for f, i, j in zip(feature_names, importances, std)}
        sorted_importance = {f: i for f, i in sorted(importance.items(), key = lambda item:item[1][0])}
        cols = ['Weights', 'Standard Deviation']

    coefs = pd.DataFrame.from_dict(sorted_importance, orient='index', columns=cols)

    return coefs


# code from https://datascience.stackexchange.com/questions/107580/is-there-a-way-to-output-feature-importance-based-on-the-outputted-class
# TODO: be critical with this, I am not sure, whether this actually makes sense
def get_feature_importances_permutation(model: Union[svm.LinearSVC, tree.DecisionTreeClassifier, ensemble.RandomForestClassifier], features: list, targets: list) -> pd.DataFrame:
    # in order to get the importances per class, just pass features and targets where the prediction was the desired class
    result = permutation_importance(model, features, targets, n_repeats=100, random_state=0)
    df = pd.DataFrame({'feature_name': feature_names, 'feature_importance': result.importances_mean})
    
    return df


def plot_feature_importance(df: pd.DataFrame, model_type: str, model_path: str) -> None:

    if model_type in ['svm', 'dt']:
        df.plot.barh(figsize=(7, 10))
        # SVM_models/SVM_model_tailored__2024-05-13
        if args.model_type == 'svm':
            add = 'coefficients'
        if args.model_type == 'dt':
            add = 'weights'
        plt.title('{} feature {}'.format(' '.join(model_path.split('/')[1].split('_')[:-1]), add))
        plt.axvline(x=0, color=".5")
        plt.xlabel('Raw {} values'.format(add))
        plt.subplots_adjust(left=0.3)

    if model_type == 'rf':
        add = 'weights'

        fig, ax = plt.subplots()
        forest_importances = pd.Series(list(df['Weights']), index=list(df.index))
        std = list(df['Standard Deviation'])
        forest_importances.plot.bar(yerr=std, ax=ax, figsize=(10, 7))
        ax.set_title('Feature importances for {} using MDI'.format(' '.join(model_path.split('/')[1].split('_')[:-1]), add))
        ax.set_ylabel('Mean decrease in impurity')
        fig.tight_layout()

    plt.savefig('plots/feature_importances/{}_feature_importances.png'.format(model_path.split('/')[1]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the pipeline in order to evaluate a specified linear model using the specified metrics.')
    parser.add_argument('model_path', type=str,
                        help='pass the path to the model to be interpreted')
    parser.add_argument('model_type', type=str,
                        help='pass the type of model')
    
    args = parser.parse_args()
    print('The script is running with the following arguments: {}'.format(args))

    estimator = load_linear_model('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.model_path))
    model = estimator.best_estimator_

    coefs = get_coefficients(model, args.model_type)

    plot_feature_importance(coefs, args.model_type, args.model_path)
