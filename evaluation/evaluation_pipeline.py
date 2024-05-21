import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import sys, argparse, json
from evaluate_linear import load_linear_model, predict_labels, evaluate_predictions, evaluate_grid_search, load_fine_tuned_model, predict_labels_transformer

sys.path.append("..")
from basics import load_sparse_csr, initialise_tokeniser
from corpus.corpus_reader import CorpusReader
from models.prepare_for_training import prep_for_dataset


parser = argparse.ArgumentParser(description='Run the pipeline in order to evaluate a specified linear model using the specified metrics.')
parser.add_argument('model_type', type=str,
                    help='pass the type of model to be evaluated (linear|transformer)')
parser.add_argument('model_path', type=str,
                    help='pass the path to the model to be evaluated')
parser.add_argument('features_path', type=str,
                    help='pass the path to the dev or test features')
parser.add_argument('store_path', type=str,
                    help='specify the path of the evaluation file')
parser.add_argument('-ev','--evaluation_metrics', nargs='+', 
                    help='specify the metrics to evaluate the model (f1, accuracy, confusion_matrix, class_report)', required=True)
parser.add_argument('-grid', action='store_true', 
                    help='state whether the model is a GridSearchCV object')

args = parser.parse_args()
print('The script is running with the following arguments: {}'.format(args))

which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']

if args.model_type == 'linear':
    model = load_linear_model('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.model_path))

    if args.grid:
        estimator = model
        model = estimator.best_estimator_

    features = load_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.features_path))

    with open ('/projekte/semrel/WORK-AREA/Users/laura/indices_targets_tdt_split_080101_balanced.json', 'r') as jsn:
        ind_n_tars = json.load(jsn)

    targets = ind_n_tars[args.features_path.split('_')[-1]]['targets']

    print(set(targets))
    print(len(targets))

    predictions = predict_labels(model, features)

    print(set(predictions))
    print(len(predictions))

    labels = model.classes_
    print(labels)

if args.model_type == 'transformer':
    model = load_fine_tuned_model('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.model_path))

    with open('/projekte/semrel/WORK-AREA/Users/laura/toy_train_dev_test_split.json', 'r') as jsn:
    # with open('/projekte/semrel/WORK-AREA/Users/laura/data_split/tdt_split_080101_balanced.json', 'r') as jsn:
        split_dict = json.load(jsn)

    # filter train data for specified labels (countries)
    split_dict_filtered = {tdt: {label: value for label, value in labels.items() if label in which_country} for tdt, labels in split_dict.items()}

    dev_dict = split_dict_filtered['dev']

    # TODO: filter_punct=True?
    # cr = CorpusReader('/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS', which_country, 'pars', filter_punct=True)
    # cr = CorpusReader('/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS', which_country, 'pars', filter_punct=True, filter_digits=True, lower=True)
    cr = CorpusReader('/projekte/semrel/WORK-AREA/Users/laura/toy_corpus', which_country, 'pars', filter_punct=True, lower=True, split_data=False, sub_sample=False)

    data = cr.data

    tokeniser = initialise_tokeniser('dccuchile/bert-base-spanish-wwm-cased')
    # get tokenised data and corresponding targets
    # tokenised_train_text, train_targets_int = prep_for_dataset(train_dict, data, args.trunc, tokeniser, which_country)
    tokenised_dev_text, targets = prep_for_dataset(dev_dict, data, args.trunc, tokeniser, which_country)

    predictions = predict_labels_transformer(model, tokenised_dev_text)

    labels = which_country
    

evaluation_string = evaluate_predictions(args.evaluation_metrics, predictions, targets, args.model_path, '_'.join(args.model_path.split('_')[2:][:-1]), labels)
if args.grid:
    evaluation_string += '\n\n{}'.format(evaluate_grid_search(estimator, 'C'))

with open('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.store_path), 'w') as stp:
    stp.write(evaluation_string)