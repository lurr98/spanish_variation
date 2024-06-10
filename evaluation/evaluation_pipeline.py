import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

import sys, argparse, json, torch
torch.cuda.empty_cache()

from evaluate_all_models import load_linear_model, predict_labels, evaluate_predictions, evaluate_grid_search, load_fine_tuned_model, predict_labels_transformer

sys.path.append("..")
from basics import load_sparse_csr, initialise_tokeniser
from corpus.corpus_reader import CorpusReader
from models.prepare_for_training import prep_for_dataset, transform_str_to_int_labels


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
parser.add_argument('-save_pred', action='store_true', 
                    help='state whether to store the predictions')

args = parser.parse_args()
print('The script is running with the following arguments: {}'.format(args))

which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']
# which_country = ['BO', 'CL']

if args.model_type == 'linear':
    model = load_linear_model('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.model_path))

    if args.grid:
        estimator = model
        model = estimator.best_estimator_

    features = load_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.features_path))

    if args.model_path.endswith('grouped'):
        dict_name = '/projekte/semrel/WORK-AREA/Users/laura/indices_targets_tdt_split_080101_balanced_grouped.json' 
    else:
        dict_name = '/projekte/semrel/WORK-AREA/Users/laura/indices_targets_tdt_split_080101_balanced.json'

    with open (dict_name, 'r') as jsn:
        ind_n_tars = json.load(jsn)

    targets = ind_n_tars[args.features_path.split('_')[-1]]['targets']

    # if args.model_path.endswith('grouped'):
    #     target_dict = {'CU': 'ANT', 'DO': 'ANT', 'PR': 'ANT', 'PA': 'ANT', 'SV': 'MCA', 'NI': 'MCA', 'HN': 'MCA', 'GT': 'GC', 'CR': 'GC', 'CO': 'CV', 'VE': 'CV', 'EC': 'EP', 'PE': 'EP', 'BO': 'EP', 'AR': 'AU', 'UY': 'AU', 'ES': 'ES', 'MX': 'MX', 'CL': 'CL', 'PY': 'PY'}
    #     targets = [target_dict[target] for target in targets]

    predictions = predict_labels(model, features)

    labels = model.classes_

if args.model_type == 'transformer':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_fine_tuned_model('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.model_path))
    model = model.to(device)

    # TODO: change for grouped
    # with open('/projekte/semrel/WORK-AREA/Users/laura/toy_train_dev_test_split.json', 'r') as jsn:
    with open('/projekte/semrel/WORK-AREA/Users/laura/data_split/tdt_split_080101_balanced.json', 'r') as jsn:
        split_dict = json.load(jsn)

    # filter train data for specified labels (countries)
    split_dict_filtered = {tdt: {label: value for label, value in labels.items() if label in which_country} for tdt, labels in split_dict.items()}

    dev_dict = split_dict_filtered['dev']

    # TODO: filter_punct=True?
    cr = CorpusReader('/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS', which_country, filter_punct=True, split_data=False, sub_sample=False)
    # cr = CorpusReader('/projekte/semrel/WORK-AREA/Users/laura/toy_corpus', which_country, 'pars', filter_punct=True, lower=True, split_data=False, sub_sample=False)

    data = cr.data

    if args.model_path.endswith('grouped'):
        which_country = ['ANT', 'MCA', 'GC', 'CV', 'EP', 'AU', 'ES', 'MX', 'CL', 'PY']

    print('initialising tokeniser')
    tokeniser = initialise_tokeniser('dccuchile/bert-base-spanish-wwm-cased')
    print('done with tokeniser')
    # get tokenised data and corresponding targets
    # tokenised_train_text, train_targets_int = prep_for_dataset(train_dict, data, args.trunc, tokeniser, which_country)
    print('preparing text and targets')
    tokenised_dev_texts, targets = prep_for_dataset(dev_dict, data, tokeniser, which_country, batch=True)
    print('done preparing text and targets')
    # print(targets)

    print('predicting labels')
    # predictions = predict_labels_transformer(model, tokenised_dev_text.to(device))
    predictions = []
    for batch in tokenised_dev_texts:
        predictions.extend(predict_labels_transformer(model, batch.to(device)))
    print('done predicting labels')
    # print(predictions)
    targets = transform_str_to_int_labels(targets, which_country, reverse=True)
    predictions = transform_str_to_int_labels(list(predictions), which_country, reverse=True)

    labels = which_country
    

evaluation_string = evaluate_predictions(args.evaluation_metrics, predictions, targets, args.model_path, labels)
if args.grid:
    evaluation_string += '\n\n{}'.format(evaluate_grid_search(estimator, 'C'))

if args.save_pred:
    with open('/projekte/semrel/WORK-AREA/Users/laura/evaluation/predictions.json', 'r') as jsn:
        prediction_dict = json.load(jsn)
    prediction_dict[args.model_path.split('/')[-1]] = list(predictions)

    with open('/projekte/semrel/WORK-AREA/Users/laura/evaluation/predictions.json', 'w') as jsn:
        json.dump(prediction_dict, jsn)
    # evaluation_string += '\n\nPredictions: {}\nTargets: {}'.format(str(list(predictions)), str(targets))

with open('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.store_path), 'w') as stp:
    stp.write(evaluation_string)