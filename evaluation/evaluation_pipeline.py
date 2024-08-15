#!/usr/bin/env python3
"""
Author: Laura Zeidler
Last changed: 14.08.2024

This script evaluates a specified model (linear or transformer) using various metrics. 
It supports evaluating linear models (e.g. SVM and Decision Trees) and transformer models (e.g. BERT). 
The script handles loading models, making predictions and generating evaluation reports. 
It also supports saving predictions and evaluating grid search results.


### Functionality

1. **Model Loading and Prediction**:
   - For **linear models**: The script loads a pre-trained linear classifier and predicts labels using the specified features.
   - For **transformer models**: The script initializes a tokenizer, loads a fine-tuned transformer model, prepares tokenized input data and predicts labels.

2. **Evaluation**:
   - The script evaluates predictions using specified metrics such as confusion matrix, classification report, accuracy and F1 score.
   - If the `-grid` flag is set, it also evaluates grid search results, if applicable.

3. **Saving Predictions**:
   - If the `-save_pred` flag is set, the script saves the predictions to a JSON file.

"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

if args.model_type == 'linear':
    model = load_linear_model('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.model_path))

    if args.grid:
        estimator = model
        model = estimator.best_estimator_

    features = load_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.features_path))

    if args.model_path.split('_')[-2] == 'grouped':
        dict_name = '/projekte/semrel/WORK-AREA/Users/laura/indices_targets_tdt_split_080101_balanced_grouped.json' 
    else:
        dict_name = '/projekte/semrel/WORK-AREA/Users/laura/indices_targets_tdt_split_080101_balanced.json'

    with open (dict_name, 'r') as jsn:
        ind_n_tars = json.load(jsn)

    targets = ind_n_tars[args.features_path.split('_')[-1]]['targets']

    predictions = predict_labels(model, features)

    labels = model.classes_

if args.model_type == 'transformer':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']
    if args.model_path.endswith('nones'):
        cr = CorpusReader('/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS', which_country, filter_punct=True, filter_digits=True, filter_nes=True, lower=True, split_data=False, sub_sample=False)
    else:
        cr = CorpusReader('/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS', which_country, filter_punct=True, filter_digits=True, filter_nes=False, lower=True, split_data=False, sub_sample=False)
    # cr = CorpusReader('/projekte/semrel/WORK-AREA/Users/laura/toy_corpus', which_country, 'pars', filter_punct=True, lower=True, split_data=False, sub_sample=False)

    data = cr.data

    print('initialising tokeniser')
    tokeniser = initialise_tokeniser('dccuchile/bert-base-spanish-wwm-cased')
    print('done with tokeniser')

    evaluation_string = '###############################\nEvaluation Transformer Models\n###############################\n\n'
    
    model = load_fine_tuned_model('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.model_path))
    model = model.to(device)

    if args.model_path.split('_')[-2] == 'grouped':
        group = True
        filename = '/projekte/semrel/WORK-AREA/Users/laura/data_split/indices_targets_tdt_split_080101_balanced_grouped.json'
        which_country = ['ANT', 'MCA', 'GC', 'CV', 'EP', 'AU', 'ES', 'MX', 'CL', 'PY']
    else:
        group = False
        filename = '/projekte/semrel/WORK-AREA/Users/laura/data_split/tdt_split_080101_balanced.json'
        which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']

    with open(filename, 'r') as jsn:
        split_dict = json.load(jsn)

    # filter train data for specified labels (countries)
    split_dict_filtered = {tdt: {label: value for label, value in labels.items() if label in which_country} for tdt, labels in split_dict.items()}
    # dev_dict = split_dict_filtered['dev']
    dev_dict = split_dict_filtered['test']

    # get tokenised data and corresponding targets
    print('preparing text and targets')
    tokenised_dev_texts, targets = prep_for_dataset(dev_dict, data, tokeniser, which_country, group, batch=True)
    print('done preparing text and targets')
    # print(targets)
    print('predicting labels')
    
    predictions = []
    for batch in tokenised_dev_texts:
        predictions.extend(predict_labels_transformer(model, batch.to(device)))
    print('done predicting labels')
    targets = transform_str_to_int_labels(targets, which_country, reverse=True)
    predictions = transform_str_to_int_labels(list(predictions), which_country, reverse=True)
    labels = which_country

evaluation_string = evaluate_predictions(args.evaluation_metrics, predictions, targets, args.model_path, labels)
if args.grid:
    evaluation_string += '\n\n{}'.format(evaluate_grid_search(estimator, 'C'))

if args.save_pred:
    with open('/projekte/semrel/WORK-AREA/Users/laura/evaluation/predictions.json', 'r') as jsn:
        prediction_dict = json.load(jsn)
    if args.model_type == 'transformer':
        prediction_dict[args.model_path.split('/')[-1]] = [list(predictions), list(targets)]
    else:    
        prediction_dict[args.model_path.split('/')[-1]] = list(predictions)

    with open('/projekte/semrel/WORK-AREA/Users/laura/evaluation/predictions.json', 'w') as jsn:
        json.dump(prediction_dict, jsn)

with open('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.store_path), 'w') as stp:
    stp.write(evaluation_string)