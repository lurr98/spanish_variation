import sys, argparse, json
from evaluate_linear import load_linear_model, predict_labels, evaluate_predictions, evaluate_grid_search

sys.path.append("..")
from basics import load_sparse_csr


parser = argparse.ArgumentParser(description='Run the pipeline in order to evaluate a specified linear model using the specified metrics.')
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

evaluation_string = evaluate_predictions(args.evaluation_metrics, predictions, targets, model, '_'.join(args.model_path.split('_')[2:][:-1]), labels)
if args.grid:
    evaluation_string += '\n\n{}'.format(evaluate_grid_search(estimator, 'C'))

with open('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.store_path), 'w') as stp:
    stp.write(evaluation_string)