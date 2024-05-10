import sys, argparse, json
from evaluate_linear import load_linear_model, predict_labels, evaluate_predictions

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

args = parser.parse_args()

model = load_linear_model('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.model_path))

features = load_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(args.features_path))

with open ('/projekte/semrel/WORK-AREA/Users/laura/indices_targets_tdt_split_080101_balanced.json', 'r') as jsn:
    ind_n_tars = json.load(jsn)

targets = ind_n_tars[args.features_path.split('_')[-1]]['targets']

predictions = predict_labels(model, features)

evaluation_string = evaluate_predictions(args.evaluation_metrics, predictions, targets, model)

with open(args.store_path, 'w') as stp:
    stp.write(evaluation_string)