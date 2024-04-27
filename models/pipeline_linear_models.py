import json, sys, argparse
from datetime import date
from prepare_for_training import load_sparse_csr, concatenate_features, shuffle_data
from implement_models import train_SVM, train_DT, train_RF, save_model


parser = argparse.ArgumentParser(description='Run the pipeline in order to train a specified linear model using the specified features.')
parser.add_argument('model', type=str,
                    help='specify the type of model (svm|dt|rf)')
parser.add_argument('features', type=str,
                    help='specify which features will be used (ngrams|tailored|both)')

args = parser.parse_args()

which_features = ['voseo', 'overt_subj', 'subj_inf', 'indef_art_poss', 'diff_tenses', 'non_inv_quest', 'diminutives', 'mas_neg', 'muy_isimo', 'ada', 'clitic_pronouns', 'ser_or_estar']
which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']

# open the file which stores the indices grouped by the set they belong to
with open('/projekte/semrel/WORK-AREA/Users/laura/train_dev_test_split.json', 'r') as jsn:
    split_dict = json.load(jsn)

# filter train data for specified labels (countries)
split_dict_filtered = {key: value for key, value in split_dict.items() if key in which_country}
train_dict = split_dict_filtered['train']

if args.features in ['tailored', 'both']:
    # load the tailored feature vectors
    with open('/projekte/semrel/WORK-AREA/Users/laura/feature_dict.json', 'r') as jsn:
        features = json.load(jsn)

if args.features in ['ngrams', 'both']:
    # load the indices corresponding to the ngram feature vectors
    with open('/projekte/semrel/WORK-AREA/Users/laura/ngram_frequencies_indices.json', 'r') as jsn:
        ngram_indices = json.load(jsn)

    # load the ngram feature vectors
    ngrams = load_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/ngram_frequencies_spmatrix.npz')


# check which features should be used
if args.features == 'ngrams':
    # obtain the ngram features of the train set and concatenate ngram features and tailored features
    train_features, train_targets, train_indices = concatenate_features(ngrams, {}, ngram_indices, train_dict)
elif args.features == 'tailored':
    # obtain the tailored features of the train set and concatenate ngram features and tailored features
    train_features, train_targets, train_indices = concatenate_features(None, features, [], train_dict, which_features)
elif args.features == 'both':
    # obtain the ngram and tailored features of the train set and concatenate ngram features and tailored features
    train_features, train_targets, train_indices = concatenate_features(ngrams, features, ngram_indices, train_dict, which_features, True)

# shuffle the train data
s_train_features, s_train_targets, s_train_indices = shuffle_data(train_features, train_targets, train_indices)

# choose a model & save it
if args.model == 'svm':
    svm_model = train_SVM(s_train_features, s_train_targets)
    save_model(svm_model, 'SVM_model_{}_{}'.format(args.features, date.today()))
if args.model == 'dt':
    dt_model = train_DT(s_train_features, s_train_targets)
    save_model(dt_model, 'DT_model_{}_{}'.format(args.features, date.today()))
if args.model == 'rf':
    rf_model = train_RF(s_train_features, s_train_targets)
    save_model(rf_model, 'RF_model_{}_{}'.format(args.features, date.today()))
