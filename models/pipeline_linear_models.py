import json, sys, argparse, time, re
from datetime import date
from prepare_for_training import load_sparse_csr, concatenate_features, shuffle_data, prepare_data_full
from implement_models import train_SVM, train_DT, train_RF, save_model

sys.path.append("..")
from basics import save_sparse_csr


parser = argparse.ArgumentParser(description='Run the pipeline in order to train a specified linear model using the specified features.')
parser.add_argument('features', type=str,
                    help='specify which features will be used (ngrams|tailored|both)')
parser.add_argument('model', type=str,
                    help='specify the type of model (svm|dt|rf|none)')
parser.add_argument('store_path', type=str,
                    help='specify the path and name of the sparse matrix file (no .npz extension, will be added automatically)')
parser.add_argument('-sit', action='store_true',
                    help='state whether to save indices and targets')
parser.add_argument('-grid', action='store_true',
                    help='state whether to use grid search to find best parameters')

args = parser.parse_args()
print('The script is running with the following arguments: {}'.format(args))

which_features = ['voseo', 'overt_subj', 'subj_inf', 'indef_art_poss', 'diff_tenses', 'non_inv_quest', 'diminutives', 'mas_neg', 'muy_isimo', 'ada', 'clitic_pronouns', 'ser_or_estar']
which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']

if args.features == 'both':
    dir_name = re.sub('ngram_features', 'both_features', args.store_path.split('.')[0])
else:
    dir_name = args.store_path.split('.')[0]

# concatenate features
if args.model == 'none':
    s_train_features, s_train_targets, s_train_indices = prepare_data_full('/projekte/semrel/WORK-AREA/Users/laura/data_split/tdt_split_080101_balanced.json', 'train', which_country, args.features, which_features, args.store_path, shuffle=args.sit)
    # s_train_features, s_train_targets, s_train_indices = prepare_data_full('/projekte/semrel/WORK-AREA/Users/laura/data_split/toy_train_dev_test_split.json', 'train', which_country, args.features, which_features, args.store_path, shuffle=args.sit)

    # if no model is to be run, concatenate the dev and test features as well and save everything 
    s_dev_features, s_dev_targets, s_dev_indices = prepare_data_full('/projekte/semrel/WORK-AREA/Users/laura/data_split/tdt_split_080101_balanced.json', 'dev', which_country, args.features, which_features, args.store_path, shuffle=args.sit)
    # s_dev_features, s_dev_targets, s_dev_indices = prepare_data_full('/projekte/semrel/WORK-AREA/Users/laura/data_split/toy_train_dev_test_split.json', 'dev', which_country, args.features, which_features, args.store_path, shuffle=args.sit)
    s_test_features, s_test_targets, s_test_indices = prepare_data_full('/projekte/semrel/WORK-AREA/Users/laura/data_split/tdt_split_080101_balanced.json', 'test', which_country, args.features, which_features, args.store_path, shuffle=args.sit)
    # s_test_features, s_test_targets, s_test_indices = prepare_data_full('/projekte/semrel/WORK-AREA/Users/laura/data_split/toy_train_dev_test_split.json', 'test', which_country, args.features, which_features, args.store_path, shuffle=args.sit)
    
    save_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/{}_train'.format(dir_name), s_train_features)
    save_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/{}_dev'.format(dir_name), s_dev_features)
    save_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/{}_test'.format(dir_name), s_test_features)

    if args.sit:
        ind_target_dict = {'train': {'indices': s_train_indices, 'targets': s_train_targets}, 'dev': {'indices': s_dev_indices, 'targets': s_dev_targets}, 'test': {'indices': s_test_indices, 'targets': s_test_targets}}
        with open('/projekte/semrel/WORK-AREA/Users/laura/indices_targets_tdt_split_080101_balanced_test.json', 'w') as jsn:
            json.dump(ind_target_dict, jsn)

# load the features and targets
if args.model in ['svm', 'dt', 'rf']:
    start = time.time()
    s_train_features = load_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/{}_train'.format(dir_name))
    with open('/projekte/semrel/WORK-AREA/Users/laura/indices_targets_tdt_split_080101_balanced.json', 'r') as jsn:
        ind_target_dict = json.load(jsn)
    s_train_targets = ind_target_dict['train']['targets']
    end = time.time()
    print('Loading train features and targets took {} seconds.'.format(end - start))

# choose a model & save it
if args.model == 'svm':
    print('Training SVM model.')
    start = time.time()
    svm_model = train_SVM(s_train_features, s_train_targets, args.grid)
    end = time.time()
    print('Training SVM model took {} seconds.'.format(end - start))
    save_model(svm_model, '/projekte/semrel/WORK-AREA/Users/laura/SVM_models/SVM_model_{}_{}_{}'.format(args.features, '_'.join(dir_name.split('/')[-1].split('_')[3:]), date.today()))
if args.model == 'dt':
    print('Training DT model.')
    start = time.time()
    dt_model = train_DT(s_train_features, s_train_targets, args.grid)
    end = time.time()
    print('Training DT model took {} seconds.'.format(end - start))
    save_model(dt_model, '/projekte/semrel/WORK-AREA/Users/laura/DT_models/DT_model_{}_{}_{}'.format(args.features, '_'.join(dir_name.split('/')[-1].split('_')[3:]), date.today()))
if args.model == 'rf':
    print('Training RF model.')
    start = time.time()
    rf_model = train_RF(s_train_features, s_train_targets, args.grid)
    end = time.time()
    print('Training RF model took {} seconds.'.format(end - start))
    save_model(rf_model, '/projekte/semrel/WORK-AREA/Users/laura/RF_models/RF_model_{}_{}_{}'.format(args.features, '_'.join(dir_name.split('/')[-1].split('_')[3:]), date.today()))
