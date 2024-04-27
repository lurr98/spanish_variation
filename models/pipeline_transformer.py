import json, sys, argparse
from datetime import date
from implement_transformer_model import CdEDataset, initialise_metric, compute_metrics, initialise_trainer, train_n_save_model
from prepare_for_training import prep_for_dataset

sys.path.append("..")
from corpus.corpus_reader import CorpusReader


parser = argparse.ArgumentParser(description='Run the pipeline in order to train a specified linear model using the specified features.')
parser.add_argument('trunc', type=str,
                    help='specify how long documents should be truncated (beginning|end|middle)')
parser.add_argument('model', type=str,
                    help='specify the pretrained transformer model')
parser.add_argument('out', type=str,
                    help='specify output directory')
parser.add_argument('save', type=str,
                    help='specify where and under which name to save the model')

args = parser.parse_args()


which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']


with open('/projekte/semrel/WORK-AREA/Users/laura/train_dev_test_split.json', 'r') as jsn:
    split_dict = json.load(jsn)

# filter train data for specified labels (countries)
split_dict_filtered = {key: value for key, value in split_dict.items() if key in which_country}

train_dict = split_dict_filtered['train']
dev_dict = split_dict_filtered['dev']

# filter_punct=True?
cr = CorpusReader('/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS', which_country, 'pars', lower=True)

data = cr.data

# get tokenised data and corresponding targets
tokenised_train_text, train_targets_int = prep_for_dataset(train_dict, args.trunc, args.model, which_country)
tokenised_dev_text, dev_targets_int = prep_for_dataset(dev_dict, args.trunc, args.model, which_country)

# transform features and targets to huggingface dataset
corpus_del_espanol_train = CdEDataset(tokenised_train_text, train_targets_int)
corpus_del_espanol_dev = CdEDataset(tokenised_dev_text, dev_targets_int)

# initialise the trainer
dialect_classification_trainer = initialise_trainer(args.out, args.model, corpus_del_espanol_train, corpus_del_espanol_dev)

# train and save the model
train_n_save_model(dialect_classification_trainer, args.save)