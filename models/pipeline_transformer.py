import json, sys, argparse, os, torch
from datetime import date

os.environ['HF_HOME'] = '/projekte/semrel/WORK-AREA/Users/laura/transformer_cache/'
torch.cuda.empty_cache()


from implement_transformer_model import CdEDataset, initialise_trainer, train_n_save_model, initialise_model
from prepare_for_training import prep_for_dataset

sys.path.append("..")
from corpus.corpus_reader import CorpusReader
from basics import initialise_tokeniser

parser = argparse.ArgumentParser(description='Run the pipeline in order to train a specified linear model using the specified features.')
parser.add_argument('model', type=str,
                    help='specify the pretrained transformer model')
# dccuchile/bert-base-spanish-wwm-cased
parser.add_argument('out', type=str,
                    help='specify output directory')
parser.add_argument('save', type=str,
                    help='specify where and under which name to save the model')
parser.add_argument('-group', action='store_true',
                    help='state whether to group the targets by broader region')
parser.add_argument('-nones', action='store_true',
                    help='state whether to filter out named entities')

args = parser.parse_args()


which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']
# which_country = ['BO']


if args.group:
    dict_name = '/projekte/semrel/WORK-AREA/Users/laura/data_split/indices_targets_tdt_split_080101_balanced_grouped.json'
else:
    dict_name = '/projekte/semrel/WORK-AREA/Users/laura/data_split/tdt_split_080101_balanced.json'

# with open('/projekte/semrel/WORK-AREA/Users/laura/toy_train_dev_test_split.json', 'r') as jsn:
with open(dict_name, 'r') as jsn:
    split_dict = json.load(jsn)

# filter train data for specified labels (countries)
split_dict_filtered = {tdt: {label: value for label, value in labels.items() if label in which_country} for tdt, labels in split_dict.items()}

train_dict = split_dict_filtered['train']
dev_dict = split_dict_filtered['dev']

cr = CorpusReader('/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS', which_country, filter_punct=True, filter_digits=True, filter_nes=False, lower=True, split_data=False, sub_sample=False)
# cr = CorpusReader('/projekte/semrel/WORK-AREA/Users/laura/toy_corpus', which_country, filter_punct=True, filter_digits=True, lower=True, split_data=False, sub_sample=False)

data = cr.data

if args.group:
    which_country = ['ANT', 'MCA', 'GC', 'CV', 'EP', 'AU', 'ES', 'MX', 'CL', 'PY']

tokeniser = initialise_tokeniser(args.model)
# get tokenised data and corresponding targets
tokenised_train_text, train_targets_int = prep_for_dataset(train_dict, data, tokeniser, which_country, group=args.group, nones=args.nones)
tokenised_dev_text, dev_targets_int = prep_for_dataset(dev_dict, data, tokeniser, which_country, group=args.group, nones=args.nones)

# transform features and targets to huggingface dataset
corpus_del_espanol_train = CdEDataset(tokenised_train_text, train_targets_int)
corpus_del_espanol_dev = CdEDataset(tokenised_dev_text, dev_targets_int)

# initialise model and set device "cuda"
model = initialise_model(args.model, len(which_country))

# initialise the trainer
dialect_classification_trainer = initialise_trainer(args.out, model, corpus_del_espanol_train, corpus_del_espanol_dev)
print(dialect_classification_trainer.args.device)

# train and save the model
train_n_save_model(dialect_classification_trainer, args.save)