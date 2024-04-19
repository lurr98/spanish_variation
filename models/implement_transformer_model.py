import torch
from transformers import BertForMaskedLM, BertTokenizer
from typing import Sequence


def tokenise_data(text_data: list, model_name: str) -> Sequence:

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    
    encodings = tokenizer(text_data)
    # TODO: Important: find a solution for maximum length problem!

class IMDbDataset(torch.utils.data.Dataset):
    # build data set from data in order to train a huggingface model on it

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.targets)


def transform_str_to_int_labels(labels: list, which_country: list) -> list:
    # return the target labels as an index so that they can be used for training with a transformer

    str_to_int = {country: idx for idx, country in enumerate(which_country)}

    int_labels = [str_to_int[label] for label in labels]

    return int_labels