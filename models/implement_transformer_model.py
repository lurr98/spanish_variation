import torch, evaluate
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer, Trainer, TrainingArguments
from typing import Sequence


# TODO: find appropriate type hints

class CdEDataset(torch.utils.data.Dataset):
    # build data set from data in order to train a huggingface model on it

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.features.items()}
        item['labels'] = torch.tensor(self.targets[idx])
        return item

    def __len__(self):
        return len(self.targets)
    

def tokenise_data(text_data: list, model_name: str) -> Sequence:

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    
    encodings = tokenizer(text_data)
    # TODO: Important: find a solution for maximum length problem!

    return encodings


def initialise_metric(metric_choice: str):

    metric = evaluate.load(metric_choice)

    return metric


def compute_metrics(eval_pred, metric):
    # from https://huggingface.co/docs/transformers/training

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)



def initialise_trainer(output_dir: str, model_name: str, train_dataset: CdEDataset, val_dataset: CdEDataset):

    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy='epoch'
    )

    model = BertForMaskedLM.from_pretrained(model_name)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        compute_metrics=compute_metrics
    )

    return trainer


def train_n_save_model(trainer, save_dir: str) -> None:
        
    trainer.train()

    trainer.save_model(save_dir)