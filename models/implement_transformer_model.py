import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import torch, evaluate, time
import numpy as np
from transformers import AutoModelForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
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


def initialise_metric(metric_choice: str):

    metric = evaluate.load(metric_choice)

    return metric


def compute_metrics(eval_pred):
    # from https://huggingface.co/docs/transformers/training

    logits, labels = eval_pred

    metric = initialise_metric('accuracy')

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


def initialise_model(model_name: str, num_labels: int=20) -> AutoModelForSequenceClassification:

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    return model


def initialise_trainer(output_dir: str, model: AutoModelForSequenceClassification, train_dataset: CdEDataset, val_dataset: CdEDataset) -> Trainer:

    torch.cuda.empty_cache()
    
    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        num_train_epochs=5,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy='epoch'
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        compute_metrics=compute_metrics
    ) 
    
    print(trainer.args.device)


    return trainer


def train_n_save_model(trainer, save_dir: str) -> None:
        
    print('Start training:')
    start = time.time()
    trainer.train()
    end = time.time()
    print('Training took {} seconds.'.format(end - start))

    print('Start evaluation:')
    start = time.time()
    eval_results = trainer.evaluate()
    print(eval_results)
    end = time.time()
    print('Evaluation took {} seconds.'.format(end - start))

    trainer.save_model(save_dir)