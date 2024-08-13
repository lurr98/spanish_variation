# Automatic Dialect Classification for Spanish

## Purpose of this Research

This repository contains the code for my Master's thesis which deals with the automatic classification of 20 Spanish dialects using different feature and model types.

In particular, there are two approaches that I followed:

1) I manually selected linguistic characteristics that are distinct for the Spanish dialects that need to be classified. The notes taken during this process can be found in `features/variation_phenomena.md`. The selected features were then counted in the data, the final selection of features as well as a description of their implementation is documented in `features/phenomena_represent.md`. The implementation itself can be found in `features/implement_features.py` and `features/features_utils.py`.

2) I furthermore fine-tuned a pre-trained BERT model, namely the `dccuchile/bert-base-spanish-wwm-cased` model, on the data and compared its performance to the performance of the SVM and DT models.


## Pipeline

### Extracting Features

Both the linguistic features as well as the unigram features can be extracted by simply running the script `features/implement_features.py`. The functions to extract the linguistic features can be found in `features/features_utils.py`.

By running the script, 


### Concatenating Features

Be

### Training Models


### Evaluation 