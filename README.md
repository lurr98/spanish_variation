# Automatic Dialect Classification for Spanish

## Purpose of this Research

This repository contains the code for my Master's thesis which deals with the automatic classification of 20 Spanish dialects using different feature and model types.

In particular, there are two approaches that I followed:

1) I manually selected linguistic characteristics that are distinct for the Spanish dialects that need to be classified. The notes taken during this process can be found in `features/variation_phenomena.md`. The selected features were then counted in the data, the final selection of features as well as a description of their implementation is documented in `features/phenomena_represent.md`. 

2) I furthermore fine-tuned a pre-trained BERT model, namely the `dccuchile/bert-base-spanish-wwm-cased` model, on the data and compared its performance to the performance of the SVM and DT models.


## Pipeline

### Extracting Features

Both the linguistic features as well as the unigram features can be extracted by simply running the script `features/implement_features.py`. The functions to extract the linguistic features can be found in `features/features_utils.py`.

By running the script the Corpus del Español is read and the features are extracted from the data. If the data has not been split into train, development and test set, this can be done in the same step by setting the corresponding flag (run `python3 implement_features.py -h` for more information on the arguments of this script).

The extracted tailored features are then stored in a JSON file while the unigram features are stored as a sparse matrix file.

### Concatenating Features

Before training the models, the features of the individual documents need to be concatenated to a matrix of feature vectors based on the documents that are included in the respective data set (train, dev and test). To do this, the script `models/pipeline_linear_models.py` has to be run with the second argument `none` (run `python3 pipeline_linear_models.py -h` for more information on the arguments of this script).

### Training Models

To train the traditional machine learning models, one can run the script `models/pipeline_linear_models.py` (run `python3 pipeline_linear_models.py -h` for more information on the arguments of this script).

To fine-tune the BERT model, one can run the script `models/pipeline_transformer.py` (run `python3 pipeline_transformer.py -h` for more information on the arguments of this script).

The trained models will be stored at the given output path for later use.

### Evaluation 

To evaluate the model, one can either run the script `evaluation/evaluation_pipeline.py` considering the corresponding arguments and flags or simply run the bash script `evaluation/evaluate_models.sh` uncommenting the lines that evaluate the desired models.