import numpy as np
from typing import Tuple, Union
from random import shuffle
from scipy.sparse import spmatrix, csr_matrix, hstack, vstack
from implement_transformer_model import tokenise_data


def load_sparse_csr(filename: str) -> spmatrix:
    # helper fnction to load the sparse matrix again
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')

    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def concatenate_features(ngram_features: Union[spmatrix, None], tailored_features: dict, ngram_indices: list, set_indices: dict, which_features: list=[], concat: bool=False) -> Tuple[spmatrix, list, list]:
    # concatenate the ngram features and the tailored features
    # transforming them into a sparse matrix

    all_features_array, targets, idx_list = False, [], []
    for label, indices in set_indices.items():
        # add indices and targets to the appropriate lists
        idx_list.extend(indices)
        targets.extend([label]*len(indices))
        for idx in indices:
            if which_features:
                idx_feature_array = []
                # first build an array with the specified tailored features
                for feature in which_features:
                    idx_feature_array.extend(tailored_features[label][idx][feature])

            if all_features_array:
                if concat:
                    # first combine ngram features and tailored features
                    comb_features = hstack((ngram_features[ngram_indices.index(idx), :], csr_matrix([idx_feature_array])))
                    # then add them to the other feature vectors
                    all_features_array = vstack([all_features_array, comb_features])
                elif which_features:
                    # add the tailored feature vector to the other feature vectors and transform new vector to sparse matrix
                    all_features_array = vstack([all_features_array, csr_matrix([idx_feature_array])])
                else:
                    # add the ngram feature vector to the other feature vectors
                    all_features_array = vstack([all_features_array, ngram_features[ngram_indices.index(idx), :]])
            else:
                if concat:
                    # initialise all_feature_array by combining the two different feature types
                    all_features_array = hstack((ngram_features[ngram_indices.index(idx), :], csr_matrix([idx_feature_array])))
                elif which_features:
                    # initialise all_feature_array with the first feature vector (transformed to a sparse matrix)
                    all_features_array = csr_matrix([idx_feature_array])
                else:
                    # initialise all_feature_array with the first ngram feature vector
                    all_features_array = ngram_features[ngram_indices.index(idx), :]

    return all_features_array, targets, idx_list


# def concatenate_features(ngram_features: spmatrix, tailored_features: dict, ngram_indices: list, set_indices: dict, which_features: list) -> Tuple[spmatrix, list, list]:
#     # concatenate the ngram features and the tailored features
#     # transforming them into a sparse matrix
# 
#     all_features_array, targets, idx_list = False, [], []
#     for label, indices in set_indices.items():
#         # add indices and targets to the appropriate lists
#         idx_list.extend(indices)
#         targets.extend([label]*len(indices))
#         for idx in indices:
#             idx_feature_array = []
#             # first build an array with the specified tailored features
#             for feature in which_features:
#                 idx_feature_array.extend(tailored_features[label][idx][feature])
# 
#             if all_features_array:
#                 # first combine ngram features and tailored features
#                 comb_features = hstack((ngram_features[ngram_indices.index(idx), :], csr_matrix([idx_feature_array])))
#                 # then add them to the other feature vectors
#                 all_features_array = vstack([all_features_array, comb_features])
#             else:
#                 # initialise all_feature_array by combining th two different feature types
#                 all_features_array = hstack((ngram_features[ngram_indices.index(idx), :], csr_matrix([idx_feature_array])))
# 
#     return all_features_array, targets, idx_list
            

def shuffle_data(all_features_array: spmatrix, targets: list, idx_list: list) -> Tuple[spmatrix, list, list]:
    # shuffle the data (same order for all three arrays)

    # gets the number of rows
    feature_indices = np.arange(all_features_array.shape[0]) 

    # zip the arrays so that they can be shuffled in the same order
    zipped_arrays = zip(feature_indices, targets, idx_list)
    shuffle(zipped_arrays)

    # unzip the arrays
    shuffled_feature_indices, shuffled_targets, shuffled_idx_list = zip(*zipped_arrays)

    # obtain the shuffled feature matrix by ordering it using the shuffled indices
    shuffled_features_array = all_features_array[list(shuffled_feature_indices)] 

    return shuffled_features_array, shuffled_targets, shuffled_idx_list


# --------------------------------------------------------
#                   TRANSFORMERS
# --------------------------------------------------------

def truncate_long_documents(text_data: list, trunc_where: str) -> list:
    # truncate documents that exceed the maximum length of 512 tokens

    if trunc_where == 'beginning':
        truncated_data = [' '.join(document[:512]) for document in text_data]
    elif trunc_where == 'end':
        truncated_data = [' '.join(document[-512:]) for document in text_data]
    elif trunc_where == 'middle':
        truncated_data = [' '.join(document[:128] + document[-382:]) for document in text_data]

    return truncated_data


def transform_str_to_int_labels(labels: list, which_country: list) -> list:
    # return the target labels as an index so that they can be used for training with a transformer

    str_to_int = {country: idx for idx, country in enumerate(which_country)}

    int_labels = [str_to_int[label] for label in labels]

    return int_labels


def prep_for_dataset(set_dict: dict, trunc: str, model: str, which_country: list) -> Tuple[list, list]:

    set_text, set_targets, set_indices = [], [], []
    for country, indices in set_dict.items():
        set_targets.extend([country]*len(indices))
        set_indices.extend(indices)
        for idx, idx_data in indices.items():
            # append first inner list which is the list of the document's token
            set_text.append(idx_data[0])
    
    # truncate long documents
    trunc_set_text = truncate_long_documents(set_text, trunc)
    
    # transform str targets to integers
    set_targets_int = transform_str_to_int_labels(set_targets, which_country)
    
    # tokenise data
    tokenised_set_text = tokenise_data(trunc_set_text, model)

    return tokenised_set_text, set_targets_int