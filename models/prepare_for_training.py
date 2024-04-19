import torch
import numpy as np
from typing import Tuple
from random import shuffle
from scipy.sparse import spmatrix, csr_matrix, hstack, vstack


def concatenate_features(ngram_features: spmatrix, tailored_features: dict, ngram_indices: list, set_indices: dict, which_features: list) -> Tuple[spmatrix, list, list]:
    # concatenate the ngram features and the tailored features
    # transforming them into a sparse matrix

    all_features_array, targets, idx_list = False, [], []
    for label, indices in set_indices.items():
        # add indices and targets to the appropriate lists
        idx_list.extend(indices)
        targets.extend([label]*len(indices))
        for idx in indices:
            idx_feature_array = []
            # first build an array with the specified tailored features
            for feature in which_features:
                idx_feature_array.extend(tailored_features[label][idx][feature])

            if all_features_array:
                # first combine ngram features and tailored features
                comb_features = hstack((ngram_features[ngram_indices.index(idx), :], csr_matrix([idx_feature_array])))
                # then add them to the other feature vectors
                all_features_array = vstack([all_features_array, comb_features])
            else:
                # initialise all_feature_array by combining th two different feature types
                all_features_array = hstack((ngram_features[ngram_indices.index(idx), :], csr_matrix([idx_feature_array])))

    return all_features_array, targets, idx_list
            

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
            