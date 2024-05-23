import json, time, sys
import numpy as np
from typing import Tuple, Union
from random import shuffle
from scipy.sparse import spmatrix, csr_matrix, hstack, vstack

sys.path.append("..")
from basics import load_sparse_csr, tokenise_data


def concatenate_features(split_type: str, ngram_features: Union[spmatrix, None], tailored_features: dict, ngram_indices: list, set_indices: dict, which_country: list, which_features: list=[], concat: bool=False, shuffle: bool=False) -> Tuple[spmatrix, list, list]:
    # concatenate the ngram features and the tailored features
    # transforming them into a sparse matrix

    def concat_helper(idx: int, label: str, all_features_array: Union[spmatrix, list], ngram_features: Union[spmatrix, None], ngram_indices: list, tailored_features: dict, which_features: list) -> spmatrix:
        
        if which_features:
            idx_feature_array = []
            # first build an array with the specified tailored features
            for feature in which_features:
                idx_feature_array.extend(tailored_features[label][idx][feature])

        if not isinstance(all_features_array, list):
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

        return all_features_array


    all_features_array, targets, idx_list = [], [], []
    if shuffle:
        for label in which_country:
            print('Working on label {}'.format(label))
            # add indices and targets to the appropriate lists
            idx_list.extend(set_indices[label])
            targets.extend([label]*len(set_indices[label]))
            for i, idx in enumerate(set_indices[label]):
                if i % 1000 == 0:
                    time.sleep(0.001)
                    print('Working on document {} out of {}'.format(i, len(set_indices[label])))
                all_features_array = concat_helper(idx, label, all_features_array, ngram_features, ngram_indices, tailored_features, which_features)

    else:
        # if the data is not supposed to be shuffled that means that there already exist shuffled features and therefore their indices
        # in order to make the different feature types more comparable, I'll use the already shuffled indices and sort the data accordingly
        # this way, the features are all "shuffled" in the same way
        # note that targets remains empty in this case (they are saved alongside the indices)
        with open('/projekte/semrel/WORK-AREA/Users/laura/indices_targets_tdt_split_080101_balanced.json', 'r') as jsn:
            indices = json.load(jsn)

        idx_list = indices[split_type]['indices']
        target_list = indices[split_type]['targets']
        for i, idx in enumerate(idx_list):
            if i % 1000 == 0:
                time.sleep(0.001)
                print('Working on document {} out of {}'.format(i, len(idx_list)))
            all_features_array = concat_helper(idx, target_list[i], all_features_array, ngram_features, ngram_indices, tailored_features, which_features)

    return all_features_array, targets, idx_list


def shuffle_data(all_features_array: spmatrix, targets: list, idx_list: list) -> Tuple[spmatrix, list, list]:
    # shuffle the data (same order for all three arrays)

    # gets the number of rows
    feature_indices = np.arange(all_features_array.shape[0]) 

    # zip the arrays so that they can be shuffled in the same order
    zipped_arrays = list(zip(feature_indices, targets, idx_list))
    shuffle(zipped_arrays)

    # unzip the arrays
    shuffled_feature_indices, shuffled_targets, shuffled_idx_list = zip(*zipped_arrays)

    # obtain the shuffled feature matrix by ordering it using the shuffled indices
    shuffled_features_array = all_features_array[list(shuffled_feature_indices)] 

    return shuffled_features_array, shuffled_targets, shuffled_idx_list



def prepare_data_full(data_split: str, split_type: str, which_country: list, feature_type: str, which_features: list, matrix_name: str='none', shuffle: bool=False) -> Tuple[spmatrix, list, list]:

    # open the file which stores the indices grouped by the set they belong to
    with open(data_split, 'r') as jsn:
    # with open('/projekte/semrel/WORK-AREA/Users/laura/toy_train_dev_test_split.json', 'r') as jsn:
        split_dict = json.load(jsn)

    # filter train data for specified labels (countries)
    split_dict = {key: value for key, value in split_dict[split_type].items() if key in which_country}

    if feature_type in ['tailored', 'both']:
        # load the tailored feature vectors
        start = time.time()
        with open('/projekte/semrel/WORK-AREA/Users/laura/tailored_features/feature_dict.json', 'r') as jsn:
            features = json.load(jsn)
        end = time.time()
        print('Loading tailored feature dict took {} seconds.'.format(end - start))

    if feature_type in ['ngrams', 'both']:
        # load the indices corresponding to the ngram feature vectors
        with open('/projekte/semrel/WORK-AREA/Users/laura/ngram_features/ngram_frequencies_indices_feature_names_counts.json', 'r') as jsn:
            ngram_indices = json.load(jsn)

        # load the ngram feature vectors
        start = time.time()
        ngrams = load_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/{}'.format(matrix_name))
        end = time.time()
        print('Loading ngram features took {} seconds.'.format(end - start))

    start = time.time()
    # check which features should be used
    if feature_type == 'ngrams':
        # obtain the ngram features of the train set and concatenate ngram features and tailored features
        split_features, split_targets, split_indices = concatenate_features(split_type, ngrams, {}, ngram_indices['indices'], split_dict, which_country, shuffle=shuffle)
    elif feature_type == 'tailored':
        print('tailored')
        # obtain the tailored features of the train set and concatenate ngram features and tailored features
        split_features, split_targets, split_indices = concatenate_features(split_type, None, features, [], split_dict, which_country, which_features, shuffle=shuffle)
    elif feature_type == 'both':
        # sort out redundant features from tailored features
        # TODO: maybe ask supervisors
        # which_features = [feature for feature in which_features if not feature in ['voseo', 'clitic_pronouns']]
        # obtain the ngram and tailored features of the train set and concatenate ngram features and tailored features
        split_features, split_targets, split_indices = concatenate_features(split_type, ngrams, features, ngram_indices['indices'], split_dict, which_country, which_features, concat=True, shuffle=shuffle)
    end = time.time()
    print('Concatenation took {} seconds.'.format(end - start))

    if shuffle:
        # shuffle the train data
        s_split_features, s_split_targets, s_split_indices = shuffle_data(split_features, split_targets, split_indices)

        return s_split_features, s_split_targets, s_split_indices
    else:
        # return the data without shuffling
        # since it is sorted corresponding to already shuffled indices
        return split_features, split_targets, split_indices


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


def transform_str_to_int_labels(labels: list, which_country: list, reverse: bool=False) -> list:
    # return the target labels as an index so that they can be used for training with a transformer

    if reverse:
        int_to_str = {idx: country for idx, country in enumerate(which_country)}

        new_labels = [int_to_str[int(label)] for label in labels]
    else:
        str_to_int = {country: idx for idx, country in enumerate(which_country)}

        new_labels = [str_to_int[label] for label in labels]

    return new_labels


def prep_for_dataset(set_dict: dict, text_dict: dict, model: str, which_country: list, batch: bool=False) -> Tuple[list, list]:

    set_text, set_targets = [], []
    for country, indices in set_dict.items():
        set_targets.extend([country]*len(indices))
        # set_indices.extend(indices)
        # for idx, idx_data in text_dict[country].items():
        for i, idx in enumerate(indices):
            if i % 100 == 0:
                time.sleep(0.01)
            try:
                # append first inner list which is the list of the document's token
                set_text.append(' '.join(text_dict[country][idx][0]))
            except KeyError as e:
                print(e)
                set_targets = set_targets[:-1]
    
    # # truncate long documents
    # trunc_set_text = truncate_long_documents(set_text, trunc)

    # shuffle data
    # zip the arrays so that they can be shuffled in the same order
    zipped_arrays = list(zip(set_text, set_targets))
    shuffle(zipped_arrays)

    # unzip the arrays
    shuffled_set_text, shuffled_set_targets = zip(*zipped_arrays)
    
    # transform str targets to integers
    set_targets_int = transform_str_to_int_labels(shuffled_set_targets, which_country)
    
    if batch:
        batches_text = []
        j = 0
        for i in range(10, len(shuffled_set_text), 10):
            batches_text.append(tokenise_data(shuffled_set_text[max(0, i-10):i], model))
            j = i
    
        print(j)
        print(len(shuffled_set_text))
        if len(shuffled_set_text) != j:
            batches_text.append(tokenise_data(shuffled_set_text[j:len(shuffled_set_text)], model))

        return batches_text, set_targets_int
    # tokenise data
    tokenised_set_text = tokenise_data(shuffled_set_text, model)

    return tokenised_set_text, set_targets_int