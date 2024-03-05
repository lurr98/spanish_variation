#!/usr/bin/env python3
"""
Author: Laura Zeidler
Last changed: 13.02.2024

**description**
"""
from collections import Counter
from nltk import ngrams, FreqDist
from typing import Type
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import fisher_exact
import numpy as np
import pandas as pd


def get_all_text(class_data: dict, gram_type: str) -> list:
    # merge text from all documents in one list
    
    type_dict = {'token': 0, 'lemma': 1, 'pos': 2}

    all_text = []
    for idx, text in class_data.items():
        all_text.extend(text[type_dict[gram_type]])
    
    return all_text


def get_ngram_frequencies(class_data: dict, n: int, gram_type: str) -> Type[Counter]:
    # get the n-gram frequencies for the class
    # gram_type specifies which type of data (token, lemma or pos) should be chosen
    
    all_text = get_all_text(class_data, gram_type)

    ngram_counts = Counter(ngrams(all_text, n))
    return ngram_counts


def length_of_paragraphs(class_data: dict) -> float:
    # get average length of paragraphs from the class data 

    total_length = 0

    for idx, text in class_data.items():
        # add the length of the paragraph (could also be text[1], text[2])
        total_length += len(text[0])

    return total_length / len(list(class_data.keys()))


def most_frequent_ngram(class_data: dict, n: int, k: int, gram_type: str) -> list:
    # get k most frequent n-grams for the class
    # gram_type specifies which type of data (token, lemma or pos) should be chosen
    ngram_counts = get_ngram_frequencies(class_data, n, gram_type)

    return ngram_counts.most_common(k)


# def chi_square():

# def wilcoxon_ranks_test():
#     ## data has to be divided into same sized samples (too much work?)


def mutual_information(class_data: dict, n: int, out_of_class_data: dict, gram_type: str) -> dict:
    # statistic measure which states how much information a word provides about a document
    # MI is invalid for low counts, threshold = 5
    # TODO: 5 in target doc or overall? for the moment in target doc

    # get ngram frequency for the class
    class_ngrams = get_ngram_frequencies(class_data, n, gram_type)

    merged_ooc_dict = {}
    # merge the dictionaries containing the id as key and the text as value of all out-of-class labels
    for label, data in out_of_class_data.items():
        merged_ooc_dict.update(data)

    # get ngram frequencies for anything but the class
    ooc_ngrams = get_ngram_frequencies(merged_ooc_dict, n, gram_type)

    mutual_information = {}

    for ngram in list(class_ngrams.keys()):
        ngram_count = class_ngrams[ngram]
        if ngram_count >= 5:
            first_formula = ngram_count / (ngram_count + (sum(class_ngrams.values()) - ngram_count))
            second_formula = (sum(class_ngrams.values()) + sum(ooc_ngrams.values())) / (ngram_count + ooc_ngrams[ngram])
            mutual_information[ngram] = math.log2(first_formula * second_formula)

    return mutual_information


def log_likelihood(class_data: dict, n: int, out_of_class_data: dict, gram_type: str) -> dict:
    # according to Kilgarriff

    # get ngram frequency for the class
    class_ngrams = get_ngram_frequencies(class_data, n, gram_type)

    merged_ooc_dict = {}
    # merge the dictionaries containing the id as key and the text as value of all out-of-class labels
    for label, data in out_of_class_data.items():
        merged_ooc_dict.update(data)

    # get ngram frequencies for anything but the class
    ooc_ngrams = get_ngram_frequencies(merged_ooc_dict, n, gram_type)

    log_likelihood = {}

    for ngram in list(class_ngrams.keys()):
        a = class_ngrams[ngram] # w in X
        b = ooc_ngrams[ngram]   # w in Y
        c = sum(class_ngrams.values()) - a  # not w in X
        d = sum(ooc_ngrams.values()) - b    # not w in Y

        # define parts of formula so final formula is readable
        a_plus_b = (a + b)*math.log(a + b)
        a_plus_c = (a + c)*math.log(a + c)
        b_plus_d = (b + d)*math.log(b + d)
        c_plus_d = (c + d)*math.log(c + d)
        big_n = (a + b + c + d)*math.log(a + b + c + d)
        log_likelihood[ngram] = 2*(a*math.log(a) + b*math.log(b) + c*math.log(c) + d*math.log(d) - a_plus_b - a_plus_c - b_plus_d - c_plus_d + big_n)

    return log_likelihood


# TODO: think about whetherit makes sense to compare these stats between all classes in pairs


def tf_idf(all_data: dict, n: int, gram_type: str, k: int) -> dict:
    # statistic measure which states how characteristic a term is for a document

    label_indices, all_texts = [], []
    for label, data in all_data.items():
        label_indices.append(label)
        all_texts.append(' '.join(get_all_text(data, gram_type)))

    vectorizer = TfidfVectorizer(ngram_range=(n,n))
    tf_idf = vectorizer.fit_transform(all_texts)
     
    # create a pandas data frame to access the data
    df = pd.DataFrame(tf_idf.toarray(), columns = vectorizer.get_feature_names_out()).T

    tf_idf_dict = {}
    # create a dictionary of the following format:
    # {'PA': (['most', 'characteristic', 'tokens'], [tfidf_score_most, tfidf_score_characteristic, tfidf_score_tokens]), …}
    for i, label in enumerate(label_indices):
        tf_idf_dict[label] = (list(df.nlargest(k, i).index), list(df.nlargest(k, i)[i]))

    return tf_idf_dict


def fishers_exact_text(class_data: dict, n: int, out_of_class_data: dict, gram_type: str) -> dict:
    # statistical test that determines if two category variables have a significant relationship
    # in this case the variables are is_word_w, is_corpus_X
    # contingency table:

    # |      |   X   |   Y   |
    # |------|-------|-------|
    # |  w   |       |       |
    # |------|-------|-------|
    # |not w |       |       |

    # get ngram frequency for the class
    class_ngrams = get_ngram_frequencies(class_data, n, gram_type)

    merged_ooc_dict = {}
    # merge the dictionaries containing the id as key and the text as value of all out-of-class labels
    for label, data in out_of_class_data.items():
        merged_ooc_dict.update(data)

    # get ngram frequencies for anything but the class
    ooc_ngrams = get_ngram_frequencies(merged_ooc_dict, n, gram_type)

    fishers_test = {}
    for ngram in list(class_ngrams.keys()):
        contingency_table = np.array([[class_ngrams[ngram], ooc_ngrams[ngram]], [sum(class_ngrams.values()) - class_ngrams[ngram], sum(ooc_ngrams.values()) - ooc_ngrams[ngram]]])
        # performing fishers exact test on the data 
        odd_ratio, p_value = stats.fisher_exact(contingency_table)
        fishers_test[ngram] = (odd_ratio, p_value)

    return fishers_test