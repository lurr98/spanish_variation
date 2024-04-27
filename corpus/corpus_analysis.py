#!/usr/bin/env python3
"""
Author: Laura Zeidler
Last changed: 13.02.2024

**description**
"""
import math, time, json, nltk, re
from corpus_reader import CorpusReader
from collections import Counter
from nltk import ngrams, FreqDist
from nltk.corpus import stopwords
from typing import Type, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import fisher_exact
from scipy import sparse
from corpus_similarity import Similarity
import numpy as np
import pandas as pd

nltk.download('stopwords', download_dir='/projekte/semrel/WORK-AREA/Users/laura/')
nltk.data.path.append('/projekte/semrel/WORK-AREA/Users/laura/')


# global variable so that this doesn't have to be passed around all the time
with open('POS_related/inverted_POS_tags.json', 'r') as jsn:
    POS_mapping = json.load(jsn)


def map_POS_tags(tag: str, POS_dict: dict) -> str:
    # map the POS tag in the corpus to the broader categories that I defined

    return POS_dict[tag]


def get_all_text(class_data: dict, gram_type: str, filter_for: list=[]) -> list:
    # merge text from all documents in one list

    def substitute_null_char(text):

        sub_indices = [idx for idx, value in enumerate(text) if value == '\x00']
        for idx in sub_indices:
            text[idx] = 'x00'

        return text
    
    type_dict = {'token': 0, 'lemma': 1, 'pos': 2}

    all_text = []
    for idx, text in class_data.items():
        # filter for specific POS tags
        if filter_for:
            try:
                all_text.extend([token for i, token in enumerate(text[type_dict[gram_type]]) if POS_mapping[text[2][i]] in filter_for])
            except KeyError as e:
                print('Key Error: {}'.format(e))
                all_text.extend([token for i, token in enumerate(text[type_dict[gram_type]]) if POS_mapping[substitute_null_char(text[2])[i]] in filter_for])

        # map the corpus POS tags to our simplified tag set
        elif gram_type == 'pos':
            try:
                all_text.extend([POS_mapping[pos] for pos in text[type_dict[gram_type]]])
            except KeyError as e:
                print('Key Error: {}'.format(e))
                sub = substitute_null_char(text[type_dict[gram_type]])
                all_text.extend([POS_mapping[pos] for pos in sub])

        else:
            all_text.extend(text[type_dict[gram_type]])
    
    return all_text


def get_ngram_frequencies(class_data: dict, n: int, gram_type: str, filter_for: list=[], keep_text: bool=False) -> Type[Counter] | Tuple[Type[Counter], list]:
    # get the n-gram frequencies for the class
    # gram_type specifies which type of data (token, lemma or pos) should be chosen
    
    all_text = get_all_text(class_data, gram_type, filter_for)

    ngram_counts = Counter(ngrams(all_text, n))

    if keep_text:
        return (ngram_counts, all_text)
    else:
        return ngram_counts


def length_of_paragraphs(class_data: dict) -> float:
    # get average length of paragraphs from the class data 

    total_length = 0

    for idx, text in class_data.items():
        # add the length of the paragraph (could also be text[1], text[2])
        total_length += len(text[0])

    return total_length / len(list(class_data.keys()))


def most_frequent_ngram(ngram_counts: dict, k: int) -> list:
    # get k most frequent n-grams for the class
    # gram_type specifies which type of data (token, lemma or pos) should be chosen

    most_common = ngram_counts.most_common(k)

    return [(' '.join(ngram[0]), most_common[i][1]) for i, ngram in enumerate(most_common)] 


def mutual_information(class_ngrams: dict, ooc_ngrams: dict) -> dict:
    # statistic measure which states how much information a word provides about a document
    # MI is invalid for low counts, threshold = 5
    # TODO: 5 in target doc or overall? for the moment in target doc

    mutual_information = {}

    for i, ngram in enumerate(list(class_ngrams.keys())):
        # if i % 10000 == 0:
        #     print('Token {} of {}'.format(i, len(list(class_ngrams.keys()))))
        ngram_count = class_ngrams[ngram]
        if ngram_count >= 5:
            sum_all_class_data = sum(class_ngrams.values())
            sum_all_ooc_data = sum(ooc_ngrams.values())
            first_formula = ngram_count / (ngram_count + (sum_all_class_data - ngram_count))
            second_formula = (sum_all_class_data + sum_all_ooc_data) / (ngram_count + ooc_ngrams[ngram])
            mutual_information[' '.join(ngram)] = math.log2(first_formula * second_formula)

    return mutual_information


def log_likelihood(class_ngrams: dict, ooc_ngrams: dict) -> dict:
    # according to Kilgarriff

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
        try:
            log_likelihood[' '.join(ngram)] = 2*(a*math.log(a) + b*math.log(b) + c*math.log(c) + d*math.log(d) - a_plus_b - a_plus_c - b_plus_d - c_plus_d + big_n)
        except ValueError as e:
            pass

    return log_likelihood


# TODO: think about whetherit makes sense to compare these stats between all classes in pairs


def tf_idf(all_data: dict, gram_type: str, helper: bool=False, n: int=1, k: int=10) -> Union[Tuple[sparse.spmatrix,list],dict]:
    # statistic measure which states how characteristic a term is for a document

    label_indices, all_texts = [], []
    for label, data in all_data.items():
        label_indices.append(label)
        all_texts.append(' '.join(get_all_text(data, gram_type)))

    # create set of stopwords to remove
    stop_words = stopwords.words('spanish')

    vectorizer = TfidfVectorizer(ngram_range=(n,n), stop_words=stop_words)
    tf_idf = vectorizer.fit_transform(all_texts)

    if helper:
        return (tf_idf, label_indices)
     
    # create a pandas data frame to access the data
    df = pd.DataFrame(tf_idf.toarray(), columns = vectorizer.get_feature_names_out()).T


    tf_idf_dict = {}
    # create a dictionary of the following format:
    # {'PA': (['most', 'characteristic', 'tokens'], [tfidf_score_most, tfidf_score_characteristic, tfidf_score_tokens]), …}
    for i, label in enumerate(label_indices):
        tf_idf_dict[label] = (list(df.nlargest(k, i).index), list(df.nlargest(k, i)[i]))

    return tf_idf_dict


def fishers_exact_text(class_ngrams: dict, ooc_ngrams: dict, log_likelihoods: list) -> dict:
    # statistical test that determines if two category variables have a significant relationship
    # in this case the variables are is_word_w, is_corpus_X
    # contingency table:

    # |      |   X   |   Y   |
    # |------|-------|-------|
    # |  w   |       |       |
    # |------|-------|-------|
    # |not w |       |       |

    fishers_test = {}
    for ngram in list(class_ngrams.keys()):
        contingency_table = np.array([[class_ngrams[ngram], ooc_ngrams[ngram]], [sum(class_ngrams.values()) - class_ngrams[ngram], sum(ooc_ngrams.values()) - ooc_ngrams[ngram]]])
        # performing fishers exact test on the data 
        odd_ratio, p_value = fisher_exact(contingency_table)
        fishers_test[' '.join(ngram)] = (odd_ratio, p_value)

    return fishers_test


# --------------------------------------------------------------------------------------------------------------
# Comparing Corpora
# --------------------------------------------------------------------------------------------------------------

# TODO: this is not efficient!
def spearmans_rho(all_text_classes: list) -> int:
    # measures the similarity between two corpora by means of the spearman's rank correlation coefficient
    # see kilgarriff

    cs = Similarity(language = "spa")
    spearmans_matrix = []

    for i, text_reference in enumerate(all_text_classes):
        spearmans_list = []
        for j, text_compare in enumerate(all_text_classes):
            result = cs.calculate(text_reference, text_compare)
            spearmans_list.append(result)

        spearmans_matrix.append(spearmans_list)

    return spearmans_matrix


def cosine_similarity_tfidf(all_data: dict, gram_type: str, n: int) -> Tuple[np.ndarray, list]:
    # measures the pairwise similarity between a set of corpora using TF-IDF vectors
    # see Introduction to Information Retrieval by Christopher Manning

    tf_idf_scores, label_indices = tf_idf(all_data, gram_type, helper=True, n=n)

    pairwise_similarity = tf_idf_scores * tf_idf_scores.T

    return (pairwise_similarity.toarray().tolist(), label_indices)


if __name__ == "__main__":
    which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']
    # which_country = ['CU', 'PA']

    # initialise dictionary to store results
    stats_dict = {}

    start = time.time()
    cr = CorpusReader('/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS', which_country, 'pars', True)
    # cr = CorpusReader('/projekte/semrel/WORK-AREA/Users/laura/toy_corpus', which_country, 'pars', True)
    end = time.time()
    print('Corpus reader took {} seconds.'.format(end - start))

    with open('/projekte/semrel/WORK-AREA/Users/laura/stats_dict.json', 'r') as jsn:
        stats_dict = json.load(jsn)

    overall_start = time.time()

    # start = time.time()
    # lemma_tf_idf_uni = tf_idf(cr.data, 'lemma', False, 1, 10)
    # lemma_tf_idf_bi = tf_idf(cr.data, 'lemma', False, 2, 10)
    # end = time.time()
    # print('TF-IDF for uni- and bigram lemma took {} seconds.'.format(end - start))

    # start = time.time()
    # pos_tf_idf_bi = tf_idf(cr.data, 'pos', False, 2, 10)
    # pos_tf_idf_tri = tf_idf(cr.data, 'pos', False, 3, 10)
    # end = time.time()
    # print('TF-IDF for bi- and trigram POS took {} seconds.'.format(end - start))

    label_indices, all_text_classes, all_text_classes_token = [], [], []

    # now execute all functions that need the class data
    for label, data in cr.data.items():

        # store order of labels
        label_indices.append(label)

        # create dictionary to store results
        stats_dict[label] = {}

        # # get ngram frequencies for LEMMA
        # lemma_uni_frequencies, all_text_class = get_ngram_frequencies(data, 1, 'lemma', keep_text=True)
        # all_text_classes.append(all_text_class)

        # get ngram frequencies for TOKEN
        token_uni_frequencies, all_text_class_token = get_ngram_frequencies(data, 1, 'token', keep_text=True)
        all_text_classes_token.append(all_text_class_token)
        token_frequencies = {''.join(gram): frequency for gram, frequency in token_uni_frequencies.items()}

        with open('/projekte/semrel/WORK-AREA/Users/laura/ngram_frequency_dict.json', 'w') as jsn:
            json.dump(token_frequencies, jsn)

        # TODO: run this but add spearmans rho!
            
        start = time.time()
        spearmans_rho_token = spearmans_rho(all_text_classes_token)
        end = time.time()
        print('Spearmans rho for token took {} seconds.'.format(end - start))
        print('Spearmans rho matrix using the tokens:')
        print('Order of labels: {}'.format(label_indices))
        print('{}\n-------------------------------------------------\n'.format(np.array(spearmans_rho_token)))

        start = time.time()
        spearmans_rho_lemma = spearmans_rho(all_text_classes)
        end = time.time()
        print('Spearmans rho for lemma took {} seconds.'.format(end - start))
        print('Spearmans rho matrix using the lemmata:')
        print('Order of labels: {}'.format(label_indices))
        print('{}\n-------------------------------------------------\n'.format(np.array(spearmans_rho_lemma)))

        stats_dict['all_countries']['spearmans_rho'] = {'token': {'unigrams': (spearmans_rho_token, label_indices)}, 'lemma': {'unigrams': (spearmans_rho_lemma, label_indices)}}

        with open('/projekte/semrel/WORK-AREA/Users/laura/stats_dict_updated.json', 'w') as jsn:
            json.dump(stats_dict, jsn)

        exit()

        noun_lemma_uni_frequencies = get_ngram_frequencies(data, 1, 'lemma', ['NN', 'NE'])
        verb_lemma_uni_frequencies = get_ngram_frequencies(data, 1, 'lemma', ['VM', 'VC', 'VIF', 'VII', 'VIP', 'VIS', 'VIMP', 'VPP', 'VPS', 'VR', 'VSF', 'VSI', 'VSJ', 'VSP'])

        out_of_class_data = {k: v for k, v in cr.data.items() if k != label}

        merged_ooc_dict = {}
        # merge the dictionaries containing the id as key and the text as value of all out-of-class labels
        for country_label, country_data in out_of_class_data.items():
            merged_ooc_dict.update(country_data)

        # get ngram frequencies for anything but the class
        lemma_ooc_uni_frequencies = get_ngram_frequencies(merged_ooc_dict, 1, 'lemma')

        print('\n-------------------------------------------------\nAnalysing dialect {}\n-------------------------------------------------\n'.format(label))
        av_par_length = length_of_paragraphs(data)
        print('Av. length of paragraphs: {}\n-------------------------------------------------\n'.format(av_par_length))
        stats_dict[label]['average_paragraph_length'] = av_par_length

        start = time.time()
        lemma_uni_most_frequent = most_frequent_ngram(lemma_uni_frequencies, 10)
        end = time.time()
        stats_dict[label]['most_frequent'] = {'lemma': {'unigrams': lemma_uni_most_frequent}}
        print('10 most frequent unigrams took {} seconds.'.format(end - start))
        print('10 most frequent unigrams: {}\n-------------------------------------------------\n'.format(lemma_uni_most_frequent[:10]))

        start = time.time()
        lemma_noun_most_frequent = most_frequent_ngram(noun_lemma_uni_frequencies, 10)
        end = time.time()
        stats_dict[label]['most_frequent']['lemma'] = {'noun_unigrams': lemma_noun_most_frequent}
        print('10 most frequent nouns took {} seconds.'.format(end - start))
        print('10 most frequent nouns: {}\n-------------------------------------------------\n'.format(lemma_noun_most_frequent[:10]))

        start = time.time()
        lemma_verb_most_frequent = most_frequent_ngram(verb_lemma_uni_frequencies, 10)
        end = time.time()
        stats_dict[label]['most_frequent']['lemma'] = {'verb_unigrams': lemma_verb_most_frequent}
        print('10 most frequent verbs took {} seconds.'.format(end - start))
        print('10 most frequent verbs: {}\n-------------------------------------------------\n'.format(lemma_verb_most_frequent[:10]))

        stats_dict[label]['TF-IDF'] = {'lemma': {'unigrams': [(lemma_tf_idf_uni[label][0][i], tf_idf_lemma) for i, tf_idf_lemma in enumerate(lemma_tf_idf_uni[label][0])]}}
        stats_dict[label]['TF-IDF']['lemma']['bigrams'] = [(lemma_tf_idf_bi[label][0][i], tf_idf_lemma) for i, tf_idf_lemma in enumerate(lemma_tf_idf_bi[label][0])]
        print('10 unigrams with highest tf-idf score: {}\n-------------------------------------------------\n'.format(lemma_tf_idf_uni[label][0][:10]))
        print('10 bigrams with highest tf-idf score: {}\n-------------------------------------------------\n'.format(lemma_tf_idf_bi[label][0][:10]))

        start = time.time()
        lemma_uni_mutual_information = sorted(mutual_information(lemma_uni_frequencies, lemma_ooc_uni_frequencies).items(), key=lambda x:x[1])
        end = time.time()
        stats_dict[label]['MI'] = {'lemma': {'unigrams': lemma_uni_mutual_information[-10:]}}
        print('MI took {} seconds.'.format(end - start))
        print('10 unigrams with highest MI score: {}\n-------------------------------------------------\n'.format(lemma_uni_mutual_information[-10:]))

        start = time.time()
        lemma_uni_log_likelihood = sorted(log_likelihood(lemma_uni_frequencies, lemma_ooc_uni_frequencies).items(), key=lambda x:x[1])
        end = time.time()
        stats_dict[label]['LL'] = {'lemma': {'unigrams': lemma_uni_log_likelihood[-10:]}}
        print('LL took {} seconds.'.format(end - start))
        print('10 unigrams with highest LL score: {}\n-------------------------------------------------\n'.format(lemma_uni_log_likelihood[-10:]))

        # start = time.time()
        # lemma_uni_fishers_exact_test = fishers_exact_text(lemma_uni_frequencies, lemma_ooc_uni_frequencies, lemma_uni_log_likelihood[-10:])
        # end = time.time()
        # stats_dict[label]['fishers_exact_test'] = {'lemma': {'unigrams': lemma_uni_fishers_exact_test}}
        # print('Fishers exact test took {} seconds.'.format(end - start))
        # print('Odd ratio and p value for the 3 unigrams ranked highest by LL:\n{}\n{}\n{}\n-------------------------------------------------\n'.format(lemma_uni_fishers_exact_test[lemma_uni_log_likelihood[-1][0]], lemma_uni_fishers_exact_test[lemma_uni_log_likelihood[-2][0]], lemma_uni_fishers_exact_test[lemma_uni_log_likelihood[-3][0]]))

        
        # get ngram frequencies for POS
        
        pos_bi_frequencies = get_ngram_frequencies(data, 2, 'pos')
        pos_tri_frequencies = get_ngram_frequencies(data, 3, 'pos')

        out_of_class_data = {k: v for k, v in cr.data.items() if k != label}

        merged_ooc_dict = {}
        # merge the dictionaries containing the id as key and the text as value of all out-of-class labels
        for country_label, country_data in out_of_class_data.items():
            merged_ooc_dict.update(country_data)

        # get ngram frequencies for anything but the class
        pos_ooc_bi_frequencies = get_ngram_frequencies(merged_ooc_dict, 2, 'pos')
        pos_ooc_tri_frequencies = get_ngram_frequencies(merged_ooc_dict, 3, 'pos')

        start = time.time()
        pos_bi_most_frequent = most_frequent_ngram(pos_bi_frequencies, 10)
        end = time.time()
        stats_dict[label]['most_frequent']['pos'] = {'bigrams': pos_bi_most_frequent}
        print('10 most frequent POS bigrams took {} seconds.'.format(end - start))
        print('10 most frequent POS bigrams: {}\n-------------------------------------------------\n'.format(pos_bi_most_frequent[:10]))

        start = time.time()
        pos_tri_most_frequent = most_frequent_ngram(pos_tri_frequencies, 10)
        end = time.time()
        stats_dict[label]['most_frequent']['pos'] = {'trigrams': pos_tri_most_frequent}
        print('10 most frequent POS trigrams took {} seconds.'.format(end - start))
        print('10 most frequent POS trigrams: {}\n-------------------------------------------------\n'.format(pos_tri_most_frequent[:10]))

        stats_dict[label]['TF-IDF'] = {'pos': {'bigrams': [(pos_tf_idf_bi[label][0][i].upper(), tf_idf_pos) for i, tf_idf_pos in enumerate(pos_tf_idf_bi[label][0])]}}
        stats_dict[label]['TF-IDF']['pos']['trigrams'] = [(pos_tf_idf_tri[label][0][i].upper(), tf_idf_pos) for i, tf_idf_pos in enumerate(pos_tf_idf_tri[label][0])]
        print('10 POS bigrams with highest tf-idf score: {}\n-------------------------------------------------\n'.format(pos_tf_idf_bi[label][0][:10]))
        print('10 POS trigrams with highest tf-idf score: {}\n-------------------------------------------------\n'.format(pos_tf_idf_tri[label][0][:10]))

        start = time.time()
        pos_bi_mutual_information = sorted(mutual_information(pos_bi_frequencies, pos_ooc_bi_frequencies).items(), key=lambda x:x[1])
        end = time.time()
        stats_dict[label]['MI']['pos'] = {'bigrams': pos_bi_mutual_information[-10:]}
        print('MI for POS bigrams took {} seconds.'.format(end - start))
        print('10 POS bigrams with highest MI score: {}\n-------------------------------------------------\n'.format(pos_bi_mutual_information[-10:]))

        start = time.time()
        pos_tri_mutual_information = sorted(mutual_information(pos_tri_frequencies, pos_ooc_tri_frequencies).items(), key=lambda x:x[1])
        end = time.time()
        stats_dict[label]['MI']['pos']['trigrams'] = pos_tri_mutual_information[-10:]
        print('MI for POS trigrams took {} seconds.'.format(end - start))
        print('10 POS trigrams with highest MI score: {}\n-------------------------------------------------\n'.format(pos_tri_mutual_information[-10:]))

        start = time.time()
        pos_bi_log_likelihood = sorted(log_likelihood(pos_bi_frequencies, pos_ooc_bi_frequencies).items(), key=lambda x:x[1])
        end = time.time()
        stats_dict[label]['LL']['pos'] = {'bigrams': pos_bi_log_likelihood[-10:]}
        print('LL for POS bigrams took {} seconds.'.format(end - start))
        print('10 POS bigrams with highest LL score: {}\n-------------------------------------------------\n'.format(pos_bi_log_likelihood[-10:]))

        start = time.time()
        pos_tri_log_likelihood = sorted(log_likelihood(pos_tri_frequencies, pos_ooc_tri_frequencies).items(), key=lambda x:x[1])
        end = time.time()
        stats_dict[label]['LL']['pos'] = {'trigrams': pos_tri_log_likelihood[-10:]}
        print('LL for POS trigrams took {} seconds.'.format(end - start))
        print('10 POS trigrams with highest LL score: {}\n-------------------------------------------------\n'.format(pos_tri_log_likelihood[-10:]))

       # start = time.time()
       # pos_bi_fishers_exact_test = fishers_exact_text(pos_bi_frequencies, pos_ooc_bi_frequencies, pos_bi_log_likelihood[-10:])
       # end = time.time()
       # stats_dict[label]['fishers_exact_test']['pos'] = {'bigrams': pos_bi_fishers_exact_test}
       # print('Fishers exact test for POS bigrams took {} seconds.'.format(end - start))
       # print('Odd ratio and p value for the 3 POS bigrams ranked highest by LL:\n{}\n{}\n{}\n-------------------------------------------------\n'.format(pos_bi_fishers_exact_test[pos_bi_log_likelihood[-1][0]], pos_bi_fishers_exact_test[pos_bi_log_likelihood[-2][0]], pos_bi_fishers_exact_test[pos_bi_log_likelihood[-3][0]]))

       # start = time.time()
       # pos_tri_fishers_exact_test = fishers_exact_text(pos_tri_frequencies, pos_ooc_tri_frequencies, pos_tri_log_likelihood[-10:])
       # end = time.time()
       # stats_dict[label]['fishers_exact_test']['pos'] = {'trigrams': pos_tri_fishers_exact_test}
       # print('Fishers exact test for POS trigrams took {} seconds.'.format(end - start))
       # print('Odd ratio and p value for the 3 POS trigrams ranked highest by LL:\n{}\n{}\n{}\n-------------------------------------------------\n'.format(pos_tri_fishers_exact_test[pos_tri_log_likelihood[-1][0]], pos_tri_fishers_exact_test[pos_tri_log_likelihood[-2][0]], pos_tri_fishers_exact_test[pos_tri_log_likelihood[-3][0]]))


    # execute functions that work on all data
    start = time.time()
    cosine_sim_token, label_indices_token = cosine_similarity_tfidf(cr.data, 'token', 1)
    end = time.time()
    print('Cosine similarity of TF-IDF vectors for token took {} seconds.'.format(end - start))
    print('Cosine similarity matrix using the tokens:')
    print('Order of labels: {}'.format(label_indices_token))
    print('{}\n-------------------------------------------------\n'.format(np.array(cosine_sim_token)))

    start = time.time()
    cosine_sim_lemma, label_indices_lemma = cosine_similarity_tfidf(cr.data, 'lemma', 1)
    end = time.time()
    print('Cosine similarity of TF-IDF vectors for lemma took {} seconds.'.format(end - start))
    print('Cosine similarity matrix using the lemmata:')
    print('Order of labels: {}'.format(label_indices_lemma))
    print('{}\n-------------------------------------------------\n'.format(np.array(cosine_sim_lemma)))

    stats_dict['all_countries'] = {'cosine_similarity_tfidf': {'token': {'unigrams': (cosine_sim_token, label_indices_token)}, 'lemma': {'unigrams': (cosine_sim_lemma, label_indices_lemma)}}}

    # start = time.time()
    # spearmans_rho_token = spearmans_rho(all_text_classes, 'token')
    # end = time.time()
    # print('Spearmans rho for token took {} seconds.'.format(end - start))
    # print('Spearmans rho matrix using the tokens:')
    # print('Order of labels: {}'.format(label_indices))
    # print('{}\n-------------------------------------------------\n'.format(np.array(spearmans_rho_token)))

    start = time.time()
    spearmans_rho_lemma = spearmans_rho(all_text_classes)
    end = time.time()
    print('Spearmans rho for lemma took {} seconds.'.format(end - start))
    print('Spearmans rho matrix using the lemmata:')
    print('Order of labels: {}'.format(label_indices))
    print('{}\n-------------------------------------------------\n'.format(np.array(spearmans_rho_lemma)))

    stats_dict['all_countries']['spearmans_rho'] = {'token': {'unigrams': (spearmans_rho_token, label_indices)}, 'lemma': {'unigrams': (spearmans_rho_lemma, label_indices)}}

    overall_end = time.time()

    print('Data Analysis took {} seconds'.format(overall_end-overall_start))

    print(stats_dict)

    with open('/projekte/semrel/WORK-AREA/Users/laura/test_stats_dict.json', 'w') as jsn:
        json.dump(stats_dict, jsn)