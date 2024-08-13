#!/usr/bin/env python3
"""
Author: Laura Zeidler

**description**
"""

import json, time, sys, spacy
import features_utils
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, spmatrix
from typing import Sequence, Union
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append("..")
from basics import save_sparse_csr
from corpus.corpus_reader import CorpusReader


class NgramFeatureExtractor:
    # define the ngram extractor class that will be used by the models to build the feature vectors

    def __init__(self, data, which_country, cut_off=2, stop_words=None, tf=True):

        # saved_args = locals()
        # print('NgramFeatureExtractor was initialised with the following arguments: {}'.format(saved_args))

        all_country_tags, all_indices, all_text = [], [], []

        # for label, class_data in data.items():
        for label in which_country:
            # if label in which_country:
            class_data = data[label]
            keys = list(class_data.keys())
            all_indices.extend(keys)
            for key in keys:
                all_country_tags.append(label)
                all_text.append(' '.join(class_data[key][0]))
        if tf:
            tf_vect = TfidfVectorizer(min_df=cut_off, stop_words=stop_words, use_idf=False)
            term_frequencies = tf_vect.fit_transform(all_text)
            feature_names = tf_vect.get_feature_names_out()
        else:
            count_vect = CountVectorizer(min_df=cut_off, stop_words=stop_words)
            term_frequencies = count_vect.fit_transform(all_text)
            feature_names = count_vect.get_feature_names_out()

        self.targets = all_country_tags
        self.indices = all_indices
        self.tfs = term_frequencies
        self.feature_names = feature_names


class LinguisticFeatureExtractor:
    # define the linguistic feature extractor class that will be used by the models to build the tailored feature vectors

    def __init__(self, data, raw_data, which_country, tf, voseo_count=[0]*12, overt_subject_count=[0]*7, subj_inf_count=[0]*3, art_poss_count=[0], tense_count=[0]*14, quest_count=[0], diminutive_count=[0]*4, mas_negation_count=[0], muy_isimo_count=[0], ada_count=[0], clitic_count=[0]*3, ser_estar_count=[0]*2):

        # saved_args = locals()
        # print('LinguisticFeatureExtractor was initialised with the following arguments: {}'.format(saved_args))

        count_name_dict = {0: 'voseo', 1: 'overt_subj', 2: 'subj_inf', 3: 'indef_art_poss', 4: 'diff_tenses', 5: 'non_inv_quest', 6: 'diminutives', 7: 'mas_neg', 8: 'muy_isimo', 9: 'ada', 10: 'clitic_pronouns', 11: 'ser_or_estar'}

        def make_count_tf(count: int, text: list, tf: bool) -> Union[float, None]:
            # return the tf version of the tailored count

            if tf:
                return [count_val / len(text[0]) for count_val in count]
            

        def reset_counts(counts: list) -> Sequence[list]:
            # reset the counts to 0 for every new document
            # hereby, make sure to keep the Falses if a count is not specified

            res_counts = []
            for count in counts:
                if count:
                    res_counts.append([0]*len(count))
                else:
                    res_counts.append(count)

            return (count for count in res_counts)

        all_document_counts = {}

        if diminutive_count or ada_count:
            nlp = spacy.load("es_core_news_lg")


        for label, class_data in data.items():
            print('\n-------------------------------------------------\nAnalysing dialect {}\n-------------------------------------------------\n'.format(label))
            if label in which_country:
                class_document_counts = {}
                for idx, text in class_data.items():
                    if voseo_count or overt_subject_count:
                        voseo_count, overt_subject_count = features_utils.forward_to_voseo_or_overt_subj(text, 2, 2, voseo_count, overt_subject_count)
                        voseo_count = make_count_tf(voseo_count, text, tf)
                        overt_subject_count = make_count_tf(overt_subject_count, text, tf)

                    if subj_inf_count:
                        subj_inf_count = features_utils.subject_preceeds_infinitive(raw_data[label][idx], subj_inf_count)
                        subj_inf_count = make_count_tf(subj_inf_count, text, tf)

                    if art_poss_count:
                        art_poss_count = features_utils.ind_article_possessive(text, art_poss_count)
                        art_poss_count = make_count_tf(art_poss_count, text, tf)

                    if tense_count:
                        tense_count = features_utils.different_tenses(text, tense_count)
                        tense_count = make_count_tf(tense_count, text, tf)

                    if quest_count:
                        quest_count = features_utils.non_inverted_questions(raw_data[label][idx], quest_count)
                        quest_count = make_count_tf(quest_count, text, tf)

                    if diminutive_count:
                        diminutive_count = features_utils.diminutives(raw_data[label][idx], diminutive_count, nlp)
                        diminutive_count = make_count_tf(diminutive_count, text, tf)

                    if mas_negation_count:
                        mas_negation_count = features_utils.mas_negation_cuba(text, mas_negation_count)
                        mas_negation_count = make_count_tf(mas_negation_count, text, tf)

                    if muy_isimo_count:
                        muy_isimo_count = features_utils.muy_and_isimo_peru(text, muy_isimo_count)
                        muy_isimo_count = make_count_tf(muy_isimo_count, text, tf)

                    if ada_count:
                        ada_count = features_utils.ada_costa_rica(raw_data[label][idx], ada_count, nlp)
                        ada_count = make_count_tf(ada_count, text, tf)

                    if clitic_count:
                        clitic_count = features_utils.different_clitic_pronouns(raw_data[label][idx], clitic_count)
                        clitic_count = make_count_tf(clitic_count, text, tf)

                    if ser_estar_count:
                        ser_estar_count = features_utils.ser_or_estar(raw_data[label][idx], ser_estar_count)
                        ser_estar_count = make_count_tf(ser_estar_count, text, tf)


                    counts = [voseo_count, overt_subject_count, subj_inf_count, art_poss_count, tense_count, quest_count, diminutive_count, mas_negation_count, muy_isimo_count, ada_count, clitic_count, ser_estar_count]
                    # add the collected counts to the dictionary that will contain all of the document counts for the class with the doc's ID as key
                    class_document_counts[idx] = {count_name_dict[i]: count for i, count in enumerate(counts)}
                    # reset the count to 0 for every new document
                    voseo_count, overt_subject_count, subj_inf_count, art_poss_count, tense_count, quest_count, diminutive_count, mas_negation_count, muy_isimo_count, ada_count, clitic_count, ser_estar_count = reset_counts(counts)
                            
            all_document_counts[label] = class_document_counts

        self.document_counts = all_document_counts


if __name__ == "__main__":

    which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']
    # which_country = ['PA']

    start = time.time()
    cr = CorpusReader('/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS', which_country, filter_punct=True, filter_digits=True, filter_nes=False, lower=True, split_data=False, group=False)
    # cr = CorpusReader('/projekte/semrel/WORK-AREA/Users/laura/toy_corpus', which_country, filter_punct=True, filter_digits=True, lower=True, split_data=False)
    end = time.time()
    print('Corpus reader took {} seconds.'.format(end - start))
    # with open('/projekte/semrel/WORK-AREA/Users/laura/ngram_frequencies_indices.json', 'r') as jsn:
    #     indices = json.load(jsn)

    # new_train = {}
    # for k, v in cr.train.items():
    #     val = [idx for idx in v if idx in indices['indices']]
    #     new_train[k] = val

    # new_dev = {}
    # for k, v in cr.dev.items():
    #     val = [idx for idx in v if idx in indices['indices']]
    #     new_dev[k] = val

    # new_test = {}
    # for k, v in cr.test.items():
    #     val = [idx for idx in v if idx in indices['indices']]
    #     new_test[k] = val

    # split_dict = {'train': cr.train, 'dev': cr.dev, 'test': cr.test}
    # split_dict = {'train': new_train, 'dev': new_dev, 'test': new_test}
    # with open('/projekte/semrel/WORK-AREA/Users/laura/toy_train_dev_test_split.json', 'w') as jsn:
    #     json.dump(split_dict, jsn)

    # reader_indices_dict = {'indices': cr.ids}
    # 
    # with open('/projekte/semrel/WORK-AREA/Users/laura/test_reader_indices.json', 'w') as jsn:
    #     json.dump(reader_indices_dict, jsn)

    # stop_words = ['vos', 'tú', 'tí', 'ti', 'vosotros', 'vosotras', 'os', 'usted', 'ustedes', 'yo', 'él', 'ella', 'ello', 'ellos', 'ellas', 'nosotros', 'nosotras', 'lo', 'le', 'les', 'boliviano', 'boliviana', 'bolivianos', 'bolivianas', 'cubano', 'cubana', 'cubanos', 'cubanas', 'argentino', 'argentina', 'argentinos', 'argentinas', 'chileno', 'chilena', 'chilenos', 'chilenas', 'colombiano', 'colombiana', 'colombianos', 'colombianas', 'costarricense', 'costarricenses', 'dominicano', 'dominicana', 'dominicanos', 'dominicanas', 'ecuatoriano', 'ecuatoriana', 'ecuatorianos', 'ecuatorianas', 'guatemalteco', 'guatemalteca', 'guatemaltecos', 'guatemaltecas', 'hondureño', 'hondureña', 'hondureños', 'hondureñas', 'mexicano', 'mexicana', 'mexicanos', 'mexicanas', 'nicaragüense', 'nicaragüenses', 'panameño', 'panameña', 'panameños', 'panameñas', 'paraguayo', 'paraguaya', 'paraguayos', 'paraguayas', 'puertorriqueño', 'puertorriqueña', 'puertorriqueños', 'puertorriqueñas', 'peruano', 'peruana', 'peruanos', 'peruanas', 'salvadoreño', 'salvadoreña', 'salvadoreños', 'salvadoreñas', 'uruguayo', 'uruguaya', 'uruguayos', 'uruguayas', 'venezolano', 'venezolana', 'venezolanos', 'venezolanas']
# 
# # 
    # overall_start = time.time()
    # extractor_ngram = NgramFeatureExtractor(cr.data, which_country, stop_words=stop_words)
# 
    # overall_end = time.time()
    # print('Feature search took {} seconds'.format(overall_end-overall_start))
# 
    # # for k,v in cr.data['PA'].items():
    # #     print(v[0])
    # extractor_dict = {'indices': extractor_ngram.indices, 'feature_names': list(extractor_ngram.feature_names)}
    # print(extractor_dict['feature_names'])
# # # 
    # with open('/projekte/semrel/WORK-AREA/Users/laura/ngram_features/ngram_frequencies_indices_feature_names_tf_nofeat_nones.json', 'w') as jsn:
    #     json.dump(extractor_dict, jsn)
    #     save_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/ngram_features/tf_nofeat/ngram_frequencies_spmatrix_tf_nofeat_nones', extractor_ngram.tfs)
# 
    # overall_start = time.time()
# 
    # extractor_ngram = NgramFeatureExtractor(cr.data, which_country, stop_words=stop_words, tf=False)
# 
    # overall_end = time.time()
    # print('Feature search took {} seconds'.format(overall_end-overall_start))
# 
    # extractor_dict = {'indices': extractor_ngram.indices, 'feature_names': list(extractor_ngram.feature_names)}
# 
    # with open('/projekte/semrel/WORK-AREA/Users/laura/ngram_features/ngram_frequencies_indices_feature_names_counts_nofeat_nones.json', 'w') as jsn:
    #     json.dump(extractor_dict, jsn)
# 
    # save_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_nones', extractor_ngram.tfs)

    
    # overall_start = time.time()
    # which_country = ['ANT', 'MCA', 'GC', 'CV', 'EP', 'AU', 'ES', 'MX', 'CL', 'PY']
# 
    # extractor_ngram = NgramFeatureExtractor(cr.data, which_country, tf=False, stop_words=stop_words)
# 
    # overall_end = time.time()
    # print('Feature search took {} seconds'.format(overall_end-overall_start))
# 
    # extractor_dict = {'indices': extractor_ngram.indices, 'feature_names': list(extractor_ngram.feature_names)}

    # with open('/projekte/semrel/WORK-AREA/Users/laura/ngram_features/ngram_frequencies_indices_feature_names_counts.json', 'w') as jsn:
    #     json.dump(extractor_dict, jsn)

    # save_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/ngram_features/counts_nofeat/ngram_frequencies_spmatrix_counts_nofeat_grouped', extractor_ngram.tfs)
# 
    # overall_start = time.time()
# 
    # extractor_ngram = NgramFeatureExtractor(cr.data, which_country)
 # 
    # overall_end = time.time()
    # print('Feature search took {} seconds'.format(overall_end-overall_start))
 # 
    # extractor_dict = {'indices': extractor_ngram.indices, 'feature_names': list(extractor_ngram.feature_names)}
 # 
    # with open('/projekte/semrel/WORK-AREA/Users/laura/ngram_features/tf/ngram_frequencies_indices_feature_names_tf.json', 'w') as jsn:
    #     json.dump(extractor_dict, jsn)
    # 
    # save_sparse_csr('/projekte/semrel/WORK-AREA/Users/laura/ngram_features/tf/ngram_frequencies_spmatrix_tf', extractor_ngram.tfs)

    overall_start = time.time()
    extractor_ling = LinguisticFeatureExtractor(cr.data, cr.raw, which_country, tf=True)
# 
    overall_end = time.time()
    print('Feature search took {} seconds'.format(overall_end-overall_start))
# 
    with open('/projekte/semrel/WORK-AREA/Users/laura/tailored_features/feature_dict_tf_updated.json', 'w') as jsn:
        json.dump(extractor_ling.document_counts, jsn)

    
    overall_start = time.time()
    extractor_ling = LinguisticFeatureExtractor(cr.data, cr.raw, which_country, tf=False)
# 
    overall_end = time.time()
    print('Feature search took {} seconds'.format(overall_end-overall_start))
# 
    with open('/projekte/semrel/WORK-AREA/Users/laura/tailored_features/feature_dict_updated.json', 'w') as jsn:
        json.dump(extractor_ling.document_counts, jsn)

    # for country, ids in cr.ids.items():
    #     print('{} -- {} IDs'.format(country, len(ids)))
# 
    # for country, num in cr.number_of_tokens.items():
    #     print('{} -- {} IDs'.format(country, num))