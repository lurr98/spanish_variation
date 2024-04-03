#!/usr/bin/env python3
"""
Author: Laura Zeidler

**description**
"""

import json, time, sys
import features_utils
import numpy as np
import numpy.typing as npt
from typing import Sequence

sys.path.append("..")
from corpus.corpus_reader import CorpusReader


class FeatureExtractor:
# define the corpus reader class that will be used by the models to access data

    def __init__(self, data, which_country, voseo_count=[0]*11, overt_subject_count=[0]*8, subj_inf_count=[0]*3, art_poss_count=[0], tense_count=[0]*14, quest_count=[0], diminutive_count=[0]*4, mas_negation_count=[0], muy_isimo_count=[0], ada_count=[0]):

        count_name_dict = {0: 'voseo', 1: 'overt_subj', 2: 'subj_inf', 3: 'indef_art_poss', 4: 'diff_tenses', 5: 'non_inv_quest', 6: 'diminutives', 7: 'mas_neg', 8: 'muy_isimo', 9: 'ada'}
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

        for label, class_data in data.items():
            print('\n-------------------------------------------------\nAnalysing dialect {}\n-------------------------------------------------\n'.format(label))
            if label in which_country:
                class_document_counts = {}
                for idx, text in class_data.items():
                    if voseo_count:
                        voseo_count = features_utils.voseo(text, 2, voseo_count)

                    if overt_subject_count:
                        overt_subject_count = features_utils.overt_subjects(text, 2, True, overt_subject_count)

                    if subj_inf_count:
                        subj_inf_count = features_utils.subject_preceeds_infinitive(text, subj_inf_count)

                    if art_poss_count:
                        art_poss_count = features_utils.ind_article_possessive(text, art_poss_count)

                    if tense_count:
                        tense_count = features_utils.different_tenses(text, tense_count)

                    if quest_count:
                        quest_count = features_utils.non_inverted_questions(text, quest_count)

                    if diminutive_count:
                        diminutive_count = features_utils.diminutives(text, diminutive_count)

                    if mas_negation_count:
                        mas_negation_count = features_utils.mas_negation_cuba(text, mas_negation_count)

                    if muy_isimo_count:
                        muy_isimo_count = features_utils.muy_and_isimo_peru(text, muy_isimo_count)

                    if ada_count:
                        ada_count = features_utils.ada_costa_rica(text, ada_count)


                    counts = [voseo_count, overt_subject_count, subj_inf_count, art_poss_count, tense_count, quest_count, diminutive_count, mas_negation_count, muy_isimo_count, ada_count]
                    # add the collected counts to the dictionary that will contain all of the document counts for the class with the doc's ID as key
                    class_document_counts[idx] = {count_name_dict[counts.index(count)]: (np.array(count) if count else count) for count in counts}
                    # reset the count to 0 for every new document
                    voseo_count, overt_subject_count, subj_inf_count, art_poss_count, tense_count, quest_count, diminutive_count, mas_negation_count, muy_isimo_count, ada_count = reset_counts(counts)
                            
            all_document_counts[label] = class_document_counts

        self.document_counts = all_document_counts


if __name__ == "__main__":

    # which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']
    which_country = ['CU', 'PA']

    start = time.time()
    # cr = CorpusReader('/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS', which_country, 'pars', True)
    cr = CorpusReader('/projekte/semrel/WORK-AREA/Users/laura/toy_corpus', which_country, 'pars')
    end = time.time()
    print('Corpus reader took {} seconds.'.format(end - start))

    overall_start = time.time()

    extractor = FeatureExtractor(cr.data, which_country)

    overall_end = time.time()
    print('Feature search took {} seconds'.format(overall_end-overall_start))


    with open('/projekte/semrel/WORK-AREA/Users/laura/test_feature_dict.json', 'w') as jsn:
        json.dump(extractor.document_counts, jsn)