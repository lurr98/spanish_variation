#!/usr/bin/env python3
"""
Author: Laura Zeidler

**description**
"""

import json, time, sys
import numpy as np
import numpy.typing as npt
sys.path.append("..")
from corpus.corpus_reader import CorpusReader


# global variable so that this doesn't have to be passed around all the time
with open('../corpus/inverted_POS_tags.json', 'r') as jsn:
    POS_mapping = json.load(jsn)


def voseo(class_data: dict, post_window: int) -> np.ndarray:
    # look for subject pronouns that constitue how/if the voseo is used
    # further look for following verb of vos and their endings, since they say something about the conjugation paradigm that is used in the dialect
    # finally build a feature vector with the counts of the occurring pronouns and endings

    def add_voseo_count(voseo_count: list, idx: int) -> tuple[list, bool]:
        # little helper function to make function more readable

        voseo_count[idx] += 1
        already_added_count = True
        print('Counted!')
        return (voseo_count, already_added_count)

    voseo_count = [0]*10
    pron_dict = {'vos': ('pp-2p', 0), 'tú': ('ps', 1), 'tu': ('pp-2cs', 1), 'usted': ('pp-2cs', 2)}
    pron_keys = list(pron_dict.keys())

    # TODO: probably put the loop in the main part so that we can look for multiple features in the same loop
    for idx, text in class_data.items():
        for i, token in enumerate(text[1]):
            # if lemma is vos, tu or usted
            if token.strip() in pron_keys:
                # check if the lemma was tagged with the appropriate POS tag
                if text[2][i].strip() == pron_dict[token.strip()][0]:
                    print(token)
                    # add count to the respective spot in the list
                    voseo_count[pron_dict[token.strip()][1]] += 1

                    try:
                        if token.strip() == 'vos':
                            # define variable that will tell us whether the voseo count was already added
                            already_added_count = False
                            print('{} -- {}'.format(text[0][i+1].strip(), text[2][i+1].strip()))
                            print('{} -- {}'.format(text[0][i+2].strip(), text[2][i+2].strip()))
                            print('{} -- {}'.format(text[0][i+3].strip(), text[2][i+2].strip()))

                            for window_idx in list(range(post_window)):
                                if already_added_count:
                                    pass
                                else:
                                    # if the following token is tagged as a verb
                                    if text[2][i+window_idx].strip().startswith('v'):
                                        # make sure the tense id only indicative present or in the mixed verbs (where also "v" belongs to)
                                        if not POS_mapping[text[2][i+window_idx].strip()] != 'VM' or not text[2][i+window_idx].strip() in ['vip-2s', 'vip-2p']:
                                            continue

                                    if text[0][i+window_idx].strip().endswith('áis') or text[0][i+window_idx].strip().endswith('ais'):
                                        # check if word is not a proper noun
                                        # ! ofc there might still be other words that end in that character sequence but that's very improbable so I'll neglect this
                                        if not text[2][i+window_idx].strip() == 'o':
                                            voseo_count, already_added_count = add_voseo_count(voseo_count, 3)
                                    elif text[0][i+window_idx].strip().endswith('éis') or text[0][i+window_idx].strip().endswith('eis'):
                                        if not text[2][i+window_idx].strip() == 'o':
                                            voseo_count, already_added_count = add_voseo_count(voseo_count, 4)
                                    elif text[0][i+window_idx].strip().endswith('ís') and text[1][i+window_idx+1].strip().endswith('er'):
                                        voseo_count, already_added_count = add_voseo_count(voseo_count, 5)
                                    elif text[0][i+window_idx].strip().endswith('ás'):
                                        voseo_count, already_added_count = add_voseo_count(voseo_count, 6)
                                    elif text[0][i+window_idx].strip().endswith('és'):
                                        voseo_count, already_added_count = add_voseo_count(voseo_count, 7)
                                    elif text[0][i+window_idx].strip().endswith('as'):
                                        voseo_count, already_added_count = add_voseo_count(voseo_count, 8)
                                    elif text[0][i+window_idx].strip().endswith('es'):
                                        voseo_count, already_added_count = add_voseo_count(voseo_count, 9)
                    except IndexError:
                        print('----------------------------------------\n')
                        pass

                    print('----------------------------------------\n')

    print('Voseo count: {}'.format(voseo_count))


def overt_subjects(class_data: dict, post_window: int, try_person: bool) -> np.ndarray:
    # look for overt subject pronouns
    # return a feature vector containing the counts for each pronoun

    subj_count = [0]*7
    subj_dict = {'yo': ('ps', 0), 'tú': ('ps', 1), 'vos': ('pp-2p', 2), 'él': ('ps', 3), 'nosotros': ('ps', 4), 'vosotros': ('ps', 5), 'ustedes': ('pp-2p', 6)}
    subj_dict_person = {'yo': ['1s', '1/3s'], 'tú': ['2s'], 'vos': ['2s'], 'él': ['3s', '1/3s'], 'nosotros': ['1p'], 'vosotros': ['2p'], 'ustedes': ['3p'], 'ellos': ['3p']}
    subj_keys = list(subj_dict.keys())

    for idx, text in class_data.items():
        for i, token in enumerate(text[1]):
            if token.strip() in subj_keys:
                already_added_count = False
                try:
                    for window_idx in list(range(post_window)):
                        if already_added_count:
                            pass
                        else:
                            # check whether a verb follows the pronoun (window of 2 currently)
                            if POS_mapping[text[2][i+window_idx].strip()].startswith('V'):
                                # TODO: check for correct person? 
                                if try_person:
                                    if text[0][i].lower() in ['ellos', 'ellas']:
                                        token = 'ellos'
                                    for verb_ending in subj_dict_person[token.strip()]:
                                        if text[2][i+window_idx].strip().endswith(verb_ending):
                                            subj_count[subj_dict[token.strip()][1]] += 1
                                            already_added_count = True
                                            break
                                else:
                                    subj_count[subj_dict[token.strip()][1]] += 1
                                    already_added_count = True
                            # break the while loop if there is some type of punctuation after the pronoun
                            # bc if a verb occurs after that it is most likely not associated with the pronoun
                            elif text[2][i+window_idx].strip() == 'y':
                                already_added_count = True
                except IndexError:
                    print('----------------------------------------\n')
                    pass

                print('----------------------------------------\n')

    return np.array(subj_count)



def subject_preceeds_infinitive(class_data: dict)  -> np.ndarray:
    # look for subjects that preceed an infinitive
    # e.g. ~ al yo venir ~

    subj_inf_count = 0

    for idx, text in class_data.items():
        for i, pos_tag in enumerate(text[2]):
            if pos_tag.strip() == 'ps':
                if text[2][i+1].strip() in ['vr', 'vpp', 'vpp-00']:
                    subj_inf_count += 1

    return np.array([subj_inf_count])


def ind_article_possessive(class_data: dict) -> np.ndarray:
    # look for the sequence `indefinite article + possessive + noun`
    # e.g. ~ una mi amiga ~

    art_poss_count = 0

    for idx, text in class_data.items():
        for i, pos_tag in enumerate(text[2]):
            if POS_mapping[pos_tag.strip()] == 'ARTI':
                if POS_mapping[text[2][i+1].strip()] in ['PP', 'DETP']:
                    if POS_mapping[text[2][i+2].strip()] in ['NN', 'NE']:
                        art_poss_count += 1

    return np.array([art_poss_count])


# TODO: think about ho to make all of this more efficient! Maybe have a big loop and let features be "triggered", e.g. by a specific POS tag!


if __name__ == "__main__":

    which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']
    # which_country = ['CU', 'PA']

    # initialise dictionary to store results
    stats_dict = {}

    start = time.time()
    # cr = CorpusReader('/projekte/semrel/Resources/Corpora/Corpus-del-Espanol/Lemma-POS', which_country, 'pars', True)
    cr = CorpusReader('/projekte/semrel/WORK-AREA/Users/laura/toy_corpus', which_country, 'pars', True)
    end = time.time()
    print('Corpus reader took {} seconds.'.format(end - start))

    overall_start = time.time()

    # now execute all functions that need the class data
    for label, data in cr.data.items():
        print('\n-------------------------------------------------\nAnalysing dialect {}\n-------------------------------------------------\n'.format(label))
        voseo(data, 2)

    overall_end = time.time()
    print('Feature search took {} seconds'.format(overall_end-overall_start))