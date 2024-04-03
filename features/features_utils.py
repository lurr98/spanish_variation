#!/usr/bin/env python3
"""
Author: Laura Zeidler

**description**
"""

import json, time, sys, spacy
import numpy as np
import numpy.typing as npt

# global variable so that this doesn't have to be passed around all the time
with open('../corpus/inverted_POS_tags.json', 'r') as jsn:
    POS_mapping = json.load(jsn)


def voseo(text: list, post_window: int, voseo_count: list) -> np.ndarray:
    # look for subject pronouns that constitue how/if the voseo is used
    # further look for following verb of vos and their endings, since they say something about the conjugation paradigm that is used in the dialect
    # finally build a feature vector with the counts of the occurring pronouns and endings

    def add_voseo_count(voseo_count: list, idx: int) -> tuple[list, bool]:
        # little helper function to make function more readable
        voseo_count[idx] += 1
        already_added_count = True
        # print('Counted!')
        # print(voseo_count)
        return (voseo_count, already_added_count)
    
    pron_dict = {'vos': ('pp-2p', 0), 'tú': ('ps', 1), 'tu': ('pp-2cs', 1), 'usted': ('pp-2cs', 2), 'vosotros': ('ps', 3)}
    pron_keys = list(pron_dict.keys())
    # # TODO: probably put the loop in the main part so that we can look for multiple features in the same loop
    # for idx, text in class_data.items():
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
                        # print('{} -- {}'.format(text[0][i+1].strip(), text[2][i+1].strip()))
                        # print('{} -- {}'.format(text[0][i+2].strip(), text[2][i+2].strip()))
                        # print('{} -- {}'.format(text[0][i+3].strip(), text[2][i+3].strip()))
                        for window_idx in list(range(post_window)):
                            window_count = window_idx + 1
                            if already_added_count:
                                pass
                            else:
                                # if the following token is tagged as a verb
                                if text[2][i+window_count].strip().startswith('v'):
                                    # make sure the tense id only indicative present or in the mixed verbs (where also "v" belongs to)
                                    if POS_mapping[text[2][i+window_count].strip()] != 'VM' and text[2][i+window_count].strip() not in ['vip-2s', 'vip-2p']:
                                        continue
                                    
                                if text[0][i+window_count].strip().endswith('áis') or text[0][i+window_count].strip().endswith('ais'):
                                    # check if word is not a proper noun
                                    # ! ofc there might still be other words that end in that character sequence but that's very improbable so I'll neglect this
                                    if not text[2][i+window_count].strip() == 'o':
                                        voseo_count, already_added_count = add_voseo_count(voseo_count, 4)
                                elif text[0][i+window_count].strip().endswith('éis') or text[0][i+window_count].strip().endswith('eis'):
                                    if not text[2][i+window_count].strip() == 'o':
                                        voseo_count, already_added_count = add_voseo_count(voseo_count, 5)
                                elif text[0][i+window_count].strip().endswith('ís') and text[1][i+window_count+1].strip().endswith('er'):
                                    voseo_count, already_added_count = add_voseo_count(voseo_count, 6)
                                elif text[0][i+window_count].strip().endswith('ás'):
                                    voseo_count, already_added_count = add_voseo_count(voseo_count, 7)
                                elif text[0][i+window_count].strip().endswith('és'):
                                    voseo_count, already_added_count = add_voseo_count(voseo_count, 8)
                                elif text[0][i+window_count].strip().endswith('as'):
                                    voseo_count, already_added_count = add_voseo_count(voseo_count, 9)
                                elif text[0][i+window_count].strip().endswith('es'):
                                    voseo_count, already_added_count = add_voseo_count(voseo_count, 10)
                                # break the while loop if there is some type of punctuation after the pronoun
                                # bc if a verb occurs after that it is most likely not associated with the pronoun
                                elif text[2][i+window_count].strip() == 'y':
                                    already_added_count = True
                except IndexError:
                    print('----------------------------------------\n')
                    pass
                
                print('----------------------------------------\n')

    print('Voseo count: {}'.format(voseo_count))
    return voseo_count


def overt_subjects(text: list, post_window: int, try_person: bool, subj_count: list) -> np.ndarray:
    # look for overt subject pronouns
    # return a feature vector containing the counts for each pronoun
    subj_dict = {'yo': ('ps', 0), 'tú': ('ps', 1), 'vos': ('pp-2p', 2), 'él': ('ps', 3), 'nosotros': ('ps', 4), 'vosotros': ('ps', 5), 'ustedes': ('pp-2p', 6), 'ellos': ('ps', 7)}
    subj_dict_person = {'yo': ['1s', '1/3s'], 'tú': ['2s'], 'vos': ['2s'], 'él': ['3s', '1/3s'], 'nosotros': ['1p'], 'vosotros': ['2p'], 'ustedes': ['3p'], 'ellos': ['3p']}
    subj_keys = list(subj_dict.keys())
    # for idx, text in class_data.items():
    for i, token in enumerate(text[1]):
        if token.strip() in subj_keys:
           already_added_count = False
           try:
               for window_idx in list(range(post_window)):
                   window_count = window_idx + 1
                   if already_added_count:
                       pass
                   else:
                       # check whether a verb follows the pronoun (window of 2 currently)
                       if POS_mapping[text[2][i+window_count].strip()].startswith('V'):
                           # TODO: check for correct person? 
                           if try_person:
                               if text[0][i].lower().strip() in ['ellos', 'ellas']:
                                   token = 'ellos'
                               for verb_ending in subj_dict_person[token.strip()]:
                                   if text[2][i+window_count].strip().endswith(verb_ending):
                                       print('---------------------------------\n{} -- {}\n{} -- {}\n'.format(token, text[2][i], text[0][i+window_count], text[2][i+window_count]))
                                       subj_count[subj_dict[token.strip()][1]] += 1
                                       already_added_count = True
                                       break
                           else:
                               print('---------------------------------\n{} -- {}\n{} -- {}\n'.format(token, text[2][i], text[0][i+window_count], text[2][i+window_count]))
                               subj_count[subj_dict[token.strip()][1]] += 1
                               already_added_count = True
                       # break the loop if there is some type of punctuation after the pronoun
                       # bc if a verb occurs after that it is most likely not associated with the pronoun
                       elif text[2][i+window_count].strip() == 'y':
                           already_added_count = True
           except IndexError:
               print('----------------------------------------\n')
               pass
           print('----------------------------------------\n')

    print('Subject count: {}'.format(subj_count))
    return subj_count


def subject_preceeds_infinitive(text: list, subj_inf_count: list)  -> np.ndarray:
    # look for subjects that preceed an infinitive
    # e.g. ~ al yo venir ~
    # TODO: ??????
    verb_list = ['VM', 'VC', 'VIF', 'VII', 'VIP', 'VIS', 'VIMP', 'VPP', 'VPS', 'VR', 'VSF', 'VSI', 'VSJ', 'VSP']
    # for idx, text in class_data.items():
    for i, pos_tag in enumerate(text[2]):
        if pos_tag.strip() == 'ps':
            try:
                if POS_mapping[text[2][i-1].strip()] != 'PREP' and POS_mapping[text[2][i-1].strip()] not in verb_list:
                    if text[2][i+1].strip() == 'vr':
                        print('---------------------------------\n')
                        for j in list(range(10))[::-1]:
                            print('{} -- {}'.format(text[0][i-(j+1)], text[2][i-(j+1)]))
                        print('{} -- {}\n{} -- {}\n'.format(text[0][i], text[2][i], text[0][i+1], text[2][i+1]))
                        for j in list(range(10)):
                            print('{} -- {}'.format(text[0][i+j+1], text[2][i+j+1]))
                        subj_inf_count[0] += 1
                    elif text[2][i+1].strip() =='vpp':
                        print('---------------------------------\n')
                        for j in list(range(10))[::-1]:
                            print('{} -- {}'.format(text[0][i-(j+1)], text[2][i-(j+1)]))
                        print('{} -- {}\n{} -- {}\n'.format(text[0][i], text[2][i], text[0][i+1], text[2][i+1]))
                        for j in list(range(10)):
                            print('{} -- {}'.format(text[0][i+j+1], text[2][i+j+1]))
                        subj_inf_count[1] += 1
                    elif text[2][i+1].strip() == 'vpp-00':
                        print('---------------------------------\n')
                        for j in list(range(10))[::-1]:
                            print('{} -- {}'.format(text[0][i-(j+1)], text[2][i-(j+1)]))
                        print('{} -- {}\n{} -- {}\n'.format(text[0][i], text[2][i], text[0][i+1], text[2][i+1]))
                        for j in list(range(10)):
                            print('{} -- {}'.format(text[0][i+j+1], text[2][i+j+1]))
                        subj_inf_count[2] += 1
            except IndexError:
                print('----------------------------------------\n')
                pass

    print('Subject before infinitive count: {}'.format(subj_inf_count))
    return subj_inf_count


def ind_article_possessive(text: list, art_poss_count: list) -> np.ndarray:
    # look for the sequence `indefinite article + possessive + noun`
    # e.g. ~ una mi amiga ~
    # for idx, text in class_data.items():
    for i, pos_tag in enumerate(text[2]):
        if POS_mapping[pos_tag.strip()] == 'ARTI':
            try:
                if POS_mapping[text[2][i+1].strip()] in ['PP', 'DETP']:
                    if POS_mapping[text[2][i+2].strip()] in ['NN', 'NE']:
                        print('---------------------------------\n{} -- {}\n{} -- {}\n{} -- {}\n---------------------------------\n'.format(text[1][i], pos_tag, text[1][i+1], text[2][i+1], text[1][i+2], text[2][i+2]))
                        art_poss_count[0] += 1
            except IndexError:
                pass

    print('Indefinite article, possessive, noun construction count: {}'.format(art_poss_count))
    return art_poss_count


def different_tenses(text: list, tense_count: list) -> np.ndarray:
    # count the different tenses that are annotated (use broader categories)
    # maybe don't take VM into the mix?
    tense_list = ['VM', 'VC', 'VIF', 'VII', 'VIP', 'VIS', 'VIMP', 'VPP', 'VPS', 'VR', 'VSF', 'VSI', 'VSJ', 'VSP']
    # for idx, text in class_data.items():
    for i, pos_tag in enumerate(text[2]):
        if POS_mapping[pos_tag.strip()] in tense_list:
            tense_count[tense_list.index(POS_mapping[pos_tag.strip()])] += 1

    print('Tense count: {}'.format(tense_count))
    return tense_count


def non_inverted_questions(text: list, quest_count: list) -> np.ndarray:
    # count the times that a WH-pronoun is followed by pronoun
    # TODO: also count inverted questions?
    subj_pron_list = ['pd-3cs"', 'pd-3fp"', 'pd-3fs', 'pd-3mp', 'pd-3ms', 'ps', 'pp-1cs', 'pp-2cp', 'pp-2cs', 'pp-2p', 'fp']
    # for idx, text in class_data.items():
    for i, pos_tag in enumerate(text[2]):
        if POS_mapping[pos_tag.strip()] == 'PINT':
            try:
                # really simplify this by just looking for subject pronoun following the WH-pronoun
                if text[2][i+1].strip() in subj_pron_list:
                    # make sure that the question is not 'por qué' bc this allows the non-inversion anyway
                    # source: DOI: https://doi.org/10.1075/avt.15.03baa 
                    if not (text[1][i-1].strip() == 'por' and text[2][i].strip() == 'pq-3cn'):
                        quest_count[0] += 1
                        print('---------------------------------\n{} -- {}\n{} -- {}\n{} -- {}\n---------------------------------\n'.format(text[0][i], pos_tag, text[1][i+1], text[2][i+1], text[0][i+2], text[2][i+2]))
            except IndexError:
                pass

    print('Non-inverted question count: {}'.format(quest_count))
    return quest_count


def diminutives(text: list, diminutive_count: list) -> np.ndarray:
    # look for diminutives with different endings
    # the possible endings are: '-ico', '-illo', '-ito', '-ingo'
    nlp = spacy.load("es_core_news_lg")
    diminutive_endings = ['ico', 'ito', 'illo', 'ingo']
    # for idx, text in class_data.items():
    for i, token in enumerate(text[1]):
        # check whether ending fits diminutive endings of three characters
        if token.strip()[-3:] in diminutive_endings[:2]:
            three_or_four = 3
        # check whether ending fits diminutive endings of four characters
        elif token.strip()[-4:] in diminutive_endings[2:]:
            three_or_four = 4
        else:
            return diminutive_count
        # check whether word is a noun
        if POS_mapping[text[2][i].strip()] == 'NN':
            doc = nlp(token)
            # check if the word is out of vocabulary, if so, accept as diminutive
            if doc[0].is_oov:
                # if a hyphen is in the word, it's likely that it's an OOV word but not a diminutive
                if not '-' in token:
                    print('---------------------------------\n{} -- {}\n---------------------------------\n'.format(token, text[2][i]))
                    diminutive_count[diminutive_endings.index(token.strip()[-three_or_four:])] += 1

    print('Diminutive count: {}'.format(diminutive_count))
    return diminutive_count


def mas_negation_cuba(text: list, mas_negation_count: list) -> np.ndarray:
    # count the occurences of 'más' preceeding 'nunca', 'nada' or 'nadie'
    # for idx, text in class_data.items():
    for i, token in enumerate(text[1]):
        if token.strip() == 'más':
            try:
                if text[1][i+1].strip() in ['nunca', 'nada', 'nadie']:
                    mas_negation_count[0] += 1
            except IndexError:
                pass

    print('"más" + negation count: {}'.format(mas_negation_count))
    return mas_negation_count


def muy_and_isimo_peru(text: list, muy_isimo_count: list) -> np.ndarray:
    # look for the combination of 'muy' and an adjective ending in '-ísimo'
    # e.g. ~ muy riquísimo ~
    # for idx, text in class_data.items():
    for i, token in enumerate(text[1]):
        if token.strip() == 'muy':
            try:
                if text[1][i+1].strip().endswith('ísimo'):
                    muy_isimo_count[0] += 1
            except IndexError:
                pass

    print('"muy" + "ísimo" count: {}'.format(muy_isimo_count))
    return muy_isimo_count


def ada_costa_rica(text: list, ada_count: list) -> np.ndarray:
    # look for nouns ending in -ada
    nlp = spacy.load("es_core_news_lg")
    # for idx, text in class_data.items():
    for i, token in enumerate(text[1]):
        if token.strip().endswith('ada'):
            if POS_mapping[text[2][i].strip()] == 'NN':
                doc = nlp(token)
                # check if the word is out of vocabulary, if so, accept as collective noun with -ada
                if doc[0].is_oov:
                    ada_count[0] += 1

    print('"-ada" count: {}'.format(ada_count))
    return ada_count