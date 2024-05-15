#!/usr/bin/env python3
"""
Author: Laura Zeidler

**description**
"""

import json, time, sys, spacy, re
import numpy as np
import numpy.typing as npt
from spacy.language import Language
from typing import Tuple


# global variable so that this doesn't have to be passed around all the time
with open('../corpus/POS_related/inverted_POS_tags.json', 'r') as jsn:
    POS_mapping = json.load(jsn)


# TODO: put voseo and overt subject together? or figure out how to convert to regex!
    

def forward_to_voseo_or_overt_subj(text: list, post_window_voseo: int, post_window_overt: int, voseo_count: list, subj_count: list) -> Tuple[list, list]:
    # decide whether 

    pron_keys = ['vos', 'tu']
    subj_keys = ['yo', 'vos', 'nosotros']

    for i, token in enumerate(text[1]):
        if token.strip() in ['tú', 'vosotros', 'usted']:
            if voseo_count:
                voseo_count = voseo(text, token, i, post_window_voseo, voseo_count)
            if subj_count:
                subj_count = overt_subjects(text, token, i,  post_window_overt, subj_count)
        else:
            if voseo_count:
                if token.strip() in pron_keys:
                    voseo_count = voseo(text, token, i, post_window_voseo, voseo_count)
            if subj_count:
                if token.strip() in subj_keys:
                    subj_count = overt_subjects(text, token, i, post_window_overt, subj_count)

    return voseo_count, subj_count



def voseo(text: list, token: str, i: int, post_window: int, voseo_count: list) -> list:
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
    
    # NOTE: 'tu' is also searched bc it refers to the direct object e.g. ~ para tí~ 
    # and this is different between 'tú' and 'vos' e.g. ~ para vos ~
    pron_dict = {'vos': ('pp-2p', 0), 'tú': ('ps', 1), 'tu': ('pp-2cs', 1), 'usted': ('pp-2cs', 2), 'vosotros': ('ps', 3), 'ustedes': ('pp-2cp', 4)}
    pron_keys = list(pron_dict.keys())
    # # TODO: probably put the loop in the main part so that we can look for multiple features in the same loop
    # for idx, text in class_data.items():
    # for i, token in enumerate(text[0]):
    #     token = token.strip().lower()
    #     if token in pron_keys:
    # check if the lemma was tagged with the appropriate POS tag
    if text[2][i].strip() == pron_dict[token][0]:
        # add count to the respective spot in the list
        voseo_count[pron_dict[token][1]] += 1
        try:
            if token == 'vos':
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
                                voseo_count, already_added_count = add_voseo_count(voseo_count, 5)
                        elif text[0][i+window_count].strip().endswith('éis') or text[0][i+window_count].strip().endswith('eis'):
                            if not text[2][i+window_count].strip() == 'o':
                                voseo_count, already_added_count = add_voseo_count(voseo_count, 6)
                        elif text[0][i+window_count].strip().endswith('ís') and text[1][i+window_count+1].strip().endswith('er'):
                            voseo_count, already_added_count = add_voseo_count(voseo_count, 7)
                        elif text[0][i+window_count].strip().endswith('ás'):
                            voseo_count, already_added_count = add_voseo_count(voseo_count, 8)
                        elif text[0][i+window_count].strip().endswith('és'):
                            voseo_count, already_added_count = add_voseo_count(voseo_count, 9)
                        elif text[0][i+window_count].strip().endswith('as'):
                            voseo_count, already_added_count = add_voseo_count(voseo_count, 10)
                        elif text[0][i+window_count].strip().endswith('es'):
                            voseo_count, already_added_count = add_voseo_count(voseo_count, 11)
                        # break the while loop if there is some type of punctuation after the pronoun
                        # bc if a verb occurs after that it is most likely not associated with the pronoun
                        elif text[2][i+window_count].strip() == 'y':
                            already_added_count = True
        except IndexError:
            pass
        
    # print('Voseo count: {}'.format(voseo_count))
    return voseo_count


def overt_subjects(text: list, token: str, i: int, post_window: int, subj_count: list) -> list:
    # look for overt subject pronouns
    # return a feature vector containing the counts for each pronoun
    subj_dict = {'yo': ('ps', 0), 'tú': ('ps', 1), 'vos': ('pp-2p', 2), 'él': ('ps', 3), 'ella': ('ps', 3), 'usted': ('ps', 4), 'nosotros': ('ps', 5), 'nosotras': ('ps', 5), 'vosotros': ('ps', 6), 'vosotras': ('ps', 6), 'ustedes': ('pp-2p', 7), 'ellos': ('ps', 8), 'ellas': ('ps', 8)}
    subj_dict_person = {'yo': ['1s', '1/3s'], 'tú': ['2s'], 'vos': ['2s'], 'él': ['3s', '1/3s'], 'ella': ['3s', '1/3s'], 'usted': ['3s', '1/3s'], 'nosotros': ['1p'], 'nosotras': ['1p'], 'vosotros': ['2p'], 'vosotras': ['2p'], 'ustedes': ['3p'], 'ellos': ['3p'], 'ellas': ['3p']}
    # for idx, text in class_data.items():
    # for i, token in enumerate(text[0]):
    #     token = token.strip().lower()
    #     if token in subj_keys:
    already_added_count = False
    try:
        for window_idx in list(range(post_window)):
            window_count = window_idx + 1
            if already_added_count:
                pass
            else:
                # check whether a verb follows the pronoun (window of 2 currently)
                if POS_mapping[text[2][i+window_count].strip()].startswith('V'):
                    for verb_ending in subj_dict_person[token]:
                        if text[2][i+window_count].strip().endswith(verb_ending):
                            # print('---------------------------------\n{} -- {}\n{} -- {}\n'.format(token, text[2][i], text[0][i+window_count], text[2][i+window_count]))
                            subj_count[subj_dict[token][1]] += 1
                            already_added_count = True
                            break
                # break the loop if there is some type of punctuation after the pronoun
                # bc if a verb occurs after that it is most likely not associated with the pronoun
                elif text[2][i+window_count].strip() == 'y':
                    already_added_count = True
    except IndexError:
        pass

    # print('Subject count: {}'.format(subj_count))
    return subj_count


def subject_preceeds_infinitive(raw_text: str, subj_inf_count: list)  -> list:
    # look for subjects that preceed an infinitive
    # e.g. ~ antes de yo venir ~
   
    pattern = re.compile(r'ps\t[\w.\-À-ÿ]+\n([\w.\-À-ÿ]+\t){2}(vr|vpp|vpp-00)')

    finds = pattern.findall(raw_text)

    for find in finds:
        if find[1].endswith('vr'):
            subj_inf_count[0] += 1
        elif find[1].endswith('vpp'):
            subj_inf_count[1] += 1
        elif find[1].endswith('vpp-00'):
            subj_inf_count += 1

    # print('Subject before infinitive count: {}'.format(subj_inf_count))
    return subj_inf_count


def ind_article_possessive(text: list, art_poss_count: list) -> list:
    # look for the sequence `indefinite article + possessive + noun`
    # e.g. ~ una mi amiga ~
    # for idx, text in class_data.items():

    pattern = re.compile(r'ARTI\t(PP|DETP)\t(NN|NE)')
    # for i, pos_tag in enumerate(text[2]):
    #     if POS_mapping[pos_tag.strip()] == 'ARTI':
    #         try:
    #             if POS_mapping[text[2][i+1].strip()] in ['PP', 'DETP']:
    #                 if POS_mapping[text[2][i+2].strip()] in ['NN', 'NE']:
    #                     print('---------------------------------\n{} -- {}\n{} -- {}\n{} -- {}\n---------------------------------\n'.format(text[1][i], pos_tag, text[1][i+1], text[2][i+1], text[1][i+2], text[2][i+2]))
    #                     art_poss_count[0] += 1
    #         except IndexError:
    #             pass

    art_poss_count[0] += len(pattern.findall('\t'.join(text[3])))

    # print('Indefinite article, possessive, noun construction count: {}'.format(art_poss_count))
    return art_poss_count


def different_tenses(text: list, tense_count: list) -> list:
    # count the different tenses that are annotated (use broader categories)
    # maybe don't take VM into the mix?

    tense_list = ['VM', 'VC', 'VIF', 'VII', 'VIP', 'VIS', 'VIMP', 'VPP', 'VPS', 'VR', 'VSF', 'VSI', 'VSJ', 'VSP']
    # for idx, text in class_data.items():
    for i, tense in enumerate(tense_list):
        pattern = re.compile(r'\b{}\b'.format(tense))
        tense_count[i] += len(pattern.findall('\t'.join(text[3])))

    # for i, pos_tag in enumerate(text[2]):
    #     if POS_mapping[pos_tag.strip()] in tense_list:
    #         tense_count[tense_list.index(POS_mapping[pos_tag.strip()])] += 1

    # print('Tense count: {}'.format(tense_count))
    return tense_count


def non_inverted_questions(raw_text: str, quest_count: list) -> list:
    # count the times that a WH-pronoun is followed by pronoun
    # TODO: also count inverted questions?
    # subj_pron_list = ['pd-3cs"', 'pd-3fp"', 'pd-3fs', 'pd-3mp', 'pd-3ms', 'ps', 'pp-1cs', 'pp-2cp', 'pp-2cs', 'pp-2p', 'fp']
    # for idx, text in class_data.items():
    # for i, pos_tag in enumerate(text[2]):
    #     if POS_mapping[pos_tag.strip()] == 'PINT':
    #         try:
    #             # really simplify this by just looking for subject pronoun following the WH-pronoun
    #             if text[2][i+1].strip() in subj_pron_list:
    #                 # make sure that the question is not 'por qué' bc this allows the non-inversion anyway
    #                 # source: DOI: https://doi.org/10.1075/avt.15.03baa 
    #                 if not (text[1][i-1].strip() == 'por' and text[2][i].strip() == 'pq-3cn'):
    #                     quest_count[0] += 1
    #                     print('---------------------------------\n{} -- {}\n{} -- {}\n{} -- {}\n---------------------------------\n'.format(text[0][i], pos_tag, text[1][i+1], text[2][i+1], text[0][i+2], text[2][i+2]))
    #         except IndexError:
    #             pass

    # make sure that the question is not 'por qué' bc this allows the non-inversion anyway
    # source: DOI: https://doi.org/10.1075/avt.15.03baa
    pattern = re.compile(r'(?<!\tpor)(\t[\w.\-À-ÿ]+){2}\n([\w.\-À-ÿ]+\t){3}PINT\n([\w.\-À-ÿ]+\t){2}(pd-3cs"|pd-3fp"|pd-3fs|pd-3mp|pd-3ms|ps|pp-1cs|pp-2cp|pp-2cs|pp-2p|fp)')

    quest_count[0] += len(pattern.findall(raw_text))

    # print('Non-inverted question count: {}'.format(quest_count))
    return quest_count


def diminutives(raw_text: str, diminutive_count: list, spacy_nlp: Language) -> list:
    # look for diminutives with different endings
    # the possible endings are: '-ico', '-illo', '-ito', '-ingo'
    diminutive_endings = ['ico', 'ito', 'illo', 'ingo']
    # # for idx, text in class_data.items():
    # for i, token in enumerate(text[1]):
    #     # check whether ending fits diminutive endings of three characters
    #     if token.strip()[-3:] in diminutive_endings[:2]:
    #         three_or_four = 3
    #     # check whether ending fits diminutive endings of four characters
    #     elif token.strip()[-4:] in diminutive_endings[2:]:
    #         three_or_four = 4
    #     else:
    #         return diminutive_count
    #     # check whether word is a noun
    #     if POS_mapping[text[2][i].strip()] == 'NN':
    #         doc = nlp(token)
    #         # check if the word is out of vocabulary, if so, accept as diminutive
    #         if doc[0].is_oov:
    #             # if a hyphen is in the word, it's likely that it's an OOV word but not a diminutive
    #             if not '-' in token:
    #                 print('---------------------------------\n{} -- {}\n---------------------------------\n'.format(token, text[2][i]))
    #                 diminutive_count[diminutive_endings.index(token.strip()[-three_or_four:])] += 1

    sub_pattern_1 = r'[\w.\-À-ÿ]{2,}'
    sub_pattern_2 = r'\t[\w.\-À-ÿ]+\tNN'
    pattern = re.compile(r'({}ico{}|{}ito{}|{}illo{}|{}ingo{})'.format(sub_pattern_1, sub_pattern_2, sub_pattern_1, sub_pattern_2, sub_pattern_1, sub_pattern_2, sub_pattern_1, sub_pattern_2))

    finds = pattern.findall(raw_text)
    for find in finds:
        token = ''.join(find.split('\t')[0])
        doc = spacy_nlp(token)
        # check if the word is out of vocabulary, if so, accept as diminutive
        # if a hyphen is in the word, it's likely that it's an OOV word but not a diminutive
        if doc[0].is_oov and '-' not in token:
            # TODO: find out if I can make this more efficient
            for i, ending in enumerate(diminutive_endings):
                if token.endswith(ending):
                    diminutive_count[i] += 1

    # print('Diminutive count: {}'.format(diminutive_count))
    return diminutive_count


def mas_negation_cuba(text: list, mas_negation_count: list) -> list:
    # count the occurences of 'más' preceeding 'nunca', 'nada' or 'nadie'
    # for idx, text in class_data.items():
    # for i, token in enumerate(text[1]):
    #     if token.strip() == 'más':
    #         try:
    #             if text[1][i+1].strip() in ['nunca', 'nada', 'nadie']:
    #                 mas_negation_count[0] += 1
    #         except IndexError:
    #             pass

    pattern = re.compile(r'\bmás\t(nunca|nada|nadie)\b')

    mas_negation_count[0] += len(pattern.findall('\t'.join(text[1])))

    # print('"más" + negation count: {}'.format(mas_negation_count))
    return mas_negation_count


def muy_and_isimo_peru(text: list, muy_isimo_count: list) -> list:
    # look for the combination of 'muy' and an adjective ending in '-ísimo'
    # e.g. ~ muy riquísimo ~
    # for idx, text in class_data.items():
    # for i, token in enumerate(text[1]):
    #     if token.strip() == 'muy':
    #         try:
    #             if text[1][i+1].strip().endswith('ísimo'):
    #                 muy_isimo_count[0] += 1
    #         except IndexError:
    #             pass

    pattern = re.compile(r'\bmuy\t[\w.\-À-ÿ]+ísimo\b')

    muy_isimo_count[0] += len(pattern.findall('\t'.join(text[1])))

    # print('"muy" + "ísimo" count: {}'.format(muy_isimo_count))
    return muy_isimo_count


def ada_costa_rica(raw_text: str, ada_count: list, spacy_nlp: Language) -> list:
    # look for nouns ending in -ada
    # for idx, text in class_data.items():
    # for i, token in enumerate(text[1]):
    #     if token.strip().endswith('ada'):
    #         if POS_mapping[text[2][i].strip()] == 'NN':
    #             doc = nlp(token)
    #             # check if the word is out of vocabulary, if so, accept as collective noun with -ada
    #             if doc[0].is_oov:
    #                 ada_count[0] += 1

    pattern = re.compile(r'[\w.\-À-ÿ]{2,}ada\t\w+\tNN')
    finds = pattern.findall(raw_text)

    for find in finds:
        token = ''.join(find.split('\t')[0])
        doc = spacy_nlp(token)
        # check if the word is out of vocabulary, if so, accept as collective noun with -ada
        if doc[0].is_oov:
            ada_count[0] += 1

    # print('"-ada" count: {}'.format(ada_count))
    return ada_count


def different_clitic_pronouns(raw_text: str, clitic_count: list) -> list:
    # look for different clitic pronouns
    # keywords loísmo, laísmo, leísmo

    pattern = re.compile(r'\b(lo|le|les)\t[\w.\-À-ÿ]+\tpo\t')
    finds = pattern.findall(raw_text)

    for i, clitic in enumerate(['lo', 'le', 'les']):
        clitic_count[i] += finds.count(clitic)

    return clitic_count


def ser_or_estar(raw_text: str, ser_estar_count: list) -> list:
    # look for "ser" or "estar" preceeding an adjective

    pattern_ser = re.compile(r'ser(\t[\w.\-À-ÿ]+){2}\n(([\w.\-À-ÿ]+\t){3}[\w.\-À-ÿ]+\n)?([\w.\-À-ÿ]+\t){3}ADJ\n')
    pattern_estar = re.compile(r'estar(\t[\w.\-À-ÿ]+){2}\n(([\w.\-À-ÿ]+\t){3}[\w.\-À-ÿ]+\n)?([\w.\-À-ÿ]+\t){3}ADJ\n')

    ser_estar_count[0] += len(pattern_ser.findall(raw_text))
    ser_estar_count[1] += len(pattern_estar.findall(raw_text))

    return ser_estar_count
