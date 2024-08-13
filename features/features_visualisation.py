import matplotlib.cm, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
from typing import Tuple

stats_name_dict = {'voseo': 'vos, tú and usted', 'vosotros': 'use of vosotros', 'voseo_endings': 'different voseo conjugation paradigms', 'overt_subj': 'use of overt subject pronouns', 'subj_inf': 'subject preceding infinitive', 'indef_art_poss': 'indefinite article and possessive constructions', 'non_inv_quest': 'non-inverted questions', 'diminutives': 'different diminutives', 'mas_neg': 'más preceeding negative markers', 'muy_isimo': 'muy preceeding -ísimo suffix', 'ada': 'words constructed with suffix -ada', 'clitic_pronouns': 'different clitic pronouns', 'ser_or_estar': 'ser or estar preceeding an adjective', 'diff_tenses': 'different tenses'}
variable_explanation = {'voseo': ['vos', 'tú', 'usted', 'vosotros', 'ustedes'], 'voseo_endings': ['-áis', '-éis', '-ís for -er verbs', '-ás', '-és', '-as', '-es'], 'subj_inf': ['SUBJ preceding VR', 'SUBJ preceding VPP', 'SUBJ preceding VPP-00'], 'indef_art_poss': [], 'diminutives': ['-ico/a', '-ito/a', '-illo/a', '-ingo/a'], 'clitic_pronouns': ['lo', 'le', 'les'], 'ser_or_estar': ['ser', 'estar'], 'diff_tenses': ['VM', 'VC', 'VIF', 'VII', 'VIP', 'VIS', 'VIMP', 'VPP', 'VPS', 'VR', 'VSF', 'VSI', 'VSJ', 'VSP']}
cmap = matplotlib.cm.get_cmap('tab20', 20)
color_list = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(0, cmap.N, 2)] + [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(1, cmap.N, 2)]


def sum_document_values(feature_dict: dict, which_country: list, stat: str) -> Tuple[dict, bool]:
    # sum the values from all documents and return them in a dicitonary

    all_stats = {}
    for country in which_country:    
        for idx, stats in feature_dict[country].items():
            # store whether there are more than 1 variables
            # print(len(stats[stat]))
            is_one = True if len(stats[stat]) == 1 else False 
            if country in all_stats:
                all_stats[country] = [old+new for old, new in zip(all_stats[country], stats[stat])]
            else:
                all_stats[country] = stats[stat]

    return all_stats, is_one


def feature_bar_chart(feature_dict: dict, which_country: list, which_stats: list, normalisation: bool | dict =False) -> None:
    # visualise features with one or not many different variables as a bar chart
        
    which_country = sorted(which_country)

    for stat in which_stats:
        # divide voseo feature into (voseo, tuteo, usted) and vosotros for better overview 
        if stat == 'vosotros':
            all_stats, is_one = sum_document_values(feature_dict, which_country, 'voseo')
            all_stats = {key: [feature_vector[3]] for key, feature_vector in all_stats.items()}
            is_one = True
        else:
            all_stats, is_one = sum_document_values(feature_dict, which_country, stat)
        print(all_stats)

        if stat == 'voseo':
            all_stats = {key: feature_vector[:3] for key, feature_vector in all_stats.items()}
        if stat == 'overt_subj':
            all_stats = {key: [sum(feature_vector)] for key, feature_vector in all_stats.items()}
            is_one = True

        # check if there is only one variable per country, if so, create standard bar plot
        if is_one:
            fig = plt.figure(figsize = (10, 5))
            print('one')
            print(stat)
            
            # creating the bar plot
            keys = sorted(list(all_stats.keys()))
            print(keys)
            print([all_stats[key] for key in keys])
            if normalisation:
            # normalise data by provided values if specifed (mostly data size probably)
                plt.bar(keys, [all_stats[key][0] / normalisation[key] for key in keys], color=color_list[0], width = 0.4)
            else:
                plt.bar(keys, [all_stats[key][0] for key in keys], color=color_list[0], width = 0.4)

        # if not, create grouped bar plot
        else:
            fig = plt.figure(figsize = (15, 5))
            print('more')
            print(stat)
            # set width of bars
            bar_width = 0.25

            # separate_stats_values = [[] for x in range(len(all_stats['PA']))][:-1]
            separate_stats_values = [[] for x in range(len(all_stats['PA']))]
            print(separate_stats_values)
            for value_idx in list(range(len(all_stats['PA']))):
            # for value_idx in list(range(len(all_stats['PA'])))[:-1]:
                for country in which_country:
                    if normalisation:
                    # normalise data by provided values if specifed (mostly data size probably)
                        separate_stats_values[value_idx].append(all_stats[country][value_idx] / normalisation[country])
                    else:
                        separate_stats_values[value_idx].append(all_stats[country][value_idx])
                    print(separate_stats_values)

            if len(separate_stats_values) > 3:
                positions = [list(range(0, len(separate_stats_values[0])*2, 2))]
            else:
                positions = [list(range(len(separate_stats_values[0])))]
            for i in list(range(len(separate_stats_values))):
                positions.append([prev_position + bar_width for prev_position in positions[-1]])

            print(positions)
            print(separate_stats_values)

            for i, bar in enumerate(separate_stats_values):
                plt.bar(positions[i], bar, color=color_list[i], width=bar_width, edgecolor='white', label=variable_explanation[stat][i])

            if len(separate_stats_values) > 3:
                plt.xticks([r + bar_width for r in range(0, len(which_country)*2, 2)], list(all_stats.keys()))
            else:
                plt.xticks([r + bar_width for r in range(len(which_country))], list(all_stats.keys()))
            plt.legend()

        plt.title('Occurrences of {} per country data'.format(stats_name_dict[stat]))
        # add xticks on the middle of the group bars
        plt.xlabel('Country tag')

        if normalisation:
            plt.ylabel('No. of occurrences of {}\nnorm. by no. of documents'.format(stats_name_dict[stat]))
            plt.savefig('plots/{}_plot{}_norm.png'.format(stat, date.today()))
        else:
            plt.ylabel('No. of occurrences of {}'.format(stats_name_dict[stat]))
            plt.savefig('plots/{}_plot{}.png'.format(stat, date.today()))


def feature_stacked_bar_chart(feature_dict: dict, which_country: list, which_stats: list, normalisation: bool | dict =False) -> None:
    # visualise features with one or not many different variables as a bar chart
        
    which_country = sorted(which_country)

    for stat in which_stats:
        fig = plt.figure(figsize = (15, 15))

        if stat == 'voseo_endings':
            all_stats, is_one = sum_document_values(feature_dict, which_country, 'voseo')
            all_stats = {key: feature_vector[5:] for key, feature_vector in all_stats.items()}
        elif stat == 'diff_tense':
            all_stats, is_one = sum_document_values(feature_dict, which_country, 'diff_tense')
        else:
            all_stats, is_one = sum_document_values(feature_dict, which_country, stat)

        endings_list = []
        for i, country in enumerate(which_country):
            endings_list.append([country] + [value / normalisation[country] for value in all_stats[country]])
        if normalisation:
            df = pd.DataFrame(endings_list,
              columns=['tense', 'VM', 'VC', 'VIF', 'VII', 'VIP', 'VIS', 'VIMP', 'VPP', 'VPS', 'VR', 'VSF', 'VSI', 'VSJ', 'VSP'])
        print(df.set_index('tense').T.reset_index().rename(columns={'index':'tense'}))
        df = df.set_index('tense').T.reset_index().rename(columns={'index':'tense'})
        print(df.columns)
        #     plt.scatter([num+1 for num in list(range(len(all_stats[country])))], [value / normalisation[country] for value in all_stats[country]], c=color_list[i], label=country)
        # else:
        #     plt.scatter([num+1 for num in list(range(len(all_stats[country])))], all_stats[country], c=color_list[i], label=country)
        
        # plt.xticks([num+1 for num in list(range(len(all_stats['PA'])))], variable_explanation[stat])

        # plt.xlabel('Feature variables')
        # plt.title('Occurrences of {} per country data'.format(stats_name_dict[stat]))
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
        # plot data in stack manner of bar type
        ax = df.plot(x='tense', kind='bar', stacked=True, title='Occurrences of different tenses and aspects per country data', rot=45, color=color_list)

        ax.legend(bbox_to_anchor=(1.0, 1.0))

        if normalisation:
            plt.ylabel('No. of occurrences of {}\nnorm. by no. of documents'.format(stats_name_dict[stat]))
            plt.savefig('plots/{}_plot{}_norm.png'.format(stat, date.today()), bbox_inches='tight')
        else:
            plt.ylabel('No. of occurrences of {}'.format(stats_name_dict[stat]))
            plt.savefig('plots/{}_plot{}.png'.format(stat, date.today()))


def feature_scatter_plot(feature_dict: dict, which_country: list, which_stats: list, normalisation: bool | dict =False) -> None:
    # visualise features with many different variables

    which_country = sorted(which_country)

    for stat in which_stats:
        fig = plt.figure(figsize = (10, 10))

        if stat == 'voseo_endings':
            all_stats, is_one = sum_document_values(feature_dict, which_country, 'voseo')
            all_stats = {key: feature_vector[5:] for key, feature_vector in all_stats.items()}
        else:
            all_stats, is_one = sum_document_values(feature_dict, which_country, stat)

        for i, country in enumerate(which_country):
            if normalisation:
                plt.scatter([num+1 for num in list(range(len(all_stats[country])))], [value / normalisation[country] for value in all_stats[country]], c=color_list[i], label=country)
            else:
                plt.scatter([num+1 for num in list(range(len(all_stats[country])))], all_stats[country], c=color_list[i], label=country)
        
        plt.xticks([num+1 for num in list(range(len(all_stats['PA'])))], variable_explanation[stat])

        plt.xlabel('Feature variables')
        plt.title('Occurrences of {} per country data'.format(stats_name_dict[stat]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if normalisation:
            plt.ylabel('No. of occurrences of {}\nnorm. by no. of documents'.format(stats_name_dict[stat]))
            plt.savefig('plots/{}_plot{}_norm.png'.format(stat, date.today()))
        else:
            plt.ylabel('No. of occurrences of {}'.format(stats_name_dict[stat]))
            plt.savefig('plots/{}_plot{}.png'.format(stat, date.today()))


if __name__ == "__main__":

    feature_path = '/projekte/semrel/WORK-AREA/Users/laura/tailored_features/feature_dict_tf.json'
    which_country = ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE']
    # which_stats = ['voseo', 'overt_subj', 'subj_inf', 'indef_art_poss', 'non_inv_quest', 'diminutives', 'mas_neg', 'muy_isimo', 'ada', 'clitic_pronouns', 'ser_or_estar']
    which_stats = ['diff_tenses']
    normalisation_dict = {"DO": 47065, "CL": 71620, "HN": 43227, "ES": 421520, "PA": 29312, "BO": 43293, "NI": 35696, "CO": 184970, "GT": 61434, "PR": 33879, "CR": 33255, "EC": 63160, "AR": 177920, "VE": 112571, "PY": 33301, "CU": 51708, "SV": 38217, "PE": 121814, "UY": 36154, "MX": 286275}

    with open(feature_path, 'r') as f:
        feature_dict = json.load(f)

    # feature_bar_chart(feature_dict, which_country, which_stats, normalisation_dict)
    feature_stacked_bar_chart(feature_dict, which_country, which_stats, normalisation_dict)
    # feature_scatter_plot(feature_dict, which_country, which_stats, normalisation_dict)
    