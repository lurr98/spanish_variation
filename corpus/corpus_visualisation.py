#!/usr/bin/env python3
"""
Author: Laura Zeidler

**description**
"""
import matplotlib.cm, json
from datetime import date
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cmap = matplotlib.cm.get_cmap('tab20', 20)
color_list = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(0, cmap.N, 2)] + [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(1, cmap.N, 2)]


def make_tables(stats_dict: dict, which_country: list, which_stat: tuple, val_type: str) -> str:

    selected_stats = {country: [] for country in which_country}
    indices = []

    for country in which_country:
        for gram_number, values in stats_dict[country][which_stat[0]][which_stat[1]].items():
            for value in values:
                if val_type == 'token':
                    selected_stats[country].append(value[0])
                elif val_type == 'count':
                    selected_stats[country].append(value[1])
                indices.append('{} {} {}'.format(which_stat[0], which_stat[1], gram_number))

    
    df = pd.DataFrame(data=selected_stats, index=indices)

    return df.to_latex()


def make_bar_plots(stats_dict: dict, which_country: list, which_stat: list) -> None:

    which_country = sorted(which_country)

    for stat in which_stat:

        fig = plt.figure(figsize = (10, 5))      

        # check if there is only one variable per country, if so, create standard bar plot
        if len(stat) == 1:
            # creating the bar plot
            plt.bar(which_country, [stats_dict[country][stat] for country in which_country], color ='maroon', width = 0.4)

            # add xticks on the middle of the group bars
            plt.xlabel('Country tag')
            plt.ylabel('Number of {}'.format(stat))
            plt.title('Number of {} per country'.format(stat))

        # if not, create grouped bar plot
        else:
            # set width of bars
            bar_width = 0.25

            bar_list = [[] for x in range(len(stat))]
            for country in which_country:
                for i, ind_stat in enumerate(stat):
                    bar_list[i].append(stats_dict[country][ind_stat])

            positions = [list(range(len(which_country)))]
            for i in list(range(len(stat)-1)):
                positions.append([prev_position + bar_width for prev_position in positions[-1]])

            for i, bar in enumerate(bar_list):
                plt.bar(positions[i], bar, color=color_list[i], width=bar_width, edgecolor='white', label=stat[i])

            plt.yscale("log")
            plt.xticks([r + bar_width for r in range(len(which_country))], which_country)
            plt.legend()

            stat_str = ' and '.join(ind_stat for ind_stat in stat)

            # add xticks on the middle of the group bars
            plt.xlabel('Country tag')
            plt.ylabel('Number of {}'.format(stat_str))
            plt.title('Number of {} per country'.format(stat_str))

        # save graphic
        if len(stat) == 1:
            plt.savefig('plots/{}_plot{}.png'.format(stat, date.today()))
        else:
            stat_str = '_'.join(ind_stat for ind_stat in stat)
            plt.savefig('plots/{}_plot{}.png'.format(stat_str, date.today()))


def plot_heatmap(stats_dict: dict, which_stat: tuple) -> None:

    ax = sns.heatmap(np.array(stats_dict['all_countries'][which_stat[0]][which_stat[1]][which_stat[2]]), linewidth=0.5)
    plt.savefig('plots/{}_plot{}.png'.format(which_stat[0], date.today()))


if __name__ == "__main__":

    stats_dict = {"DO": {"documents": 47065, "token": 34591907}, "CL": {"documents": 71620, "token": 65559977}, "HN": {"documents": 43227, "token": 34150553}, "ES": {"documents": 421520, "token": 423091653}, "PA": {"documents": 29312, "token": 22797585}, "BO": {"documents": 43293, "token": 39462190}, "NI": {"documents": 35696, "token": 30508428}, "CO": {"documents": 184970, "token":160074530}, "GT": {"documents": 61434, "token": 53558251}, "PR": {"documents": 33879, "token": 32061618}, "CR": {"documents": 33255, "token": 29862729}, "EC": {"documents": 63160, "token": 52109837}, "AR": {"documents": 177920, "token":169949379}, "VE": {"documents": 112571, "token":92024874}, "PY": {"documents": 33301, "token": 28577446}, "CU": {"documents": 51708, "token": 57678235}, "SV": {"documents": 38217, "token": 36366532}, "PE": {"documents": 121814, "token":105611118}, "UY": {"documents": 36154, "token":38579425}, "MX": {"documents": 286275, "token":238856348}}

    # with open('/projekte/semrel/WORK-AREA/Users/laura/test_stats_dict.json', 'r') as jsn:
    #     stats_dict = json.load(jsn)

    make_bar_plots(stats_dict, ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE'], ['documents', 'token', ['documents', 'token']])