#!/usr/bin/env python3
"""
Author: Laura Zeidler
Last changed: 14.08.2024

This script provides various functions for analyzing and visualizing linguistic statistics from a dataset. 
It includes functionality for creating bar plots and heatmaps and plotting Zipf's law distributions. 
The visualizations and outputs are saved as images and LaTeX files for further use in reports or analysis.

To generate LaTeX tables, bar plots, heatmaps, or Zipf's law distribution plots, call the appropriate function with the required data and parameters.

"""
import matplotlib.cm, json
from random import shuffle
from datetime import date
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cmap = matplotlib.cm.get_cmap('tab20', 20)
color_list = [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(0, cmap.N, 2)] + [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(1, cmap.N, 2)]


def make_bar_plots(stats_dict: dict, which_country: list, which_stat: list) -> None:
    # creates bar plots or grouped bar plots based on selected statistics for multiple countries

    which_country = sorted(which_country)

    for stat in which_stat:

        fig = plt.figure(figsize = (10, 5))      

        # check if there is only one variable per country, if so, create standard bar plot
        if len(stat) == 1:
            stat = stat[0]
            # creating the bar plot
            plt.bar(which_country, [stats_dict[country][stat] for country in which_country], color=color_list[0], width=0.4)

            # add xticks on the middle of the group bars
            plt.xlabel('Country tag')
            plt.ylabel('Average document length')
            plt.title('Average document length per country')
            # plt.xlabel('Country tag')
            # plt.ylabel('Number of {}'.format(stat))
            # plt.title('Number of {} per country'.format(stat))

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
            # plt.ylabel('Number of {}'.format(stat_str))
            plt.ylabel('Average document length')
            # plt.title('Number of {} per country'.format(stat_str))
            plt.title('Average document length per country')

        # save graphic
        if isinstance(stat, str):
            plt.savefig('plots/{}_plot{}.png'.format(stat, date.today()))
        else:
            stat_str = '_'.join(ind_stat for ind_stat in stat)
            plt.savefig('plots/{}_plot{}.png'.format(stat_str, date.today()))


def plot_heatmap(stats_dict: dict, which_stat: tuple) -> None:
    # plots a heatmap of cosine similarity between classes in a dataset (e.g., based on n-gram statistics)

    sorted_data = []
    for sims in stats_dict['all_countries'][which_stat[0]][which_stat[1]][which_stat[2]][0]:
        zipped_arrays = list(zip(stats_dict['all_countries'][which_stat[0]][which_stat[1]][which_stat[2]][1], sims))
        sorted_targets, sorted_data_sims = zip(*sorted(zipped_arrays))
        sorted_data.append(sorted_data_sims)
    zipped_arrays = list(zip(stats_dict['all_countries'][which_stat[0]][which_stat[1]][which_stat[2]][1], sorted_data))
    sorted_targets, sorted_data = zip(*sorted(zipped_arrays))
    
    fig, ax = plt.subplots(figsize=(12,10))
    plot = ax.imshow(sorted_data)
    cbar = ax.figure.colorbar(plot, ax = ax)
    cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")
    plt.title('Cosine similarity between the classes in the CdE')
    plt.yticks(range(len(sorted_targets)), list(sorted_targets))
    plt.xticks(range(len(sorted_targets)), list(sorted_targets))
    plt.savefig('plots/{}_{}_plot{}.png'.format(which_stat[0], which_stat[2], date.today()))


def plot_zipf_distribution(frequencies: dict, without_digits: bool) -> None:
    # plots Zipf's law distribution based on word frequencies from the dataset
    # generates two subplots: one for the general distribution and one for lower-frequency words.

    fig, (ax_x, ax_y) = plt.subplots(2, figsize=(10, 8))
    ax_x.set_title('Zipf\'s Curve (truncated x-axis)')
    ax_y.set_title('Closer Look into Lower Frequencies (truncated y-axis)')

    keys = list(frequencies.keys())
    if without_digits:
        freq_values = [frequencies[key] for key in keys]
    else:
        freq_values = [frequencies[key] for key in keys]

    sorted_zip = sorted(zip(freq_values, keys), reverse=True)
    sorted_freqs, sorted_keys = zip(*sorted_zip)

    # add ticks and title
    ax_x.set_xlabel('Token Rank')
    ax_x.set_ylabel('Token Frequency')

    ax_y.set_xlabel('Token Rank')
    ax_y.set_ylabel('Token Frequency')

    plt.subplots_adjust(hspace=0.5)

    ax_x.plot(sorted_freqs[:200], color='royalblue')
    ax_y.set_ylim(0, 50)
    ax_y.plot(sorted_freqs, color='royalblue')    
    plt.savefig('plots/zipfs_law_plot{}.png'.format(date.today()))



if __name__ == "__main__":

    # stats_dict = {"DO": {"documents": 47065, "token": 34591907}, "CL": {"documents": 71620, "token": 65559977}, "HN": {"documents": 43227, "token": 34150553}, "ES": {"documents": 421520, "token": 423091653}, "PA": {"documents": 29312, "token": 22797585}, "BO": {"documents": 43293, "token": 39462190}, "NI": {"documents": 35696, "token": 30508428}, "CO": {"documents": 184970, "token":160074530}, "GT": {"documents": 61434, "token": 53558251}, "PR": {"documents": 33879, "token": 32061618}, "CR": {"documents": 33255, "token": 29862729}, "EC": {"documents": 63160, "token": 52109837}, "AR": {"documents": 177920, "token":169949379}, "VE": {"documents": 112571, "token":92024874}, "PY": {"documents": 33301, "token": 28577446}, "CU": {"documents": 51708, "token": 57678235}, "SV": {"documents": 38217, "token": 36366532}, "PE": {"documents": 121814, "token":105611118}, "UY": {"documents": 36154, "token":38579425}, "MX": {"documents": 286275, "token":238856348}}

    with open('/projekte/semrel/WORK-AREA/Users/laura/data_analysis/ngram_frequency_dict.json', 'r') as jsn:
        freq_dict = json.load(jsn)
    
    with open('/projekte/semrel/WORK-AREA/Users/laura/data_analysis/stats_dict_updated.json', 'r') as jsn:
        stats_dict = json.load(jsn)

    # example:
    plot_heatmap(stats_dict, ('cosine_similarity_tailored_tf_grouped', 'token', 'unigram'))
    plot_zipf_distribution(freq_dict)
    make_bar_plots(stats_dict, ['AR', 'BO', 'CL', 'CO', 'CR', 'CU', 'DO', 'EC', 'ES', 'GT', 'HN', 'MX', 'NI', 'PA', 'PE', 'PR', 'PY', 'SV', 'UY', 'VE'], [['av_len']])