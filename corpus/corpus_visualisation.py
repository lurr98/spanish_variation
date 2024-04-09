#!/usr/bin/env python3
"""
Author: Laura Zeidler

**description**
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


def plot_heatmap(stats_dict: dict, which_stat: tuple) -> None:

    ax = sns.heatmap(np.array(stats_dict['all_countries'][which_stat[0]][which_stat[1]][which_stat[2]]), linewidth=0.5)
    plt.show()