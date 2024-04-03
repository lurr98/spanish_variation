#!/usr/bin/env python3
"""
Author: Laura Zeidler

**description**
"""
import pandas as pd

def make_tables(stats_dict: dict, which_country: list, which_stat: tuple, val_type: str) -> str:

    selected_stats = {country: [] for country in which_country}

    for country in which_country:
        for gram_type, values in stats_dict[country][which_stat[0]][which_stat[1]].items():
            for value in values:
                if val_type:
                    selected_stats[country].append(value[0])
                else:
                    selected_stats[country].append(value)
