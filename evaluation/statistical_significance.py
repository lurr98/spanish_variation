#!/usr/bin/env python3
"""
Author: Laura Zeidler
Last changed: 14.08.2024

This script compares the predictions of two models to determine if there is a statistically significant difference between them. 
It uses McNemar's test, which is appropriate for comparing the performance of two models on the same dataset, particularly in terms of their classification decisions.

"""

import argparse, json
from statsmodels.stats.contingency_tables import mcnemar


def test_statistical_significance(predictions1: list, predictions2: list, targets: list) -> float:
    # test the null hypothesis using the mcnemar (chi-square) test
    # null hypothesis is that the models are the same
    # p value needs to be 0.05 or lower to discard the null hypothesis
        
    corr_both = len([pred for i, pred in enumerate(predictions1) if targets[i] == pred and predictions2[i] == pred])
    corr_m1 = len([pred for i, pred in enumerate(predictions1) if targets[i] == pred and predictions2[i] != pred])
    corr_m2 = len([pred for i, pred in enumerate(predictions2) if targets[i] == pred and predictions1[i] != pred])
    corr_none = len([pred for i, pred in enumerate(predictions1) if targets[i] != pred and predictions2[i] != pred])

    confusion_matrix = [[corr_both, corr_m1], [corr_m2, corr_none]]
    result = mcnemar(confusion_matrix, exact=False)
    
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compare the predictions of two models and figure out, whether the models are actually different according to a statistical test.')
    parser.add_argument('preds_model1', type=str,
                        help='specify the name of the first model to be compared')
    parser.add_argument('preds_model2', type=str,
                        help='specify the name of the second model to be compared')
    parser.add_argument('data_set', type=str,
                        help='specify which of the data sets is used (train|dev|test)')

    args = parser.parse_args()
    print('The script is running with the following arguments: {}'.format(args))

    with open('/projekte/semrel/WORK-AREA/Users/laura/indices_targets_tdt_split_080101_balanced.json', 'r') as jsn:
        ind_target_dict = json.load(jsn)

    targets = ind_target_dict[args.data_set]['targets']

    with open('/projekte/semrel/WORK-AREA/Users/laura/evaluation/predictions.json', 'r') as jsn:
        predictions = json.load(jsn)

    preds_model1 = predictions[args.preds_model1]
    preds_model2 = predictions[args.preds_model2]

    pvalue = test_statistical_significance(preds_model1, preds_model2, targets).pvalue
    print('p-value for model {} and {} is: {}'.format(args.preds_model1, args.preds_model2, pvalue))
    print('p-value is less than or equal to 0.05: {}'.format(pvalue <= 0.05))