from pingouin import partial_corr
import numpy as np
from scipy.stats import spearmanr, kendalltau
import pandas as pd


def output_partial_correlations(properties, target, data, features_to_control = ['score_delta']):
    correlations = dict()
    for property in properties:
        if property not in features_to_control:
            spearman_partial = partial_corr(data=data, x=target, y=property, covar=features_to_control, method='spearman')
            correlations[property] = {'spearman_partial': np.round(spearman_partial['r'].values[0], 3), 
                                      'p-value': np.round(spearman_partial['p-val'].values[0], 3)}
    return pd.DataFrame(correlations)