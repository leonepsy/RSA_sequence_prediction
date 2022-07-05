# -*- coding: utf-8 -*-
"""Create model specifications for single-trial estimation (least square all)."""

from pathlib import Path
import pandas as pd
import simplejson as json

# Helper function
def _create_spec_lsa(beh_df, confounds_df, item_index=None, confounds_list=None,
                     fd_threshold=None):
    import numpy as np
    from decimal import Decimal 

    # Get model information
    # take care of item model
    # item_index should be a 1-based list which each number indicates a single item regressor. 
    if item_index is None: #
        n_regressors = beh_df.shape[0] 
        query_list = np.array(range(n_regressors)) 
    else:
        n_regressors = len(np.unique(np.array(item_index)))
        query_list = np.array(item_index) - 1 

    spec = dict()
    # Main regressors
    spec['main'] = dict()
    for regressor_id in range(n_regressors):
        regressor_name = f'main_{regressor_id+1:03d}' 
        dat = beh_df.loc[query_list == regressor_id, :] 
        spec['main'][regressor_name] = {
            'onsets': dat['Stim_real_onset'].values.tolist(), 
            'durations': [2.0] * dat.shape[0], 
            'amplitudes': [1.0] * dat.shape[0]
        }

    # Confound regressors
    if confounds_list is not None:
        spec['confounds'] = dict()
        for col_id in confounds_list:
            spec['confounds'][col_id] = [
                round(Decimal(x), 6) for x in np.nan_to_num(confounds[col_id]) 
            ]

    # Additioal censor regressor based on Framewise Displacement
    if fd_threshold is not None:
        if 'confounds' not in spec:
            spec['confounds'] = dict()
        outliers = np.where(
            np.nan_to_num(confounds_df['framewise_displacement'].values) > fd_threshold)[0] 
        if outliers.shape[0] > 0:
            for idx in outliers:
                regressor = np.zeros(confounds_df.shape[0])
                regressor[idx] = 1
                spec['confounds'][f'tr_{idx+1:03d}'] = regressor.tolist() 

    return spec
