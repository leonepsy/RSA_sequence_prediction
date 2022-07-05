# -*- coding: utf-8 -*-
"""Create model specifications for single-trial estimation (least square single)."""

from pathlib import Path
import pandas as pd
import simplejson as json

# Helper function
def _create_spec_lss(beh_df,
                     confounds_df,
                     group_index=None, 
                     item_index=None,
                     confounds_list=None,
                     fd_threshold=None):
    import numpy as np
    from decimal import Decimal

    spec = dict()
    # Main regressors
    spec['main'] = {
        'onsets': beh_df['Stim_real_onset'].values.tolist(),
        'durations': [2.0] * beh_df.shape[0], #beh_df['duration'].values.tolist(),
        'amplitudes': [1.0] * beh_df.shape[0]
    }
    # Group index
    if group_index is None:
        spec['group_index'] = [1] * beh_df.shape[0]
    else:
        spec['group_index'] = group_index
    # Item index
    if item_index is None:
        spec['item_index'] = [1] * beh_df.shape[0]
    else:
        spec['item_index'] = item_index
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

