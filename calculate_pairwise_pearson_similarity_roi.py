# -*- coding: utf-8 -*-
""""Calculate pairwise pattern pearson similarity for ROIs."""

from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import scipy
from scipy import spatial, stats
import multiprocessing as mp
import argparse
import simplejson as json

from nilearn import image as nli
from nilearn import masking as nlm

# Get parallelization parameters
parser = argparse.ArgumentParser(description='Parallelization parameters.')
parser.add_argument(
    '--n-procs',
    action='store',
    default=1,
    type=int,
    help='The maximum number of processes running in parallel..')
args = parser.parse_args()
n_procs = int(args.n_procs)

# Directories
bids_dir = Path('/seastor/leoneshi/fMRI/bids')
deriv_dir = bids_dir.joinpath('derivatives')

# Get participants list
subj_info = pd.read_csv(Path('/home/leoneshi/CSM_project/bids').joinpath('participants.tsv'), delimiter='\t')
subj_list = [i.replace('sub-', '') for i in subj_info['participant_id'].tolist()]


def _split_roi_id(roi_list):

    roi_ids = []
    roi_hemis = []
    for roi_name in roi_list:

        # roi name
        roi_name = roi_name.split('_')
        roi_id = roi_name[0]
        if len(roi_name) == 2 and roi_name[1] == 'L':
            hemi_id = 'left'
        elif len(roi_name) == 2 and roi_name[1] == 'R':
            hemi_id = 'right'
        elif len(roi_name) == 1:
            hemi_id = 'bilateral'

        roi_ids.append(roi_id)
        roi_hemis.append(hemi_id)

    return roi_ids, roi_hemis


def _extract_data(src_img, mask_img, roi_fids):

    mask_dat = mask_img.get_data().astype('int16')

    # loop for roi
    results = []
    for roi_fid in roi_fids:

        roi_img = nli.load_img(Path(roi_fid).as_posix())

        # take intersection between roi and whole brain mask
        roi_dat = np.multiply(mask_dat, roi_img.get_data().astype('int16'))
        roi_img = nli.new_img_like(roi_img, roi_dat, affine=roi_img.affine, copy_header=True)

        # extract data within roi
        if np.sum(roi_dat) > 0:
            results.append(nlm.apply_mask(src_img, roi_img))
        else:
            results.append(None)

    return results


def _extract_data_prewhiting(src_img, res_img, mask_img, roi_id,shrinkage,shrink_est):

    mask_dat = mask_img.get_data().astype('int16')

    # loop for roi
    results = []
    roi_img = nli.load_img(Path(roi_id).as_posix())

    # take intersection between roi and whole brain mask
    roi_dat = np.multiply(mask_dat, roi_img.get_data().astype('int16'))
    roi_img = nli.new_img_like(roi_img, roi_dat, affine=roi_img.affine, copy_header=True)

    # extract data within roi
    if np.sum(roi_dat) > 0:
        src_dat=nlm.apply_mask(src_img, roi_img) # T1*P
        res_dat=nlm.apply_mask(res_img, roi_img) # T2*P
        res_dat=np.array(res_dat)
        if shrinkage is True: ## (refer to the code of paper "The unreliable influence of multivariate noise normalization on the reliability of neural dissimilarity")
            res_dat -= np.nanmean(res_dat, axis=0, keepdims=True) # demean raw residual data
            res_cov_matrix=np.dot(np.transpose(res_dat),res_dat)/res_dat.shape[0] # P*P (voxel covariance matrix)
            F=(np.trace(res_cov_matrix)/res_dat.shape[1])*np.eye(res_dat.shape[1]) 
            if shrink_est is True: # using the OAS method to estimate the shrinkage rate
                c1=1-2/res_dat.shape[1]
                c2=res_dat.shape[0]+1-2/res_dat.shape[1]
                c3=1-res_dat.shape[0]/res_dat.shape[1]
                SS=np.trace(np.linalg.matrix_power(res_cov_matrix,2))
                rho=(c1*SS) / (c2*SS + c3*SS)
                res_cov_matrix=rho*F+rho*res_cov_matrix
            else:
                res_cov_matrix=0.4*F+0.6*res_cov_matrix # shrinkage (pre-defined threshold) (refer to paper "On the distribution of cross-validated Mahalanobis distances")
            results=np.dot(src_dat,scipy.linalg.fractional_matrix_power(res_cov_matrix,-1/2)) # T1*P (prewhiting)
        else:
            res_cov_matrix=np.dot(np.transpose(res_dat),res_dat)/res_dat.shape[0] # P*P
            results=np.dot(src_dat,scipy.linalg.fractional_matrix_power(res_cov_matrix,-1/2)) # T1*P
    else:
        results.append(None)

    return results


def _fisher_z_transform(x):
    """
    Calculate fisher r to z transformation.

    Input should be a ndarray or a single number
    """
    import numpy as np

    np.warnings.filterwarnings('ignore')

    z = np.arctanh(x)
    if isinstance(x, np.ndarray):
        z[np.where(np.isinf(z))] = np.nan
        z[np.where(np.isnan(z))] = np.nan
    elif (np.isnan(z) | np.isinf(z)):
        z = 0

    return z


def _calculate_similarity(src_list, simi_type='correlation', fisher_z=True):

    simi = []
    feature_num = []
    for roi_dat in src_list:

        if roi_dat is not None:
            # remove invalid features
            idx1 = np.where(np.sum(np.isnan(roi_dat), axis=0) == 0, True, False)
            idx2 = np.where(np.sum(np.isinf(roi_dat), axis=0) == 0, True, False)
            idx3 = np.where(np.sum(np.std(roi_dat), axis=0) > 1e-5, True, False)
            idx = idx1 * idx2 * idx3
            roi_dat = roi_dat[:, idx]

            # calculate pairwise similarity
            r = 1 - spatial.distance.pdist(roi_dat, metric=simi_type)
            if fisher_z:
                r = _fisher_z_transform(r)

            # effective feature number
            vox_num = roi_dat.shape[1]

            simi.append(r)
            feature_num.append(vox_num)
        else:
            simi.append(None)
            feature_num.append(0)

    return simi, feature_num


def _pairwise_similarity(subj_id, src_list, res_list, mask_fid, roi_fids,beh_fid, demean=True, zscore=False, prewhiting=True,shrinkage=True,shrink_est=False):

    # load additional brain mask
    mask_img = nli.load_img(Path(mask_fid).as_posix())
    mask_dat = mask_img.get_data().astype('int16')
    # load data
    if prewhiting is True:
        roi_dats = []
        for roi_id in roi_fids:
            roi_dat_prew_all=[]
            for src_fid, res_id in zip(src_list,res_list):

                src_img = nli.load_img(Path(src_fid).as_posix())
                tmp_dat = src_img.get_data()
                # scale data
                with np.errstate(divide='ignore', invalid='ignore'):
                    if demean is True and zscore is False:
                        tmp_dat -= np.nanmean(tmp_dat, axis=3, keepdims=True)
                    elif zscore is True:
                        tmp_dat = stats.zscore(tmp_dat, axis=3)

                src_img = nli.new_img_like(src_img, tmp_dat, affine=src_img.affine, copy_header=True)

                res_img=nli.load_img(Path(res_id).as_posix())
                 # extract roi data
                roi_dat_prew = _extract_data_prewhiting(src_img,res_img,mask_img,roi_id,shrinkage,shrink_est) 
                roi_dat_prew_all.append(roi_dat_prew)
            roi_dat_prew_all=np.array(roi_dat_prew_all)
            roi_dat_prew_all=np.reshape(roi_dat_prew_all,(roi_dat_prew_all.shape[0]*roi_dat_prew_all.shape[1],roi_dat_prew_all.shape[2]))
            roi_dats.append(roi_dat_prew_all)
    else:
        bold_img=[]
        for src_fid in src_list:

            tmp_img = nli.load_img(Path(src_fid).as_posix())
            tmp_dat = tmp_img.get_data()
            # scale data
            with np.errstate(divide='ignore', invalid='ignore'):
                if demean is True and zscore is False:
                    tmp_dat -= np.nanmean(tmp_dat, axis=3, keepdims=True)
                elif zscore is True:
                    tmp_dat = stats.zscore(tmp_dat, axis=3)

            tmp_img = nli.new_img_like(tmp_img, tmp_dat, affine=tmp_img.affine, copy_header=True)
            bold_img.append(tmp_img)

        bold_img = nli.concat_imgs(bold_img)

        # extract roi data
        roi_dats = _extract_data(bold_img, mask_img, roi_fids)

    # calculate pattern similarity
    simi_all, vox_nums = _calculate_similarity(roi_dats, simi_type='correlation', fisher_z=True)

    # generate results table
    simi_df_all = []
    for simi, roi_id, hemi_id, vox_num in zip(simi_all, roi_ids, roi_hemis, vox_nums):

        if simi is not None:
            simi_df = pd.DataFrame({
                'subj_id': subj_id,
                'atlas_id': atlas_list[0],
                'roi_id': roi_id,
                'roi_hemi': hemi_id,
                'similarity': simi,
                'feature_num': vox_num,
                'pair_index': range(1, len(simi) + 1)
            })
            beh = pd.read_csv(beh_fid, delimiter='\t')
            simi_df_join = pd.concat([simi_df, beh], axis=1)
            simi_df_all.append(simi_df_join)
        else:
            simi_df = pd.DataFrame({
                'subj_id': subj_id,
                'atlas_id': atlas_list[0],
                'roi_id': roi_id,
                'roi_hemi': hemi_id,
                'similarity': [np.nan],
                'feature_num': vox_num,
                'pair_index': [np.nan]
            })
            beh = pd.read_csv(beh_fid, delimiter='\t')
            simi_df_join = pd.concat([simi_df, beh], axis=1)
            simi_df_all.append(simi_df_join)

    # merge table
    simi_df_all = pd.concat(simi_df_all)

    return simi_df_all

#Main
bold_space = 'T1w'
atlas_list = ['Harvard-Oxford']
method_flavors = ['leastsquare-all']
preproc_flavors = ['hp0p01-smooth2p0']
estimate_flavors = ['betas']


for method_id, preproc_id, estimate_id in product(method_flavors, preproc_flavors, estimate_flavors):

    # load roi description
    with open(deriv_dir.joinpath('roi', f'roi_description_atlas-Harvard-Oxford.json')) as fid:
        roi_info = json.load(fid)
    roi_list = roi_info['roi_info']
    roi_type = roi_info['type']
    roi_space = roi_info['space']

    roi_ids, roi_hemis = _split_roi_id(roi_list)

    # loop for subjects
    results = []
    out_fids = []
    pool = mp.Pool(processes=n_procs)
    for subj_id in subj_list: 

        # output file
        out_dir = deriv_dir.joinpath('pattern_similarity_LSA_Prewhiting', f'sub-{subj_id}',
                                     f'similarity_desc-roi_Harvard-Oxford-pearson') 
        out_dir.mkdir(exist_ok=True, parents=True)
        out_fid = out_dir.joinpath(
            f'sub-{subj_id}_space-{bold_space}_'
            f'desc-{method_id}-{preproc_id}-{estimate_id}-zr_similarity_LSA_Prewhiting.tsv')
        out_fids.append(out_fid)

        # input file list
        bold_fids = []
        res_fids = []
        bold_list=['task-semantic_run-1','task-semantic_run-2','task-semantic_run-3','task-semantic_run-4']
        for bold_id in bold_list:
            bold_fids.append(
                deriv_dir.joinpath(
                    f'parameters_singletrial-{method_id}', f'sub-{subj_id}', 'func',
                    f'sub-{subj_id}_{bold_id}_space-{bold_space}_'
                    f'desc-{preproc_id}_{estimate_id}.nii.gz'))
            ## input the residual file lists
            res_fids.append(
                deriv_dir.joinpath(
                    f'parameters_singletrial-{method_id}', f'sub-{subj_id}', 'func',
                    f'sub-{subj_id}_{bold_id}_space-{bold_space}_'
                    f'desc-{preproc_id}_res4d.nii.gz'))

        # mask file
        mask_fid = deriv_dir.joinpath(
            'preprocessed_data', f'sub-{subj_id}', 'func',
            f'sub-{subj_id}_space-{bold_space}_desc-brain-averaged_mask.nii.gz')

        # roi files
        roi_fids = []
        for roi_id, hemi_id in zip(roi_ids, roi_hemis):
            roi_fid = Path(
                deriv_dir, 'roi', f'sub-{subj_id}', roi_type,
                f'sub-{subj_id}_space-{roi_space}_atlas-{atlas_list[0]}_'
                f'roi-{roi_id}_desc-{hemi_id}_mask.nii.gz').as_posix()
            roi_fids.append(roi_fid)

        # beh files
        beh_fid = Path(deriv_dir, 'pattern_similarity', f'sub-{subj_id}', 'beh_labels', f'sub-{subj_id}_labels_pairwise.tsv').as_posix()

        # calculate similarity
        res = pool.apply_async(_pairwise_similarity,
                               (subj_id, bold_fids, res_fids, mask_fid, roi_fids,beh_fid))
        results.append(res)

    pool.close()
    pool.join()
    simi_all = [res.get() for res in results]

    # save results to tsv
    for simi, out_fid in zip(simi_all, out_fids):
        simi.to_csv(out_fid, sep='\t', index=None, float_format='%.6f', na_rep='n/a')
