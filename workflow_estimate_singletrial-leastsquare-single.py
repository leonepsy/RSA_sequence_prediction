# -*- coding: utf-8 -*-
"""Estimate single-trial response (least square single)."""

import argparse
from pathlib import Path

from nipype.pipeline import engine as pe
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu

from nipype.interfaces import fsl
from nipype.algorithms.modelgen import SpecifyModel

parser = argparse.ArgumentParser(description='Perform single-trial estimation (LSS).')
parser.add_argument(
    'bids_dir',
    action='store',
    help='The root folder of a BIDS dataset.')
parser.add_argument(
    'output_dir',
    action='store',
    help='The directory where the output files should be stored.')
parser.add_argument(
    'analysis_level',
    action='store',
    help='processing stage to be run, only "participant" in the case of this pipeline.',
    choices=['participant'])
parser.add_argument(
    '--participant_label',
    action='store',
    nargs='+',
    help='One or more participant identifiers (the sub- prefix should be removed).')
parser.add_argument(
    '--task-id',
    required=True,
    action='store',
    help=('Select a specific task to be processed.'))
parser.add_argument(
    '--run-id',
    action='store',
    nargs='+',
    default=['single_run'],
    help='Select specific run to be processed.')
parser.add_argument(
    '--data-id',
    required=True,
    action='store',
    help=('Select folder for input brain data. The folder should under derivatives folder.'))
parser.add_argument(
    '--data-desc',
    required=True,
    action='store',
    help=('File identifier corresponding to desc field.'))
parser.add_argument(
    '--bold-space',
    action='store',
    choices=['MNI152NLin2009cAsym', 'T1w'],
    help='The space which BOLD data aligned.')
parser.add_argument(
    '--brainmask',
    action='store',
    default=['fmriprep'],
    nargs='+',
    help=('Select brainmask for functional images relative to derivatives directory.'
          'eg., brainmask/sub-01/...'
          'If only one mask was given, it will use for all BOLD files specified.'
          'Otherwise, the given the number of mask file must match the number of BOLD file.'
          'If brainmask was not specified, the default is the brainmask derived from fMRIPrep.'))
parser.add_argument(
    '--repetition-time',
    action='store',
    type=float,
    default=2.0,
    help='Repetition time of BOLD images.')
parser.add_argument(
    '--highpass',
    action='store',
    type=float,
    default=0.01,
    help='Highpass temporal filter frequency (in herzt).')
parser.add_argument(
    '--n_procs',
    action='store',
    default=1,
    type=int,
    help='Maximum number of threads across all processes.')
parser.add_argument(
    '--work-dir',
    action='store',
    default='works',
    help='Path where intermediate results should be stored.')
args = parser.parse_args()


# ----------------
# Helper functions
# ----------------
def _prefix_id(subj_id):
    return f'sub-{subj_id}'


def _create_lss_regressor(regressors_file):
    import simplejson as json
    import numpy as np
    from nipype.interfaces.base import Bunch

    with open(regressors_file) as f:
        spec = json.load(f)

    # get model information
    # group_index should be a list with same length of main regressor. It should be 1-based and
    # indicates how to group other trials into other_* regressors.
    # item_index should be a list with same length of main regressor. It should be 1-based and
    # indicates which trials belong to one item need to be estimated together.
    if len(set(spec['item_index'])) == 1:
        n_regressors = len(spec['main']['onsets'])
        query_list = np.array(range(n_regressors))
    else:
        n_regressors = len(np.unique(np.array(spec['item_index'])))
        query_list = np.array(spec['item_index']) - 1
    n_groups = len(set(spec['group_index'])) 
    group_index = np.array(spec['group_index'])

    all_infos = []
    # main regressors
    raw_onsets = np.array(spec['main']['onsets'])
    raw_durations = np.array(spec['main']['durations'])
    raw_amplitudes = np.array(spec['main']['amplitudes'])
    for regressor_id in range(n_regressors):

        conditions = ['main'] + [f'other_{x+1:03d}' for x in list(range(n_groups))] 
        onsets = [[] for x in range(n_groups + 1)] 
        durations = [[] for x in range(n_groups + 1)]
        amplitudes = [[] for x in range(n_groups + 1)]

        # target regressor
        onsets[0] += raw_onsets[query_list == regressor_id].tolist() 
        durations[0] += raw_durations[query_list == regressor_id].tolist()
        amplitudes[0] += raw_amplitudes[query_list == regressor_id].tolist()

        # other main regressors
        for group_id in range(n_groups):
            onsets[group_id + 1] += raw_onsets[(query_list != regressor_id)
                                               & (group_index == (group_id + 1))].tolist()
            durations[group_id + 1] += raw_durations[(query_list != regressor_id)
                                                     & (group_index == (group_id + 1))].tolist()
            amplitudes[group_id + 1] += raw_amplitudes[(query_list != regressor_id)
                                                       & (group_index == (group_id + 1))].tolist()

        # additional regressors (need to convolve with HRF)
        if spec['extra_evs_convolution'] is not None:
            for regressor_name, regressor_dict in spec['extra_evs_convolution'].items():
                conditions = conditions + [regressor_name]
                onsets.append(regressor_dict['onsets'])
                durations.append(regressor_dict['durations'])
                amplitudes.append(regressor_dict['amplitudes'])

        # additional regressors (doesn't need to convolve with HRF)
        regressor_names = []
        regressors = []
        if spec['extra_evs_no-convolution'] is not None:
            for regressor_name, regressor_values in spec['extra_evs_no-convolution'].items():
                regressor_names.append(regressor_name)
                regressors.append(regressor_values)

        # confounds
        if spec['confounds'] is not None:
            for confound_name, confound_value in spec['confounds'].items():
                regressor_names.append(confound_name)
                regressors.append(confound_value)

        # set regressors variables to None if there are no extra regressors
        if len(regressor_names) == 0:
            regressor_names = None
            regressors = None

        # organize into Bunch object
        subject_info = Bunch(
            conditions=conditions,
            onsets=onsets,
            durations=durations,
            amplitudes=amplitudes,
            regressor_names=regressor_names,
            regressors=regressors,
            tmod=None,
            pmod=None)
        all_infos.append(subject_info)

    return all_infos


def _create_lss_contrast(regressors_file):

    # simply assume the regressor of interest is the first one named 'main'
    contrast_info = [('target', 'T', ['main'], [1.0])]

    return contrast_info


def _select_pe(in_list, idx):

    num_pe = len(in_list)
    out_list = []
    for pe_id in range(num_pe):
        out_list.append(in_list[pe_id][idx])

    return out_list


def _output_filename(in_file, subj_id, task_id, run_id, bold_space, data_desc, suffix):
    import os
    from os.path import join as opj
    from shutil import copyfile

    out_file = opj(os.getcwd(),
                   (f'sub-{subj_id}_task-{task_id}_{run_id}'
                    f'space-{bold_space}_desc-{data_desc}_{suffix}.nii.gz'))

    copyfile(in_file, out_file)

    return out_file


# -----------------------
# Setup basic inforamtion
# -----------------------

# Directories information
bids_dir = Path(args.bids_dir)
output_dir = Path(args.output_dir)
deriv_dir = bids_dir.joinpath('derivatives')
log_dir = bids_dir.joinpath('logs')
log_dir.mkdir(exist_ok=True)

# Participant information
if args.participant_label:
    subj_list = args.participant_label
else:
    subj_dirs = sorted(list(deriv_dir.joinpath(args.data_id).glob('sub-*')))
    subj_list = [subj_dir.split('-')[-1] for subj_dir in subj_dirs]

# Task specific information
analysis_id = output_dir.name
task_id = args.task_id
run_id = args.run_id
data_id = args.data_id
data_desc = args.data_desc
bold_space = args.bold_space
brainmask = args.brainmask
repetition_time = args.repetition_time
highpass_cutoff = (1 / args.highpass)
film_threshold = 1000
film_ms = 5

if run_id == ['single_run']:
    multiple_mask = False
elif brainmask == ['fmriprep'] and len(run_id) > 1:
    multiple_mask = True
elif isinstance(brainmask, list) and len(brainmask) > 1:
    multiple_mask = True
else:
    multiple_mask = False

if run_id != ['single_run']:
    run_id = [f'run-{x}_' for x in run_id]

# Pipeline specific information
n_procs = int(args.n_procs)
if args.work_dir is 'works':
    work_dir = bids_dir / 'works'
else:
    work_dir = Path(args.work_dir)
work_dir.mkdir(exist_ok=True)

# -------------------
# Preprocess pipeline
# -------------------
for subj_id in subj_list:

    wf = pe.Workflow(name=f'single_subject_{subj_id}_wf')
    wf.base_dir = work_dir

    # Inputnode
    inputnode = pe.Node(
        interface=niu.IdentityInterface(
            fields=['analysis_id', 'subj_id', 'task_id', 'run_id', 'data_id', 'data_desc',
                    'bold_space', 'brainmask']),
        name="inputnode")
    inputnode.inputs.analysis_id = analysis_id
    inputnode.inputs.subj_id = subj_id
    inputnode.inputs.task_id = task_id
    inputnode.inputs.run_id = run_id
    inputnode.inputs.data_id = data_id
    inputnode.inputs.data_desc = data_desc
    inputnode.inputs.bold_space = bold_space
    if run_id == ['single_run']:
        inputnode.inputs.run_id = ''
        inputnode.inputs.brainmask = brainmask
    elif multiple_mask and (brainmask != ['fmriprep']):
        inputnode.iterables = [('run_id', run_id),
                               ('brainmask', brainmask)]
        inputnode.synchronize = True
    else:
        inputnode.iterables = [('run_id', run_id)]
        inputnode.inputs.brainmask = brainmask

    # Grab input data
    datasource = pe.Node(
        interface=nio.DataGrabber(
            infields=[
                'analysis_id', 'subj_id', 'task_id', 'run_id', 'data_id', 'data_desc',
                'bold_space', 'brainmask'
            ],
            outfields=['func', 'brainmask', 'regressors']),
        name='datasource')
    datasource.inputs.base_directory = deriv_dir.as_posix()
    datasource.inputs.template = '*'
    datasource.inputs.sort_filelist = True
    datasource.inputs.field_template = dict(
        func='%s/sub-%s/func/sub-%s_task-%s_%sspace-%s_desc-%s_bold.nii.gz',
        regressors='%s/sub-%s/modelspec/sub-%s_task-%s_%sdesc-leastsquare-separate_modelspec.json')
    datasource.inputs.template_args = dict(
        func=[['data_id', 'subj_id', 'subj_id', 'task_id', 'run_id', 'bold_space', 'data_desc']],
        regressors=[['analysis_id', 'subj_id', 'subj_id', 'task_id', 'run_id']])
    if brainmask == ['fmriprep']:
        datasource.inputs.field_template.update({
            'brainmask': 'fmriprep/sub-%s/func/sub-%s_task-%s_%sspace-%s_desc-brain_mask.nii.gz'
        })
        datasource.inputs.template_args.update({
            'brainmask': [['subj_id', 'subj_id', 'task_id', 'run_id', 'bold_space']]
        })
    else:
        datasource.inputs.field_template.update({'brainmask': '%s'})
        datasource.inputs.template_args.update({'brainmask': [['brainmask']]})

    # Generate singletrial model regressors
    create_regressor = pe.Node(
        interface=niu.Function(
            input_names=['regressors_file'],
            output_names=['subject_info'],
            function=_create_lss_regressor),
        name='create_regressors')
    # Generate singletrial model contrasts
    create_contrasts = pe.Node(
        interface=niu.Function(
            input_names=['regressors_file'],
            output_names=['contrast_info'],
            function=_create_lss_contrast),
        name='create_contrast')
    # Mask functional image
    masked_func = pe.Node(interface=fsl.ApplyMask(), name='skullstrip_func')
    # Generate singletrial model infos
    model_spec = pe.MapNode(
        interface=SpecifyModel(),
        iterfield=['subject_info'],
        name='create_model_infos')
    model_spec.inputs.input_units = 'secs'
    model_spec.inputs.time_repetition = repetition_time
    model_spec.inputs.high_pass_filter_cutoff = highpass_cutoff
    # Generate singletrial model designs
    design_lss = pe.MapNode(
        interface=fsl.Level1Design(),
        iterfield=['session_info'],
        name='create_feat_designs')
    design_lss.inputs.interscan_interval = repetition_time
    design_lss.inputs.bases = {'dgamma': {'derivs': False}}
    design_lss.inputs.model_serial_correlations = True
    # Generate singletrial model
    model_lss = pe.MapNode(
        interface=fsl.FEATModel(),
        iterfield=['fsf_file', 'ev_files'],
        name='create_models')
    # Estimate singletrial model
    model_estimate = pe.MapNode(
        interface=fsl.FILMGLS(),
        iterfield=['design_file', 'tcon_file'],
        name='estimate_models')
    model_estimate.inputs.smooth_autocorr = True
    model_estimate.inputs.mask_size = film_ms
    model_estimate.inputs.threshold = film_threshold

    # Merge estimated parameters
    merge_betas = pe.Node(interface=fsl.Merge(), name='merge_betas')
    merge_betas.inputs.dimension = 't'
    # Merge tstats
    merge_tstats = pe.Node(interface=fsl.Merge(), name='merge_tstats')
    merge_tstats.inputs.dimension = 't'
    # rename estimated parameters
    rename_betas = pe.Node(
        interface=niu.Function(
            input_names=['in_file', 'subj_id', 'task_id', 'run_id', 'bold_space',
                         'data_desc', 'suffix'],
            output_names=['out_file'],
            function=_output_filename),
        name='rename_betas')
    rename_betas.inputs.suffix = 'betas'
    # rename tstats
    rename_tstats = pe.Node(
        interface=niu.Function(
            input_names=['in_file', 'subj_id', 'task_id', 'run_id', 'bold_space',
                         'data_desc', 'suffix'],
            output_names=['out_file'],
            function=_output_filename),
        name='rename_tstats')
    rename_tstats.inputs.suffix = 'tstats'

    # Collect each run's results
    datasink = pe.Node(interface=nio.DataSink(), name='datasink')
    datasink.inputs.base_directory = output_dir.as_posix()
    datasink.inputs.parameterization = False

    wf.connect([
        (inputnode, datasource, [('analysis_id', 'analysis_id'),
                                 ('subj_id', 'subj_id'),
                                 ('task_id', 'task_id'),
                                 ('run_id', 'run_id'),
                                 ('data_id', 'data_id'),
                                 ('data_desc', 'data_desc'),
                                 ('bold_space', 'bold_space'),
                                 ('brainmask', 'brainmask')]),
        (inputnode, rename_betas, [('subj_id', 'subj_id'),
                                   ('task_id', 'task_id'),
                                   ('run_id', 'run_id'),
                                   ('bold_space', 'bold_space'),
                                   ('data_desc', 'data_desc')]),
        (inputnode, rename_tstats, [('subj_id', 'subj_id'),
                                    ('task_id', 'task_id'),
                                    ('run_id', 'run_id'),
                                    ('bold_space', 'bold_space'),
                                    ('data_desc', 'data_desc')]),
        (inputnode, datasink, [(('subj_id', _prefix_id), 'container')]),
        (datasource, create_regressor, [('regressors', 'regressors_file')]),
        (datasource, create_contrasts, [('regressors', 'regressors_file')]),
        (datasource, masked_func, [('func', 'in_file'),
                                   ('brainmask', 'mask_file')]),
        (masked_func, model_spec, [('out_file', 'functional_runs')]),
        (masked_func, model_estimate, [('out_file', 'in_file')]),
        (create_regressor, model_spec, [('subject_info', 'subject_info')]),
        (model_spec, design_lss, [('session_info', 'session_info')]),
        (create_contrasts, design_lss, [('contrast_info', 'contrasts')]),
        (design_lss, model_lss, [('ev_files', 'ev_files'),
                                 ('fsf_files', 'fsf_file')]),
        (model_lss, model_estimate, [('design_file', 'design_file'),
                                     ('con_file', 'tcon_file')]),
        (model_estimate, merge_betas, [(('param_estimates', _select_pe, 0), 'in_files')]),
        (model_estimate, merge_tstats, [('tstats', 'in_files')]),
        (merge_betas, rename_betas, [('merged_file', 'in_file')]),
        (merge_tstats, rename_tstats, [('merged_file', 'in_file')]),
        (rename_betas, datasink, [('out_file', 'func.@pes_file')]),
        (rename_tstats, datasink, [('out_file', 'func.@tmaps_file')])
    ])

    # Run workflow
    wf.write_graph(graph2use='colored')
    wf.config['logging'] = {'log_to_file': 'true', 'log_directory': log_dir}
    wf.config['execution'] = {
        'stop_on_first_crash': 'true',
        'crashfile_format': 'txt',
        'crashdump_dir': log_dir,
        'job_finished_timeout': '650'
    }
    wf.config['monitoring'] = {'enabled': 'true'}
    wf.run(plugin='MultiProc', plugin_args={'n_procs': n_procs})
