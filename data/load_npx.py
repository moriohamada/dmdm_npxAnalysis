"""
Functions for extracting full FR matrix (nN x nT) + neural info,
"""

from config import PATHS, ANALYSIS_OPTIONS
from utils.smoothing import causal_boxcar
from responses import calculate_event_aligned_
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from scipy.stats import zscore

def load_session(sess_folder: str):

    trials = pd.read_parquet(os.path.join(sess_folder, 'trials.parquet'))

    if os.path.isfile(os.path.join(sess_folder, 'neural.parquet')): # recording session
        neural = pd.read_parquet(os.path.join(sess_folder, 'neural.parquet')) \
            .drop(columns=['brain_region', 'x', 'y', 'z'])
        daq = pd.read_parquet(os.path.join(sess_folder, 'daq.parquet'))
        move = pickle.load(open(os.path.join(sess_folder, 'movement.pkl'), 'rb'))

        return trials, daq, move, neural

    else:
        return trials


def extract_FR_matrix(neural: pd.DataFrame,
                      bin_size: float = ANALYSIS_OPTIONS['sp_bin_width'],
                      smoothing: float|None = ANALYSIS_OPTIONS['sp_smooth_width'],
                      normalize: bool = False):

    t_start = neural['spike_time'].min()
    t_end = neural['spike_time'].max()
    bins = np.arange(t_start-1e-10, t_end + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2

    # assign spikes to time bins
    neural['time_bin'] = pd.cut(neural['spike_time'], bins=bins, labels=bin_centers)

    # count spikes per bin
    spike_counts = (neural.groupby(['cluster_id', 'time_bin'], observed=True)
                    .size()
                    .unstack(fill_value=0)
                    .reindex(columns=bin_centers, fill_value=0)
                    )

    # convert to FR matrix
    fr_matrix = spike_counts / bin_size

    # smoothing
    if smoothing is not None and smoothing > 0:
        fr_matrix = causal_boxcar(fr_matrix, smoothing/bin_size)

    # normalize
    if normalize:
        fr_matrix.apply(zscore, axis=1)

    return fr_matrix

def save_fr_matrix(fr_matrix: pd.DataFrame,
                   fr_save_path: str):

    save_dir = Path(fr_save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    fr_matrix.to_parquet(fr_save_path, index=False)


def extract_session_data(npx_dir_ceph: str = PATHS['npx_dir_ceph'],
                         npx_dir_local: str = PATHS['npx_dir_local'],
                         ops: dict = ANALYSIS_OPTIONS):
    """
    Loop through each session and:
    1) extract firing rate matrix for entire session
    2) extract event-aligned responses
    3) save averaged event-aligned responses
    """

    subj_folders = [f.path for f in os.scandir(npx_dir_ceph) if f.is_dir()]
    for subj_folder in subj_folders:
        sess_folders = [f.path for f in os.scandir(subj_folder) if f.is_dir()]
        for sess_folder in sess_folders:
            if os.path.isfile(os.path.join(sess_folder, 'neural.parquet')):
                trials, daq, move, neural = load_session(sess_folder)

                # extract and save fr matrix
                fr_matrix = extract_FR_matrix(neural=neural, bin_size=ops['sp_bin_width'])
                fr_save_path = sess_folder.replace(npx_dir_ceph, npx_dir_local) + '/FR_matrix.parquet'
                save_fr_matrix(fr_matrix, fr_save_path)

                # extract event-aligned responses, averages





