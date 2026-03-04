"""
Functions for extracting full FR matrix (nN x nT) + neural info.
"""

from config import PATHS, ANALYSIS_OPTIONS
from data.session import Session
from data.stimulus import get_trials_from_block_start
from data.responses import extract_all_timings, get_event_aligned_responses
from utils.smoothing import causal_boxcar
from utils.norm import zscore_fr
import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from scipy.stats import zscore


def extract_FR_matrix(neural: pd.DataFrame,
                      bin_size: float = ANALYSIS_OPTIONS[ 'spBinWidth'],
                      smoothing: float | None = ANALYSIS_OPTIONS['spSmoothWidth'],
                      normalize: bool = True) -> pd.DataFrame:

    t_start = neural['spike_time'].min()
    t_end   = neural['spike_time'].max()
    bins        = np.arange(t_start - 1e-10, t_end + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2

    neural = neural.copy()
    neural['time_bin'] = pd.cut(neural['spike_time'], bins=bins, labels=bin_centers)

    fr_matrix = (neural.groupby(['cluster_id', 'time_bin'], observed=True)
                 .size()
                 .unstack(fill_value=0)
                 .reindex(columns=bin_centers, fill_value=0)
                 ) / bin_size

    if smoothing is not None and smoothing > 0:
        fr_matrix = causal_boxcar(fr_matrix, smoothing / bin_size)

    if normalize:
        fr_matrix = pd.DataFrame(
            zscore_fr(fr_matrix.values),
            index=fr_matrix.index,
            columns=fr_matrix.columns
        )

    return fr_matrix


def save_fr_matrix(fr_matrix: pd.DataFrame,
                   save_path: str):

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    fr_matrix.to_parquet(save_path)


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

            session = Session.from_folder(sess_folder)

            # add some useful columns to trials
            session = get_trials_from_block_start(session)


            if not session.has_neural:
                continue

            session.fr_matrix = extract_FR_matrix(session.neural,
                                                  bin_size=ops['spBinWidth'],
                                                  normalize=True)

            save_path = (sess_folder.replace(npx_dir_ceph, npx_dir_local) +
                         '/FR_matrix.parquet')
            save_fr_matrix(session.fr_matrix, save_path)

            # extract event-aligned responses, averages
            session = extract_all_timings(session, ops)
            get_event_aligned_responses(session, ops)

            # save session
            session.save(save_path)
