import numpy as np
import pandas as pd
import os
import gc

from config import ANALYSIS_OPTIONS
from utils.filing import load_fr_matrix


def downsample_bins(data: np.ndarray|pd.DataFrame, factor, axis=-1):
    """
    Downsample by specified factor.
    Truncates any remainder bins at the end.
    Works on ndarrays and pandas dataframes (averages columns in latter case).
    """
    factor = int(round(factor))
    if factor <= 1:
        return data

    is_df = isinstance(data, pd.DataFrame)
    if is_df:
        arr = data.values
        cols = data.columns.values
        n_keep = (len(cols) // factor) * factor
        new_cols = cols[:n_keep].reshape(-1, factor).mean(axis=1)
        new_vals = arr[:, :n_keep].reshape(arr.shape[0], -1, factor).mean(axis=2)
        return pd.DataFrame(new_vals, index=data.index, columns=new_cols)

    arr = np.asarray(data)
    n = arr.shape[axis]
    n_keep = (n // factor) * factor
    slc = [slice(None)] * arr.ndim
    slc[axis] = slice(0, n_keep)
    trimmed = arr[tuple(slc)]
    new_shape = list(trimmed.shape)
    new_shape[axis] = n_keep // factor
    new_shape.insert(axis + 1, factor)
    return trimmed.reshape(new_shape).mean(axis=axis + 1)

def downsample_session(fr_path, ds_path, ds_factor):
    """Downsample a single session FR matrix and save."""
    fr = load_fr_matrix(fr_path)
    fr_ds = downsample_bins(fr, ds_factor)
    fr_ds.columns = np.round(fr_ds.columns.values, 4)
    fr_ds.to_parquet(ds_path)


def _downsample_session_wrapper(args):
    """Wrapper for downsample_session"""
    downsample_session(*args)


def save_downsampled_fr(npx_dir, ops=ANALYSIS_OPTIONS, n_workers=1):
    """
    Pre-downsample all FR_matrix.parquet files and save as FR_matrix_ds.parquet.
    Skips sessions that already have a downsampled file.
    Annoyingly, ran into issue with memory accumulation (del/gc.collect didnt solve -
    so this is run as a parallel process - req even if n_workers=1).
    """
    ds_factor = round(ops['pop_bin_width'] / ops['sp_bin_width'])
    if ds_factor <= 1:
        print('No downsampling needed: pop_bin_width = sp_bin_width (see config.py)')
        return

    jobs = []
    for subj in os.listdir(npx_dir):
        subj_dir = os.path.join(npx_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for sess in os.listdir(subj_dir):
            sess_dir = os.path.join(subj_dir, sess)
            fr_path = os.path.join(sess_dir, 'FR_matrix.parquet')
            ds_path = os.path.join(sess_dir, 'FR_matrix_ds.parquet')

            if not os.path.exists(fr_path):
                continue
            if os.path.exists(ds_path):
                print(f'    Data already downsampled - skipping...')
                continue

            jobs.append((fr_path, ds_path, ds_factor))

    print(f'{len(jobs)} sessions to downsample')
    from multiprocessing import Pool
    with Pool(n_workers, maxtasksperchild=1) as pool:
        for i, _ in enumerate(pool.imap(_downsample_session_wrapper, jobs)):
            print(f'  {i+1}/{len(jobs)} done')
