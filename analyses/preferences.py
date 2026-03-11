"""
Function for getting unit prefs (TF index, lick modulation, block pref?)
"""

import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from config import ANALYSIS_OPTIONS, PATHS, PLOT_COLOURS
from data.session import Session
from analyses.load_responses import load_psth
from utils.filing import get_response_files

from concurrent.futures import ProcessPoolExecutor
from functools import partial


def _load_condition_resp(psth_path: str,
                         event_type: str,
                         condition: str,
                         resp_win: list[float] | tuple[float]):

    arr, t = load_psth(psth_path, event_type, condition)
    t_mask = (t > resp_win[0]) & (t < resp_win[1])
    r = np.nanmean(arr[:, :, t_mask], axis=2).T  # nN x nEv
    return r

def _calculate_preference_index(r1: np.ndarray,
                                r2: np.ndarray,
                                stat: str = 'mean',
                                compute: str = 'index',
                                n_iter: int = 1000) -> (int, float):
    """
    Calculate preference index and signficance via permutation test.
    Args:
        r1: nN x nEv array of response values for event 1
        r2: nN x nEv array of response values for event 2
        stat: 'mean' or 'median'; summary stat to compare between groups
        compute: 'index' or 'delta'
        n_iter: number of shuffling iterations
    Returns:
        measure: normalized index ((r1 - r2)/(r1 + r2)) or just difference (r1 - r2),
        depending on 'stat' arg
        p: p value from permutation test. this is only done on (r1-r2),
        since denominator remains unchanged after shuffling
    """
    if stat == 'mean':
        fn = np.nanmean
    elif stat == 'median':
        fn = np.nanmedian
    else:
        raise NotImplementedError

    nN = r1.shape[0]

    s1 = fn(r1, axis=1)  # (nN,)
    s2 = fn(r2, axis=1)  # (nN,)

    delta = s1 - s2
    if compute == 'index':
        measure = delta / (s1 + s2)
    elif compute == 'delta':
        measure = delta
    else:
        raise NotImplementedError

    # Permutation test on delta
    n1 = r1.shape[1]
    pooled = np.concatenate([r1, r2], axis=1)  # nN x (nEv1 + nEv2)
    n_total = pooled.shape[1]

    null_delta = np.zeros((nN, n_iter))
    for i in range(n_iter):
        idx = np.random.permutation(n_total)
        shuf1 = pooled[:, idx[:n1]]
        shuf2 = pooled[:, idx[n1:]]
        null_delta[:, i] = fn(shuf1, axis=1) - fn(shuf2, axis=1)

    # Two-tailed p-value
    p = np.mean(np.abs(null_delta) >= np.abs(delta)[:, None], axis=1)

    return measure, p


def _extract_tf_preference(psth_path, ops):
    resp_win = ops['tf_resp_win']
    comps = {
        'tf': ['*_pos', '*_neg'],
        'tf_earlyBlock_early': ['earlyBlock_early_pos', 'earlyBlock_early_neg'],
        'tf_lateBlock_early': ['lateBlock_early_pos', 'lateBlock_early_neg'],
        'tf_lateBlock_late': ['lateBlock_late_pos', 'lateBlock_late_neg'],
    }
    prefs = {}
    for name, conds in comps.items():
        r1 = _load_condition_resp(psth_path, 'tf', conds[0], resp_win)
        r2 = _load_condition_resp(psth_path, 'tf', conds[1], resp_win)
        idx, p = _calculate_preference_index(r1, r2, stat='mean', compute='index',
                                             n_iter=ops['n_iter'])
        prefs[f'{name}_idx'] = idx
        prefs[f'{name}_p'] = p

    return prefs

def _extract_block_preference(psth_path, ops):
    resp_win = ops['tf_context']
    prefs={}
    r1 = _load_condition_resp(psth_path, 'tf', 'earlyBlock_early_*', resp_win)
    r2 = _load_condition_resp(psth_path, 'tf', 'lateBlock_early_*', resp_win)
    block_idx, block_p = _calculate_preference_index(r2, r1, n_iter=ops['n_iter'])
    # also from pre-lick
    resp_win = ops['lick_bl']
    r1 = _load_condition_resp(psth_path, 'lick', 'earlyBlock_early_fa', resp_win)
    r2 = _load_condition_resp(psth_path, 'lick', 'lateBlock_early_fa', resp_win)
    block_lick_idx, block_lick_p = _calculate_preference_index(r2, r1, n_iter=ops['n_iter'])
    prefs = {
        'block_idx': block_idx,
        'block_p': block_p,
        'block_lick_idx': block_lick_idx,
        'block_lick_p': block_lick_p,
    }
    return prefs

def _extract_time_preference(psth_path, ops):
    resp_win = ops['tf_context']
    r1 = _load_condition_resp(psth_path, 'tf', 'lateBlock_early_*', resp_win)
    r2 = _load_condition_resp(psth_path, 'tf', 'lateBlock_late_*', resp_win)
    idx, p = _calculate_preference_index(r2, r1, n_iter=ops['n_iter'])
    prefs = {'time_idx': idx,
             'time_p': p}
    return prefs

def _extract_lick_modulation(psth_path, ops):
    comps = {
        'lick': '*_fa',
        'lick_earlyBlock_early': 'earlyBlock_early_fa',
        'lick_lateBlock_early': 'lateBlock_early_fa',
        'lick_lateBlock_late': 'lateBlock_late_fa'
    }
    prefs = {}
    win_pre = ops['lick_bl']
    win_lick = ops['lick_pre']
    for name, cond in comps.items():
        r1 = _load_condition_resp(psth_path, 'lick', cond, resp_win=win_pre)
        r2 = _load_condition_resp(psth_path, 'lick', cond, resp_win=win_lick)
        idx, p = _calculate_preference_index(r2, r1, n_iter=ops['n_iter'])
        prefs[f'{name}_idx'] = idx
        prefs[f'{name}_p'] = p
    return prefs


def extract_preferences(psth_path: str,
                        ops: dict = ANALYSIS_OPTIONS) -> pd.DataFrame:
    """ Main runner for extract all tf prefs for a single unit"""
    prefs = {}
    # tf pref - overall and by block
    prefs.update(_extract_tf_preference(psth_path, ops))
    # block pref (tf context - early vs late block)
    prefs.update(_extract_block_preference(psth_path, ops))
    # time pref (tf context - early vs late in trial, late block)
    prefs.update(_extract_time_preference(psth_path, ops))
    # lick mod
    prefs.update(_extract_lick_modulation(psth_path, ops))

    return pd.DataFrame(prefs)

def extract_all_unit_preferences(npx_dir: str = PATHS['npx_dir_local'],
                                 ops: dict = ANALYSIS_OPTIONS):
    psth_paths = get_response_files(npx_dir)

    for psth_path in psth_paths:
        sess_data = Session.load(psth_path.replace('psths.h5', 'session.pkl'))
        save_dir = Path(npx_dir) / sess_data.animal / sess_data.name

        prefs = extract_preferences(psth_path, ops)

        prefs.to_csv(save_dir / 'preferences.csv', index=False)

