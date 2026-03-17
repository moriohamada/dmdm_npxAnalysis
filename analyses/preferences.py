"""
Function for getting unit prefs (TF index, lick modulation, block pref?)
"""

import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import pickle
from config import ANALYSIS_OPTIONS, PATHS
from data.session import Session
from analyses.load_responses import load_psth
from utils.filing import get_response_files

from concurrent.futures import ProcessPoolExecutor
from functools import partial


def _load_condition_resp(psth_path: str,
                         event_type: str,
                         condition: str,
                         resp_win: list[float] | tuple[float],
                         ):

    arr, t = load_psth(psth_path, event_type, condition)
    t_mask = (t > resp_win[0]) & (t < resp_win[1])
    r = np.nanmean(arr[:, :, t_mask], axis=2).T  # nN x nEv
    return r

def _calculate_preference_index(r1: np.ndarray,
                                r2: np.ndarray,
                                stat: str = 'mean',
                                compute: str = 'index',
                                n_iter: int = 1000):
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

def _denorm(r: np.ndarray, means: np.ndarray, sds: np.ndarray):
    return r * sds[:,None] + means[:,None]

def _extract_tf_preference(psth_path, sess_data, ops):
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
        if sess_data.fr_normed:
            r1 = _denorm(r1, sess_data.fr_stats['mean'].values, sess_data.fr_stats['sd'].values)
            r2 = _denorm(r2, sess_data.fr_stats['mean'].values, sess_data.fr_stats['sd'].values)

        idx, p = _calculate_preference_index(r1, r2, stat='mean', compute='index',
                                             n_iter=ops['n_iter'])
        prefs[f'{name}_idx'] = idx
        prefs[f'{name}_p'] = p

    return prefs

def _extract_block_preference(psth_path, sess_data, ops):
    resp_win = ops['tf_context']
    prefs={}
    r1 = _load_condition_resp(psth_path, 'tf', 'earlyBlock_early_*', resp_win)
    r2 = _load_condition_resp(psth_path, 'tf', 'lateBlock_early_*', resp_win)
    if sess_data.fr_normed:
        r1 = _denorm(r1, sess_data.fr_stats['mean'].values, sess_data.fr_stats['sd'].values)
        r2 = _denorm(r2, sess_data.fr_stats['mean'].values, sess_data.fr_stats['sd'].values)

    block_idx, block_p = _calculate_preference_index(r2, r1, n_iter=ops['n_iter'])
    # also from pre-lick
    resp_win = ops['lick_bl']
    r1 = _load_condition_resp(psth_path, 'lick', 'earlyBlock_early_fa', resp_win)
    r2 = _load_condition_resp(psth_path, 'lick', 'lateBlock_early_fa', resp_win)
    if sess_data.fr_normed:
        r1 = _denorm(r1, sess_data.fr_stats['mean'].values, sess_data.fr_stats['sd'].values)
        r2 = _denorm(r2, sess_data.fr_stats['mean'].values, sess_data.fr_stats['sd'].values)

    block_lick_idx, block_lick_p = _calculate_preference_index(r2, r1, n_iter=ops['n_iter'])
    prefs = {
        'block_idx': block_idx,
        'block_p': block_p,
        'block_lick_idx': block_lick_idx,
        'block_lick_p': block_lick_p,
    }
    return prefs

def _extract_time_preference(psth_path, sess_data, ops):
    resp_win = ops['tf_context']
    r1 = _load_condition_resp(psth_path, 'tf', 'lateBlock_early_*', resp_win)
    r2 = _load_condition_resp(psth_path, 'tf', 'lateBlock_late_*', resp_win)

    if sess_data.fr_normed:
        r1 = _denorm(r1, sess_data.fr_stats['mean'].values, sess_data.fr_stats['sd'].values)
        r2 = _denorm(r2, sess_data.fr_stats['mean'].values, sess_data.fr_stats['sd'].values)

    idx, p = _calculate_preference_index(r2, r1, n_iter=ops['n_iter'])
    prefs = {'time_idx': idx,
             'time_p': p}
    return prefs

def _extract_lick_modulation(psth_path, sess_data, ops):
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

        if sess_data.fr_normed:
            r1 = _denorm(r1, sess_data.fr_stats['mean'].values, sess_data.fr_stats['sd'].values)
            r2 = _denorm(r2, sess_data.fr_stats['mean'].values, sess_data.fr_stats['sd'].values)

        idx, p = _calculate_preference_index(r2, r1, n_iter=ops['n_iter'])
        prefs[f'{name}_idx'] = idx
        prefs[f'{name}_p'] = p
    return prefs


def extract_preferences(psth_path: str,
                        sess_data: Session,
                        ops: dict = ANALYSIS_OPTIONS) -> pd.DataFrame:
    """ Main runner for extract all tf prefs for a single unit"""
    prefs = {}
    # tf pref - overall and by block
    prefs.update(_extract_tf_preference(psth_path, sess_data, ops))
    # block pref (tf context - early vs late block)
    prefs.update(_extract_block_preference(psth_path, sess_data, ops))
    # time pref (tf context - early vs late in trial, late block)
    prefs.update(_extract_time_preference(psth_path, sess_data, ops))
    # lick mod
    prefs.update(_extract_lick_modulation(psth_path, sess_data, ops))

    return pd.DataFrame(prefs)

def extract_all_unit_preferences(npx_dir: str = PATHS['npx_dir_local'],
                                 ops: dict = ANALYSIS_OPTIONS):
    psth_paths = get_response_files(npx_dir)

    for i, psth_path in enumerate(psth_paths):
        sess_data = Session.load(psth_path.replace('psths.h5', 'session.pkl'))

        print(f'{sess_data.animal}_{sess_data.name} ({i + 1}/{len(psth_paths)})')
        save_dir = Path(npx_dir) / sess_data.animal / sess_data.name

        prefs = extract_preferences(psth_path, sess_data, ops)

        prefs.to_csv(save_dir / 'preferences.csv', index=False)


def combine_preference_data(preference_paths: list[str]):
    """
    Combine preference data across all sessions
    """
    return pd.concat([pd.read_csv(path) for path in preference_paths], ignore_index=True)


def clean_preference_data(prefs: pd.DataFrame):
    for col in prefs.columns:
        if '_idx' in col:
            to_zero = (prefs[col].abs() >= 1) | prefs[col].isna()
            prefs.loc[to_zero, col] = 0
    return prefs

def identify_preference_sign_flippers(npx_dir: str = PATHS['npx_dir_local'],
                                      indexes: tuple[str,str] =  (
                                              'tf_earlyBlock_early', 'tf_lateBlock_early'),
                                      sig_flag: str = 'both',
                                      alpha: float = .05):
    psth_paths = get_response_files(npx_dir)
    pref_paths = [path.replace('psths.h5', 'preferences.csv') for path in psth_paths]
    sess_paths = [path.replace('psths.h5', 'session.pkl') for path in psth_paths]

    for i, path in enumerate(pref_paths):
        prefs = pd.read_csv(path)
        prefs = clean_preference_data(prefs)
        sess_data = Session.load(sess_paths[i])
        if sig_flag is None:
            mask = np.ones(len(prefs), dtype=bool)
        elif sig_flag == 'either':
            mask = (prefs[indexes[0] + '_p'] < alpha) | (prefs[indexes[1] + '_p'] < alpha)
        elif sig_flag == 'both':
            mask = (prefs[indexes[0] + '_p'] < alpha) & (prefs[indexes[1] + '_p'] < alpha)
        else:
            raise Exception(f"Invalid value for sig_flag: {sig_flag}")

        rel_pref = prefs.loc[mask, [idx + '_idx' for idx in indexes]]
        sign_flip = (rel_pref[indexes[0]+'_idx'] * rel_pref[indexes[1]+'_idx']) < 0
        flipper_ids = sign_flip[sign_flip].index
        if len(flipper_ids)>0:
            print(f'---{sess_data.animal}_{sess_data.name}---')
            units = sess_data.unit_info.loc[flipper_ids,:]
            print(units)


