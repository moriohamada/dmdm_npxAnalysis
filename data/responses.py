"""
Functions for extracting event-aligned neural responses
"""
import numpy as np
from scipy.stats import mode
import pandas as pd
from config import PATHS, ANALYSIS_OPTIONS
from data.stimulus import *
from data.session import Session
from functools import partial
from pathlib import Path
import h5py

def extract_all_timings(session: Session = None,
                        ops: dict = ANALYSIS_OPTIONS
                        ):
    """
    Get timings and info for task events of interest
    """
    session = get_baseline_onset_times(session)
    session = get_tf_outliers(session, ops)
    session = get_lick_onset_times(session)
    session = get_change_onset_times(session)

    return session

def get_event_aligned_responses(
        session: Session,
        ops: dict = ANALYSIS_OPTIONS,
        save_path: str = PATHS['npx_dir_local']):
    """
    Extract event-aligned firing rates for all units in given session.
    Saves a file with all events per event type, as well as an
    average_resp file containing mean responses for every event type.
    """

    psths = dict(blOn=dict(), bl=dict(), tf=dict(), ch=dict(), lick=dict(), abort=dict())
    t_ax = dict()

    psth_fn = partial(compute_psth,
        X=session.fr_matrix.values,
        t_ax=session.fr_matrix.columns.values
    )

    # Baseline onset
    resp_win = (-1, 2)
    valid    = session.bl_onsets['tr_dur'] > (resp_win[1] + ops['rmvTimeAround'])
    non_trans = session.bl_onsets['tr_in_block'] > ops['ignoreFirstBlockTrials']
    e_block  = (session.bl_onsets['block'] == 'early') & non_trans
    l_block  = (session.bl_onsets['block'] == 'late')  & non_trans

    for key, mask in {'early': e_block, 'late': l_block}.items():
        psths['blOn'][key], t_ax['blOn'] = psth_fn(
            event_times=session.bl_onsets.loc[mask & valid, 'time'],
            resp_win=resp_win
        )

    # Long baseline     resp_win = (-1, 5)
    valid    = session.bl_onsets['tr_dur'] > (resp_win[1] + ops['rmvTimeAround'])

    for key, mask in {'early': e_block, 'late': l_block}.items():
        psths['bl'][key], t_ax['bl'] = psth_fn(
            event_times=session.bl_onsets.loc[mask & valid, 'time'],
            resp_win=resp_win
        )

    # TF pulses
    resp_win  = (-0.5, 1.5)
    non_trans = session.tf_pulses['tr_in_block'] > ops['ignoreFirstBlockTrials']
    e_block   = (session.tf_pulses['block'] == 'early') & non_trans
    l_block   = (session.tf_pulses['block'] == 'late')  & non_trans
    early_tr  = ((session.tf_pulses['tr_time'] > ops['rmvTimeAround']) &
                 (session.tf_pulses['tr_time'] < ops['trSplitTime']))
    late_tr   = session.tf_pulses['tr_time'] > ops['trSplitTime']
    t_to_event = np.fmin(session.tf_pulses['time_to_lick'],
                         session.tf_pulses['time_to_abort'])
    valid     = ((session.tf_pulses['tr_time'] > ops['rmvTimeAround']) &
                 (t_to_event > ops['rmvTimeAround']))
    pos       = session.tf_pulses['tf'] > 0

    tf_conditions = {
        'earlyBlock_early_pos':  pos  & e_block & early_tr & valid,
        'earlyBlock_early_neg': ~pos  & e_block & early_tr & valid,
        'lateBlock_early_pos':   pos  & l_block & early_tr & valid,
        'lateBlock_early_neg':  ~pos  & l_block & early_tr & valid,
        'lateBlock_late_pos':    pos  & l_block & late_tr  & valid,
        'lateBlock_late_neg':   ~pos  & l_block & late_tr  & valid,
    }

    for key, mask in tf_conditions.items():
        psths['tf'][key], t_ax['tf'] = psth_fn(
            event_times=session.tf_pulses.loc[mask, 'time'],
            resp_win=resp_win
        )

    # Change onset
    resp_win = (-0.5, 1.5)
    ch_tfs = np.sort(session.ch_onsets['ch_tf'].unique())
    non_trans = session.ch_onsets['tr_in_block'] > ops['ignoreFirstBlockTrials']
    e_block = (session.ch_onsets['block'] == 'early') & non_trans
    l_block = (session.ch_onsets['block'] == 'late') & non_trans
    hit = session.ch_onsets['is_hit'] == 1
    block_masks = {'early': e_block, 'late': l_block}
    hit_masks = {'hit': hit, 'miss': ~hit}

    ch_conditions = {
        f'{block}_{outcome}_tf{ch}': block_mask & hit_mask & (
                    session.ch_onsets['ch_tf'] == ch)
        for ch in ch_tfs
        for block, block_mask in block_masks.items()
        for outcome, hit_mask in hit_masks.items()
    }

    for key, mask in ch_conditions.items():
        psths['ch'][key], t_ax['ch'] = psth_fn(
            event_times=session.ch_onsets.loc[mask, 'time'],
            resp_win=resp_win
        )

    # Lick times
    resp_win = (-1.5, 0.5)
    non_trans = session.lick_times['tr_in_block'] > ops['ignoreFirstBlockTrials']
    e_block = (session.lick_times['block'] == 'early') & non_trans
    l_block = (session.lick_times['block'] == 'late') & non_trans
    early_tr = ((session.lick_times['tr_time'] > ops['rmvTimeAround']) &
                (session.lick_times['tr_time'] < ops['trSplitTime']))
    late_tr = session.lick_times['tr_time'] > ops['trSplitTime']
    hit = session.lick_times['is_hit'] == 1
    fa = session.lick_times['is_FA'] == 1

    lick_conditions = {
        'earlyBlock_early_hit': e_block & early_tr & hit,
        'earlyBlock_early_fa': e_block & early_tr & fa,
        'lateBlock_early_hit': l_block & early_tr & hit,
        'lateBlock_early_fa': l_block & early_tr & fa,
        'lateBlock_late_hit': l_block & late_tr & hit,
        'lateBlock_late_fa': l_block & late_tr & fa,
    }

    for key, mask in lick_conditions.items():
        psths['lick'][key], t_ax['lick'] = psth_fn(
            event_times=session.lick_times.loc[mask, 'time'],
            resp_win=resp_win
        )

    # Save
    save_path = Path(save_path) / session.animal / session.name
    save_path.mkdir(parents=True, exist_ok=True)
    with h5py.File(save_path / 'psths.h5', 'w') as f:
        for event_type, conditions in psths.items():
            grp = f.create_group(event_type)
            mean_grp = f.create_group(f'{event_type}_mean')
            for condition, arr in conditions.items():
                grp.create_dataset(condition, data=arr)
                if arr.shape[0] == 0:
                    print(f'  Warning: no events for {event_type}/{condition}')
                    continue

                mean_grp.create_dataset(condition,
                                        data=np.nanmean(arr, axis=0))  # nN x nT
        t_ax_grp = f.create_group('t_ax')
        for event_type, ax in t_ax.items():
            t_ax_grp.create_dataset(event_type, data=ax)

def compute_psth(
        X: np.ndarray, # nN x nT
        t_ax: np.ndarray,
        event_times: pd.Series | np.ndarray | list[float],
        resp_win: tuple[float, float]
        ) -> (tuple[np.ndarray, np.ndarray]):
    """
    Takes FR matrix (nN x nT) and returns (nEv x nN x nT) array
    """
    nEv = len(event_times)
    nN = X.shape[0]
    dt = round(np.mean(np.diff(t_ax)), 3)
    ev_t_ax = np.arange(resp_win[0], resp_win[1], dt)
    nT = len(ev_t_ax)

    psth = np.zeros((nEv, nN, nT))

    for i, t in enumerate(event_times):
            idx = np.searchsorted(t_ax, t + resp_win[0])
            psth[i] = X[:, idx:idx + nT]

    return psth, ev_t_ax




