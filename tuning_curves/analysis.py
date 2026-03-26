"""single-unit TF tuning curves by block: extraction, OLS fitting, permutation tests"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import gc

from config import PATHS, ANALYSIS_OPTIONS, TUNING_CURVE_OPS
from data.session import Session
from utils.filing import get_session_dirs_by_animal, load_fr_matrix


#%% shared utilities

def _get_all_tf_pulses(session):
    """
    get times and TF values for ALL pulses (not just outliers) from session trials.
    same logic as data.stimulus.get_tf_outliers but without the outlier threshold
    """
    tfs, times, tr_times, trials, blocks = [], [], [], [], []
    tr_in_blocks, time_to_licks, time_to_aborts = [], [], []

    for tr, row in session.trials.iterrows():
        block = row['hazardblock']
        tr_in_block = row['tr_in_block']

        if row['IsFA']:
            tr_lick = row['motion_onset'] if not pd.isna(row['motion_onset']) \
                else (row['Baseline_ON_rise'] + row['rt_FA'])
        elif row['IsHit']:
            tr_lick = row['motion_onset']
        else:
            tr_lick = np.nan
        tr_abort = row['rt_abort']

        tf_seq = row['TF'][row['TF'].nonzero()]
        ch_t = row['stimT']
        ch_fr = round(ch_t * 60)
        bl_tf = np.log2(tf_seq[:ch_fr:3])
        fr_t = row['frame_time'][~np.isnan(row['frame_time'])][:ch_fr:3]

        if len(fr_t) != len(bl_tf):
            continue

        bl_onset = row['Baseline_ON_rise']
        for i in range(len(bl_tf)):
            tfs.append(bl_tf[i])
            times.append(fr_t[i])
            tr_times.append(fr_t[i] - bl_onset)
            trials.append(tr)
            blocks.append(block)
            tr_in_blocks.append(tr_in_block)
            time_to_licks.append(tr_lick - fr_t[i])
            time_to_aborts.append(tr_abort - fr_t[i])

    return pd.DataFrame({
        'tf': tfs, 'time': times, 'tr_time': tr_times,
        'trial': trials, 'block': blocks, 'tr_in_block': tr_in_blocks,
        'time_to_lick': time_to_licks, 'time_to_abort': time_to_aborts,
    })


def _load_all_tf_resps(sess_dir, ops=ANALYSIS_OPTIONS, bm_ops=TUNING_CURVE_OPS):
    """
    load mean responses for ALL TF pulses from the FR matrix.
    averages over the response window (no smoothing - the window averaging is sufficient).
    returns {block: {mean_resp: (nEv, nN), tf_vals: (nEv,)}}, session
    """
    session = Session.load(str(sess_dir / 'session.pkl'))
    all_tf = _get_all_tf_pulses(session)

    # filters
    non_trans = all_tf['tr_in_block'] > ops['ignore_first_trials_in_block']
    early_tr = ((all_tf['tr_time'] > ops['rmv_time_around_bl']) &
                (all_tf['tr_time'] < bm_ops['trial_split_time']))
    t_to_event = np.fmin(all_tf['time_to_lick'], all_tf['time_to_abort'])
    valid = ((all_tf['tr_time'] > ops['rmv_time_around_bl']) &
             (t_to_event > ops['rmv_time_around_move']))

    # load FR matrix (downsampled)
    fr = load_fr_matrix(str(sess_dir / 'FR_matrix_ds.parquet'))
    fr_vals = fr.values
    t_ax_fr = fr.columns.values.astype(float)
    del fr
    gc.collect()

    resp_win = bm_ops['tf_resp_win']

    e_block = (all_tf['block'] == 'early') & non_trans
    l_block = (all_tf['block'] == 'late') & non_trans

    results = {}
    for block, block_mask in [('early', e_block), ('late', l_block)]:
        mask = block_mask & early_tr & valid
        events = all_tf[mask]
        event_times = events['time'].values

        i_starts = np.searchsorted(t_ax_fr, event_times + resp_win[0])
        i_ends = np.searchsorted(t_ax_fr, event_times + resp_win[1])

        n_events = len(events)
        n_neurons = fr_vals.shape[0]
        mean_resp = np.full((n_events, n_neurons), np.nan)
        for i in range(n_events):
            if i_ends[i] > i_starts[i] and i_ends[i] <= fr_vals.shape[1]:
                mean_resp[i] = np.mean(fr_vals[:, i_starts[i]:i_ends[i]], axis=1)

        results[block] = dict(mean_resp=mean_resp, tf_vals=events['tf'].values)

    return results, session


#%% TF binning

def bin_responses_by_tf(responses, tf_vals, edges):
    """
    bin per-trial responses using pre-computed bin edges.
    responses: (nEv, nN), tf_vals: (nEv,)
    edges: (n_bins + 1,) bin edges
    returns: binned (n_bins, nN), bin_sem (n_bins, nN), bin_centres (n_bins,)
    """
    n_bins = len(edges) - 1
    n_neurons = responses.shape[1]
    bin_idx = np.digitize(tf_vals, edges[1:-1])  # 0 to n_bins-1

    binned = np.full((n_bins, n_neurons), np.nan)
    bin_sem = np.full((n_bins, n_neurons), np.nan)
    bin_centres = np.full(n_bins, np.nan)

    for b in range(n_bins):
        mask = bin_idx == b
        count = mask.sum()
        if count > 0:
            binned[b] = np.nanmean(responses[mask], axis=0)
            bin_centres[b] = np.median(tf_vals[mask])
            if count > 1:
                bin_sem[b] = np.nanstd(responses[mask], axis=0, ddof=1) / np.sqrt(count)

    # smooth across bins with [.25, .5, .25] kernel, renormalised at edges
    for col in range(n_neurons):
        valid = ~np.isnan(binned[:, col])
        if valid.sum() >= 3:
            vals = binned[valid, col].copy()
            n_v = len(vals)
            smoothed = np.empty(n_v)
            smoothed[0] = (0.5 * vals[0] + 0.25 * vals[1]) / 0.75
            smoothed[-1] = (0.25 * vals[-2] + 0.5 * vals[-1]) / 0.75
            for i in range(1, n_v - 1):
                smoothed[i] = 0.25 * vals[i-1] + 0.5 * vals[i] + 0.25 * vals[i+1]
            binned[valid, col] = smoothed

    return binned, bin_sem, bin_centres


#%% analysis 1: single-unit TF tuning curves by block

def _vectorised_ols(x, Y):
    """
    fit y = gain * x + offset for all columns of Y simultaneously.
    x: (n,) predictor, Y: (n, p) responses
    returns gains (p,), offsets (p,)
    """
    x_mean = x.mean()
    Y_mean = Y.mean(axis=0)
    x_centered = x - x_mean
    ss_x = x_centered @ x_centered
    if ss_x == 0:
        return np.zeros(Y.shape[1]), Y_mean
    gains = (x_centered @ Y) / ss_x
    offsets = Y_mean - gains * x_mean
    return gains, offsets


def _vectorised_ols_batch(X, Y):
    """
    fit y = gain * x + offset for batches of (x, Y) pairs.
    X: (batch, n) predictors, Y: (batch, n, p) responses
    returns gains (batch, p), offsets (batch, p)
    """
    X_mean = X.mean(axis=1, keepdims=True)  # (batch, 1)
    Y_mean = Y.mean(axis=1)  # (batch, p)
    X_centered = X - X_mean
    ss_x = np.sum(X_centered ** 2, axis=1, keepdims=True)  # (batch, 1)
    xy = np.einsum('bn,bnp->bp', X_centered, Y)  # (batch, p)
    safe_ss = np.where(ss_x > 0, ss_x, 1.0)
    gains = np.where(ss_x > 0, xy / safe_ss, 0.0)
    offsets = Y_mean - gains * X_mean
    return gains, offsets


def extract_tuning_curves(sess_dir,
                          ops=ANALYSIS_OPTIONS,
                          bm_ops=TUNING_CURVE_OPS):
    """
    extract per-unit TF tuning curve parameters (gain, offset) per block
    for a single session. uses ALL TF pulses (not just outliers).
    """
    data, session = _load_all_tf_resps(sess_dir, ops, bm_ops)
    n_perm = bm_ops['n_permutations']
    n_bins = bm_ops['n_tf_bins']
    min_events = 10 * n_bins

    # drop events with any NaN (response window fell outside FR matrix)
    block_mean_resp = {}
    block_tf = {}
    for block in ['early', 'late']:
        resp = data[block]['mean_resp']
        tf = data[block]['tf_vals']
        valid = ~np.any(np.isnan(resp), axis=1)
        block_mean_resp[block] = resp[valid]
        block_tf[block] = tf[valid]

    # skip sessions with insufficient events
    for block in ['early', 'late']:
        if len(block_tf[block]) < min_events:
            print(f'    skipping: only {len(block_tf[block])} valid {block}-block events '
                  f'(need {min_events})')
            return None

    # subsample larger block to match smaller block
    n_early = len(block_tf['early'])
    n_late = len(block_tf['late'])
    if n_early != n_late:
        rng = np.random.default_rng(0)
        if n_early > n_late:
            idx = rng.choice(n_early, size=n_late, replace=False)
            block_mean_resp['early'] = block_mean_resp['early'][idx]
            block_tf['early'] = block_tf['early'][idx]
        else:
            idx = rng.choice(n_late, size=n_early, replace=False)
            block_mean_resp['late'] = block_mean_resp['late'][idx]
            block_tf['late'] = block_tf['late'][idx]

    n_neurons = block_mean_resp['early'].shape[1]

    # fit tuning curves for all neurons at once per block
    gains = np.zeros((n_neurons, 2))
    offsets = np.zeros((n_neurons, 2))
    for bi, block in enumerate(['early', 'late']):
        g, o = _vectorised_ols(block_tf[block], block_mean_resp[block])
        gains[:, bi] = g
        offsets[:, bi] = o

    # test 1a: per-block gain significance (shuffle TF within each block)
    gain_p_block = np.ones((n_neurons, 2))
    for bi, block in enumerate(['early', 'late']):
        tf_vals = block_tf[block]
        resp = block_mean_resp[block]
        real_g = gains[:, bi]

        tf_centered = tf_vals - tf_vals.mean()
        ss_x = tf_centered @ tf_centered

        if ss_x > 0:
            shuf_tf = np.array([np.random.permutation(tf_vals)
                                for _ in range(n_perm)])
            shuf_centered = shuf_tf - tf_vals.mean()
            null_g = (shuf_centered @ resp) / ss_x
            gain_p_block[:, bi] = np.mean(
                np.abs(null_g) >= np.abs(real_g), axis=0)

    # test 1b: pooled gain significance (shuffle TF across both blocks)
    all_resp = np.concatenate([block_mean_resp['early'],
                               block_mean_resp['late']], axis=0)
    all_tf = np.concatenate([block_tf['early'], block_tf['late']])
    n_total = len(all_tf)

    real_gains_pooled, _ = _vectorised_ols(all_tf, all_resp)

    tf_centered = all_tf - all_tf.mean()
    ss_x = tf_centered @ tf_centered

    shuf_tf = np.array([np.random.permutation(all_tf)
                        for _ in range(n_perm)])
    shuf_centered = shuf_tf - all_tf.mean()
    null_gains = (shuf_centered @ all_resp) / ss_x
    gain_p_tf = np.mean(np.abs(null_gains) >= np.abs(real_gains_pooled), axis=0)

    # test 2: block difference (shuffle block labels)
    real_gain_diff = gains[:, 0] - gains[:, 1]
    real_offset_diff = offsets[:, 0] - offsets[:, 1]
    n_early = len(block_mean_resp['early'])

    shuf_indices = np.array([np.random.permutation(n_total)
                             for _ in range(n_perm)])
    idx_1 = shuf_indices[:, :n_early]
    idx_2 = shuf_indices[:, n_early:]

    g1, o1 = _vectorised_ols_batch(all_tf[idx_1], all_resp[idx_1])
    g2, o2 = _vectorised_ols_batch(all_tf[idx_2], all_resp[idx_2])

    null_gain_diff = g1 - g2   # (n_perm, nN)
    null_offset_diff = o1 - o2

    gain_diff_p = np.mean(
        np.abs(null_gain_diff) >= np.abs(real_gain_diff), axis=0)
    offset_diff_p = np.mean(
        np.abs(null_offset_diff) >= np.abs(real_offset_diff), axis=0)

    # bin tuning curves per unit for plotting (quantile bins, equal trial count)
    # use pooled TF values from both blocks for consistent bin edges
    all_tf_pooled = np.concatenate([block_tf['early'], block_tf['late']])
    n_bins = bm_ops['n_tf_bins']
    edges = np.percentile(all_tf_pooled, np.linspace(0, 100, n_bins + 1))

    binned = {}
    binned_sem = {}
    bin_centres = None
    for block in ['early', 'late']:
        b, s, c = bin_responses_by_tf(block_mean_resp[block], block_tf[block], edges)
        binned[block] = b
        binned_sem[block] = s
        if bin_centres is None:
            bin_centres = c

    results = {
        'gain': gains,
        'offset': offsets,
        'gain_pooled': real_gains_pooled,  # (nN,) pooled gain across blocks
        'gain_p_block': gain_p_block,      # (nN, 2) per-block gain p-values
        'gain_p_tf': gain_p_tf,
        'gain_diff_p': gain_diff_p,
        'offset_diff_p': offset_diff_p,
        'binned': binned,                  # {block: (n_bins, nN)}
        'binned_sem': binned_sem,          # {block: (n_bins, nN)}
        'bin_centres': bin_centres,        # (n_bins,) median TF per bin
        'unit_info': session.unit_info,
        'animal': session.animal,
        'session': session.name,
    }

    return results


def extract_all_tuning_curves(npx_dir=PATHS['npx_dir_local'],
                              ops=ANALYSIS_OPTIONS,
                              bm_ops=TUNING_CURVE_OPS,
                              plot_during_extraction: bool = TUNING_CURVE_OPS[
                                  'plot_during_extraction'],
                              overwrite: bool = False):
    """extract tuning curves for all sessions, save per session"""
    animal_sessions = get_session_dirs_by_animal(npx_dir)

    for animal, sess_dirs in animal_sessions.items():
        for sess_dir in sess_dirs:
            print(f'{animal}/{sess_dir.name}')
            tc_path = sess_dir / 'tuning_curves.pkl'
            if tc_path.exists() and not overwrite:
                print(f'    tuning curves already extracted.')
                continue
            results = extract_tuning_curves(sess_dir, ops, bm_ops)
            if results is None:
                continue
            with open(tc_path, 'wb') as f:
                pickle.dump(results, f)

            # per-neuron figures immediately after each session
            if plot_during_extraction:
                import gc
                import matplotlib.pyplot as plt
                from tuning_curves.plotting import plot_session_su_tuning
                plots_dir = str(Path(PATHS['plots_dir']) / 'tuning_curves')
                plot_session_su_tuning(results, sess_dir, save_dir=plots_dir)
                plt.close('all')
                gc.collect()

        # summary figures after each animal
        if plot_during_extraction:
            import matplotlib.pyplot as plt
            from tuning_curves.plotting import (plot_tuning_curves,
                                                    plot_gain_offset_distributions)
            plots_dir = str(Path(PATHS['plots_dir']) / 'tuning_curves')
            plot_tuning_curves(npx_dir=npx_dir, save_dir=plots_dir)
            plot_gain_offset_distributions(npx_dir=npx_dir, save_dir=plots_dir)
            plt.close('all')
