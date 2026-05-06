"""per-block ridge poisson glm with time-windowed tf kernels"""
import numpy as np
import pickle
from pathlib import Path

from config import ANALYSIS_OPTIONS, GLM_OPTIONS
from data.session import Session
from neuron_prediction.data import (
    load_glm_inputs, get_trial_fold_indices, analysis_trials,
)
from neuron_prediction.glm_ridge.fit import fit_neuron
from neuron_prediction.results.peth import build_event_spec


BLOCKS = ('early', 'late')
TF_TIME_WINDOWS = [(0, 4), (4, 8), (8, 16)]


#%% predictor manipulation

def drop_predictor(X, col_map, name):
    """remove a named predictor from X and reindex col_map"""
    if name not in col_map:
        return X, col_map

    col_slice, _ = col_map[name]
    n_drop = col_slice.stop - col_slice.start

    keep = np.ones(X.shape[1], dtype=bool)
    keep[col_slice] = False
    X_new = X[:, keep]

    col_map_new = {}
    for k, (cs, lags) in col_map.items():
        if k == name:
            continue
        if cs.start >= col_slice.stop:
            new_slice = slice(cs.start - n_drop, cs.stop - n_drop)
        else:
            new_slice = cs
        col_map_new[k] = (new_slice, lags)

    return X_new, col_map_new


def time_shift(signal, kernel_win, bin_width):
    """expand (T,) signal into (T, n_lags) by shifting at each lag"""
    T = len(signal)
    lag_start = round(kernel_win[0] / bin_width)
    lag_end = round(kernel_win[1] / bin_width)
    lags = np.arange(lag_start, lag_end)

    shifted = np.zeros((T, len(lags)), dtype=np.float32)
    for i, lag in enumerate(lags):
        if lag >= 0:
            shifted[lag:, i] = signal[:T - lag]
        else:
            shifted[:T + lag, i] = signal[-lag:]

    return shifted, lags


#%% windowed tf predictor

def trial_time(session, t_ax):
    """seconds from baseline onset per bin, -1 outside trials"""
    tr_time = np.full(len(t_ax), -1, dtype=np.float32)
    for _, row in analysis_trials(session).iterrows():
        bl_on = row['Baseline_ON_rise']
        tr_end = np.nanmax([row['Baseline_ON_fall'],
                            row.get('Change_ON_fall', np.nan)])
        if np.isnan(bl_on) or np.isnan(tr_end):
            continue
        mask = (t_ax >= bl_on) & (t_ax < tr_end)
        tr_time[mask] = t_ax[mask] - bl_on
    return tr_time


def build_tf_signal(session, t_ax):
    """log2(TF) at each 50ms bin during baseline, 0 elsewhere"""
    T = len(t_ax)
    tf_signal = np.zeros(T, dtype=np.float32)

    for _, row in analysis_trials(session).iterrows():
        tf_raw = np.array(row['TF'])
        ft_raw = np.array(row['frame_time'])

        tf_seq = tf_raw[tf_raw.nonzero()]
        ch_fr = round(row['stimT'] * 60)
        tf_20hz = np.log2(tf_seq[:ch_fr:3])
        ft_20hz = ft_raw[~np.isnan(ft_raw)][:ch_fr:3]

        if len(tf_20hz) == 0 or len(ft_20hz) == 0:
            continue
        n = min(len(tf_20hz), len(ft_20hz))
        tf_20hz, ft_20hz = tf_20hz[:n], ft_20hz[:n]

        bl_end = row.get('Baseline_ON_fall', np.nan)
        if np.isnan(bl_end):
            bl_end = ft_20hz[-1]
        bl_mask = ft_20hz <= bl_end

        bin_mask = (t_ax >= ft_20hz[0]) & (t_ax <= bl_end)
        bin_idx = np.where(bin_mask)[0]
        if len(bin_idx) == 0:
            continue

        insert_idx = np.searchsorted(ft_20hz[bl_mask], t_ax[bin_idx], side='right') - 1
        insert_idx = np.clip(insert_idx, 0, bl_mask.sum() - 1)
        tf_signal[bin_idx] = tf_20hz[bl_mask][insert_idx]

    return tf_signal


def split_tf_predictor(X, col_map, session, t_ax):
    """replace single tf predictor with per-time-window versions"""
    X, col_map = drop_predictor(X, col_map, 'tf')

    tf_full = build_tf_signal(session, t_ax)
    tr_time = trial_time(session, t_ax)

    kern_tf = GLM_OPTIONS['kern_tf']
    bin_width = GLM_OPTIONS['bin_width']

    new_blocks = [X]
    col_offset = X.shape[1]

    for t_start, t_end in TF_TIME_WINDOWS:
        name = f'tf_{t_start}_{t_end}'
        tf_win = tf_full.copy()
        tf_win[(tr_time < t_start) | (tr_time >= t_end)] = 0
        shifted, lags = time_shift(tf_win, kern_tf, bin_width)
        col_map[name] = (slice(col_offset, col_offset + shifted.shape[1]), lags)
        col_offset += shifted.shape[1]
        new_blocks.append(shifted)

    return np.concatenate(new_blocks, axis=1), col_map


#%% fold assignment

def get_block_fold_indices(trials_df, t_ax, n_folds, block, ignore_first_n,
                           max_trial_dur):
    """fold ids for the first max_trial_dur s of each trial in this block"""
    block_trials = trials_df[trials_df['hazardblock'] == block]
    return get_trial_fold_indices(block_trials, t_ax, n_folds,
                                  ignore_first_n=ignore_first_n,
                                  max_trial_dur=max_trial_dur)


#%% main entry point

def fit_neuron_perblock_from_disk(sess_dir, neuron_idx, ops=GLM_OPTIONS):
    """fit ridge GLM for one neuron separately per block"""
    sess_dir = Path(sess_dir)
    counts, X, col_map, t_ax, valid_mask = load_glm_inputs(str(sess_dir))

    X, col_map = drop_predictor(X, col_map, 'block')

    sess = Session.load(str(sess_dir / 'session.pkl'))
    X, col_map = split_tf_predictor(X, col_map, sess, t_ax)

    ignore_n = ANALYSIS_OPTIONS['ignore_first_trials_in_block']
    max_dur = max(w[1] for w in TF_TIME_WINDOWS)

    # update lesion groups: drop block, replace tf with windowed names
    lesion_groups = dict(ops['lesion_groups'])
    lesion_groups.pop('block', None)
    tf_names = [f'tf_{s}_{e}' for s, e in TF_TIME_WINDOWS]
    lesion_groups['tf'] = tf_names
    ops = {**ops, 'lesion_groups': lesion_groups}

    results_dir = sess_dir / 'glm_perblock_results'
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / 'col_map.pkl', 'wb') as f:
        pickle.dump(col_map, f)

    event_spec = build_event_spec(
        sess,
        kinds=['tf', 'lick_prep', 'lick_exec'],
        t_ax=t_ax,
        bin_width=ops['bin_width'],
        tf_sd_threshold=ops['tf_sd_threshold'])

    for block in BLOCKS:
        fold_ids = get_block_fold_indices(
            sess.trials, t_ax, ops['n_folds'], block, ignore_n, max_dur)
        n_valid = (fold_ids >= 0).sum()
        n_folds_actual = len(set(fold_ids[fold_ids >= 0]))
        print(f'  {block}: {n_valid} valid bins, {n_folds_actual} folds')

        result = fit_neuron(counts[neuron_idx], X, col_map, fold_ids,
                            event_spec=event_spec, ops=ops)
        if result is None:
            print(f'  {block}: skipped (no valid data)')
            continue

        out_path = results_dir / f'neuron_{neuron_idx}_{block}.npz'
        np.savez(out_path, **result)
        print(f'  {block}: r={np.nanmean(result["full_r"]):.4f}, '
              f'lambda={float(result["best_lambda"]):.0e}, '
              f'saved to {out_path.name}')