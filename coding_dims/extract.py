"""coding dimension extraction: TF and premotor coding directions by block"""

import numpy as np
import pandas as pd
import pickle
import gc
from pathlib import Path
from multiprocessing import Pool

from config import PATHS, ANALYSIS_OPTIONS, CODING_DIM_OPS
from data.load_responses import load_psth
from data.session import Session
from utils.filing import get_session_dirs_by_animal, load_fr_matrix, file_suffix
from utils.selection import get_neuron_mask, get_window_bins
from utils.smoothing import causal_boxcar
from utils.time import window_label, time_mask
from utils.shuffle import circular_shift_labels
from utils.stats import roc_auc, cosine_similarity, l2_normalise


def _dprime_unpaired(a, b):
    """per-neuron d' from two unpaired sample arrays shaped (nEv, nN).
    neurons with sigma=0 get weight 0"""
    mu_diff = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)
    sigma = np.sqrt((np.nanvar(a, axis=0) + np.nanvar(b, axis=0)) / 2)
    d = np.zeros_like(mu_diff)
    valid = sigma > 0
    d[valid] = mu_diff[valid] / sigma[valid]
    return d


def _dprime_paired(a, b):
    """per-neuron paired d' from two arrays shaped (nEv, nN) with matched events.
    neurons with sigma=0 get weight 0"""
    diff = a - b
    mu = np.nanmean(diff, axis=0)
    sigma = np.nanstd(diff, axis=0)
    d = np.zeros_like(mu)
    valid = sigma > 0
    d[valid] = mu[valid] / sigma[valid]
    return d


#%% data loading

def _load_tf_resps_by_block(sess_dir, ops=ANALYSIS_OPTIONS, cd_ops=CODING_DIM_OPS):
    """
    load per-trial outlier TF responses, split by block and fast/slow.
    returns:
        {block: {'fast': (nEv, nN, nT), 'slow': (nEv, nN, nT)}}
        t_ax
    """
    psth_path = str(sess_dir / 'psths.h5')

    results = {}
    t_ax = None
    for block, pos_key, neg_key in [
        ('early', 'earlyBlock_early_pos', 'earlyBlock_early_neg'),
        ('late', 'lateBlock_early_pos', 'lateBlock_early_neg'),
    ]:
        pos_resp, t_ax = load_psth(psth_path, 'tf', pos_key)
        neg_resp, _ = load_psth(psth_path, 'tf', neg_key)
        results[block] = {'fast': pos_resp, 'slow': neg_resp}

    return results, t_ax


def _load_lick_resps_by_block(sess_dir, lick_type='fa'):
    """
    load per-trial lick-aligned responses, split by block.
    lick_type: 'fa' for false alarms only, 'hit' for hits only, 'all' for both
    returns {block: (nEv, nN, nT) or None}, t_ax
    """
    psth_path = str(sess_dir / 'psths.h5')
    results = {}
    t_ax = None

    if lick_type == 'all':
        suffixes = {'early': 'earlyBlock_early_*', 'late': 'lateBlock_early_*'}
    else:
        suffixes = {'early': f'earlyBlock_early_{lick_type}',
                    'late': f'lateBlock_early_{lick_type}'}

    for block, pattern in suffixes.items():
        try:
            resp, t_ax = load_psth(psth_path, 'lick', pattern)
            if resp.shape[0] == 0:
                results[block] = None
            else:
                results[block] = resp
        except (KeyError, ValueError):
            results[block] = None

    return results, t_ax


def _session_valid_for_tf(tf_data):
    """both blocks need sufficient fast and slow TF trials"""
    return all(tf_data[b]['fast'].shape[0] > 500 and tf_data[b]['slow'].shape[0] > 500
               for b in ['early', 'late'])


def _session_valid_for_motor(lick_data, min_trials=2):
    return all(lick_data[b] is not None and lick_data[b].shape[0] >= min_trials
               for b in ['early', 'late'])


#%% block coding dimensions — data loading

def _load_block_resps(sess_dir, ops=ANALYSIS_OPTIONS, cd_ops=CODING_DIM_OPS):
    """
    load per-trial mean FR vectors from the downsampled FR matrix, split by
    block and time window. applies causal smoothing, excludes bins near
    licks/aborts and bins after change onset within each trial

    returns {block: {window_label: (nTrials, nN) array or None}}
    """
    fr_path = sess_dir / 'FR_matrix_ds.parquet'
    if not fr_path.exists():
        return None

    session = Session.load(str(sess_dir / 'session.pkl'))
    trials = session.trials

    # load FR matrix: rows = neurons, columns = time bins (seconds)
    fr = load_fr_matrix(str(fr_path))
    t_ax = fr.columns.values.astype(float)

    # apply causal smoothing (FR_matrix_ds is at pop_bin_width)
    window_bins = get_window_bins(cd_ops, ops['pop_bin_width'])
    fr_vals = causal_boxcar(fr.values, window_bins, axis=1)  # (nN, nT)
    del fr
    gc.collect()

    windows = cd_ops['block_coding_windows']
    ignore_n = ops['ignore_first_trials_in_block']
    rmv_move = ops['rmv_time_around_move']

    # precompute a session-wide mask for bins near any lick or abort
    # lick times: FA trials with motion_onset or Baseline_ON_rise + rt_FA, hit trials with motion_onset
    lick_times = []
    abort_times = []
    for _, row in trials.iterrows():
        if row['IsFA']:
            lt = row['motion_onset'] if not pd.isna(row.get('motion_onset', np.nan)) \
                else (row['Baseline_ON_rise'] + row['rt_FA'])
            if not np.isnan(lt):
                lick_times.append(lt)
        elif row['IsHit']:
            if not np.isnan(row.get('motion_onset', np.nan)):
                lick_times.append(row['motion_onset'])
        if not np.isnan(row.get('rt_abort', np.nan)):
            abort_times.append(row['Baseline_ON_rise'] + row['rt_abort'])

    # build boolean mask: True = bin is OK (not near any lick/abort)
    move_mask = np.ones(len(t_ax), dtype=bool)
    for mt in lick_times + abort_times:
        move_mask &= np.abs(t_ax - mt) > rmv_move

    # collect full chronological block labels (all trials, for circular shift null)
    all_block_labels = trials['hazardblock'].values.copy()

    # iterate trials, extract per-trial mean FR in each window
    results = {b: {window_label(w): [] for w in windows} for b in ['early', 'late']}
    trial_list = []  # per-included-trial: {seq_idx, block, window_means}

    for tr_idx, row in trials.iterrows():
        block = row['hazardblock']
        if block not in ('early', 'late'):
            continue
        tr_in_block = row.get('tr_in_block', 0)
        if tr_in_block < ignore_n:
            continue

        bl_onset = row['Baseline_ON_rise']
        if np.isnan(bl_onset):
            continue

        # change onset for this trial only (hits and misses have changes)
        change_t = row.get('Change_ON_rise', np.nan)
        if pd.isna(change_t):
            change_t = np.inf

        trial_means = {}
        any_valid = False
        for win in windows:
            wl = window_label(win)
            t_start = bl_onset + win[0]
            t_end = bl_onset + win[1]

            # bin mask: within window, not near movement, before change onset
            bin_mask = (t_ax >= t_start) & (t_ax < t_end) & move_mask & (t_ax < change_t)

            if bin_mask.sum() == 0:
                continue

            trial_mean = np.nanmean(fr_vals[:, bin_mask], axis=1)  # (nN,)
            results[block][wl].append(trial_mean)
            trial_means[wl] = trial_mean
            any_valid = True

        if any_valid:
            # seq_idx: position in the full trials DataFrame for circular shift
            seq_idx = list(trials.index).index(tr_idx)
            trial_list.append({
                'seq_idx': seq_idx,
                'block': block,
                'window_means': trial_means,
            })

    # stack into arrays
    for block in ['early', 'late']:
        for wl in list(results[block].keys()):
            trials_list = results[block][wl]
            if len(trials_list) == 0:
                results[block][wl] = None
            else:
                results[block][wl] = np.stack(trials_list, axis=0)  # (nTrials, nN)

    del fr_vals
    gc.collect()
    return results, trial_list, all_block_labels


def _session_valid_for_block(block_data, min_trials=10):
    """keep only blocks with sufficient trials in at least one window"""
    if block_data is None:
        return False
    for wl in block_data['early']:
        early_ok = block_data['early'][wl] is not None and block_data['early'][wl].shape[0] >= min_trials
        late_ok = block_data['late'][wl] is not None and block_data['late'][wl].shape[0] >= min_trials
        if early_ok and late_ok:
            return True
    return False


def _compute_block_directions(sess_trial_lists, sess_fit_idx, fit_labels, wl):
    """
    compute block coding directions (raw + dprime) from fit trials.
    fit_labels[s] is a list of block labels for the fit trials in session s,
    in the same order as sess_fit_idx[s].
    returns {'raw': (w_normed, norm), 'dprime': (w_normed, norm)}
    either entry can be (None, 0.0) if not enough data
    """
    raw_early_means = []
    raw_late_means = []
    dp_parts = []
    for s in range(len(sess_trial_lists)):
        trial_list = sess_trial_lists[s]
        early_vecs = []
        late_vecs = []
        for j, i in enumerate(sess_fit_idx[s]):
            if wl not in trial_list[i]['window_means']:
                continue
            v = trial_list[i]['window_means'][wl]
            if fit_labels[s][j] == 'early':
                early_vecs.append(v)
            elif fit_labels[s][j] == 'late':
                late_vecs.append(v)
        if not early_vecs or not late_vecs:
            continue

        early_arr = np.stack(early_vecs)  # (n_early_trials, n_neurons)
        late_arr = np.stack(late_vecs)    # (n_late_trials, n_neurons)
        raw_early_means.append(early_arr.mean(axis=0))
        raw_late_means.append(late_arr.mean(axis=0))
        dp_parts.append(_dprime_unpaired(early_arr, late_arr))

    raw_out = (None, 0.0)
    dp_out = (None, 0.0)

    if raw_early_means and raw_late_means:
        w = np.concatenate(raw_early_means) - np.concatenate(raw_late_means)
        w, norm = l2_normalise(w)
        if norm > 0:
            raw_out = (w, norm)

    if dp_parts:
        w = np.concatenate(dp_parts)
        w, norm = l2_normalise(w)
        if norm > 0:
            dp_out = (w, norm)

    return {'raw': raw_out, 'dprime': dp_out}


def _project_test_auc(sess_trial_lists, sess_test_idx, w, neuron_offsets, wl):
    """
    project test trials onto direction w and compute AUC-ROC.
    test trials always use their real block labels
    """
    projections = []
    labels = []
    for s in range(len(sess_trial_lists)):
        trial_list = sess_trial_lists[s]
        w_slice = w[neuron_offsets[s]:neuron_offsets[s + 1]]
        for i in sess_test_idx[s]:
            if wl not in trial_list[i]['window_means']:
                continue
            projections.append(np.dot(trial_list[i]['window_means'][wl], w_slice))
            labels.append(trial_list[i]['block'] == 'early')

    if len(projections) < 4:
        return np.nan
    return roc_auc(np.array(labels), np.array(projections))


def _get_fit_labels(sess_trial_lists, sess_fit_idx, sess_all_labels=None):
    """
    get block labels for fit trials in each session.
    if sess_all_labels is None, use real labels from trial_list.
    if provided, look up shifted labels by seq_idx
    """
    fit_labels = []
    for s in range(len(sess_trial_lists)):
        trial_list = sess_trial_lists[s]
        if sess_all_labels is None:
            fit_labels.append([trial_list[i]['block'] for i in sess_fit_idx[s]])
        else:
            fit_labels.append([sess_all_labels[s][trial_list[i]['seq_idx']]
                               for i in sess_fit_idx[s]])
    return fit_labels


def _process_block_animal(animal, sess_dirs, ops, cd_ops, area, unit_filter, save_dir):
    """
    extract block coding dimensions (early vs late) with held-out AUC test.
    uses circular shift of block labels on the full trial sequence as null
    """
    windows = cd_ops['block_coding_windows']
    n_perm = cd_ops['n_permutations']
    min_n = cd_ops.get('min_neurons', 5)
    min_trials = 10
    test_frac = 0.2
    suffix = file_suffix(area, unit_filter)

    # per-session data
    sess_trial_lists = []    # [{seq_idx, block, window_means}, ...]
    sess_all_labels = []     # full chronological block labels per session
    included_sessions = []
    n_neurons_per_session = []
    unit_ids = []

    for sess_dir in sess_dirs:
        loaded = _load_block_resps(sess_dir, ops, cd_ops)
        if loaded is None:
            continue
        data, trial_list, all_labels = loaded

        if not _session_valid_for_block(data, min_trials):
            continue

        neuron_mask = get_neuron_mask(sess_dir, area, unit_filter)
        if neuron_mask.sum() < min_n:
            continue

        # apply neuron mask to aggregated data and re-check
        masked_data = {}
        for block in ['early', 'late']:
            masked_data[block] = {}
            for wl in [window_label(w) for w in windows]:
                if data[block][wl] is not None:
                    masked_data[block][wl] = data[block][wl][:, neuron_mask]
                else:
                    masked_data[block][wl] = None

        if not _session_valid_for_block(masked_data, min_trials):
            continue

        # apply neuron mask to per-trial data
        for trial in trial_list:
            trial['window_means'] = {
                wl: v[neuron_mask] for wl, v in trial['window_means'].items()
            }

        session = Session.load(str(sess_dir / 'session.pkl'))
        cluster_ids = session.unit_info['cluster_id'].values[neuron_mask]
        unit_ids.extend([(sess_dir.name, int(cid)) for cid in cluster_ids])

        sess_trial_lists.append(trial_list)
        sess_all_labels.append(all_labels)
        included_sessions.append(sess_dir.name)
        n_neurons_per_session.append(int(neuron_mask.sum()))

        del data
        gc.collect()

    if not included_sessions:
        print(f'  skipping {animal}: no valid sessions for block dims ({suffix})')
        return animal, None

    n_sessions = len(included_sessions)
    n_total = sum(n_neurons_per_session)
    print(f'  {animal} [{suffix}] block: {n_sessions} sessions, {n_total} neurons')

    neuron_offsets = np.cumsum([0] + n_neurons_per_session)

    # stratified fit/test split within each session (fixed across permutations)
    sess_fit_idx = []
    sess_test_idx = []
    rng_split = np.random.default_rng(42)
    for trial_list in sess_trial_lists:
        early_idx = [i for i, t in enumerate(trial_list) if t['block'] == 'early']
        late_idx = [i for i, t in enumerate(trial_list) if t['block'] == 'late']
        rng_split.shuffle(early_idx)
        rng_split.shuffle(late_idx)
        n_test_e = max(1, int(len(early_idx) * test_frac))
        n_test_l = max(1, int(len(late_idx) * test_frac))
        sess_test_idx.append(early_idx[:n_test_e] + late_idx[:n_test_l])
        sess_fit_idx.append(early_idx[n_test_e:] + late_idx[n_test_l:])

    # real direction and AUC per window (raw + dprime)
    real_fit_labels = _get_fit_labels(sess_trial_lists, sess_fit_idx)

    dimensions = {}
    direction_norm = {}
    real_aucs = {}
    dimensions_dprime = {}
    direction_norm_dprime = {}
    real_aucs_dprime = {}

    for win in windows:
        wl = window_label(win)
        dirs = _compute_block_directions(
            sess_trial_lists, sess_fit_idx, real_fit_labels, wl)
        w, norm = dirs['raw']
        if w is not None:
            dimensions[wl] = w
            direction_norm[wl] = norm
            real_aucs[wl] = _project_test_auc(
                sess_trial_lists, sess_test_idx, w, neuron_offsets, wl)
        w_dp, norm_dp = dirs['dprime']
        if w_dp is not None:
            dimensions_dprime[wl] = w_dp
            direction_norm_dprime[wl] = norm_dp
            real_aucs_dprime[wl] = _project_test_auc(
                sess_trial_lists, sess_test_idx, w_dp, neuron_offsets, wl)

    # circular-shift null (raw + dprime)
    null_aucs = {window_label(w): np.full(n_perm, np.nan) for w in windows}
    null_aucs_dprime = {window_label(w): np.full(n_perm, np.nan) for w in windows}
    rng = np.random.default_rng(0)

    for p in range(n_perm):
        shifted = [circular_shift_labels(al, rng) for al in sess_all_labels]
        shifted_fit_labels = _get_fit_labels(
            sess_trial_lists, sess_fit_idx, sess_all_labels=shifted)

        for win in windows:
            wl = window_label(win)
            dirs = _compute_block_directions(
                sess_trial_lists, sess_fit_idx, shifted_fit_labels, wl)
            if wl in dimensions and dirs['raw'][0] is not None:
                null_aucs[wl][p] = _project_test_auc(
                    sess_trial_lists, sess_test_idx, dirs['raw'][0],
                    neuron_offsets, wl)
            if wl in dimensions_dprime and dirs['dprime'][0] is not None:
                null_aucs_dprime[wl][p] = _project_test_auc(
                    sess_trial_lists, sess_test_idx, dirs['dprime'][0],
                    neuron_offsets, wl)

    p_values = {}
    p_values_dprime = {}
    for win in windows:
        wl = window_label(win)
        if wl in real_aucs:
            valid = null_aucs[wl][~np.isnan(null_aucs[wl])]
            p_values[wl] = np.mean(valid >= real_aucs[wl]) if len(valid) > 0 else np.nan
        if wl in real_aucs_dprime:
            valid = null_aucs_dprime[wl][~np.isnan(null_aucs_dprime[wl])]
            p_values_dprime[wl] = np.mean(valid >= real_aucs_dprime[wl]) if len(valid) > 0 else np.nan

    # save per-session intermediate data for pooled aggregation
    sess_intermediate = []
    for s in range(n_sessions):
        sess_intermediate.append({
            'trial_list': sess_trial_lists[s],
            'all_block_labels': sess_all_labels[s],
            'fit_idx': sess_fit_idx[s],
            'test_idx': sess_test_idx[s],
            'n_neurons': n_neurons_per_session[s],
        })

    result = {
        'dimensions': dimensions,
        'dimensions_dprime': dimensions_dprime,
        'direction_norm': direction_norm,
        'direction_norm_dprime': direction_norm_dprime,
        'real_aucs': real_aucs,
        'real_aucs_dprime': real_aucs_dprime,
        'null_aucs': null_aucs,
        'null_aucs_dprime': null_aucs_dprime,
        'p_values': p_values,
        'p_values_dprime': p_values_dprime,
        'included_sessions': included_sessions,
        'n_neurons_per_session': n_neurons_per_session,
        'unit_ids': unit_ids,
        'sess_intermediate': sess_intermediate,
    }

    with open(save_dir / f'block_dimensions_{animal}_cd_{suffix}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return animal, result


#%% TF coding dimensions

def _process_tf_animal(animal, sess_dirs, ops, cd_ops, area, unit_filter, save_dir):
    """process a single animal to extract tf coding dim"""
    # PSTHs are sliced from the high-res FR matrix at sp_bin_width
    window_bins = get_window_bins(cd_ops, ops['sp_bin_width'])
    windows = cd_ops['tf_coding_windows']
    n_perm = cd_ops['n_permutations']
    min_n = cd_ops.get('min_neurons', 5)
    suffix = file_suffix(area, unit_filter)

    # (n_events, n_neurons) per window, for permutation
    sess_tavg = {window_label(w): {'early': {'fast': [], 'slow': []},
                                    'late':  {'fast': [], 'slow': []}}
                 for w in windows}

    # (n_neurons, n_time) event-averaged responses, for projections
    sess_mean = {'early': {'fast': [], 'slow': []},
                 'late':  {'fast': [], 'slow': []}}

    # event counts per block per group, for permutation split
    sess_n = {'early': {'fast': [], 'slow': []},
              'late':  {'fast': [], 'slow': []}}

    included_sessions = []
    n_neurons_per_session = []
    unit_ids = []  # (session_name, cluster_id) per neuron in order

    for sess_dir in sess_dirs:
        data, t_ax = _load_tf_resps_by_block(sess_dir, ops, cd_ops) # data: (events, nN, nT)
        if not _session_valid_for_tf(data):
            continue

        neuron_mask = get_neuron_mask(sess_dir, area, unit_filter)
        if neuron_mask.sum() < min_n:
            del data
            gc.collect()
            continue

        # record unit identities
        session = Session.load(str(sess_dir / 'session.pkl'))
        cluster_ids = session.unit_info['cluster_id'].values[neuron_mask]
        unit_ids.extend([(sess_dir.name, int(cid)) for cid in cluster_ids])

        for block in ['early', 'late']:
            fast = causal_boxcar(data[block]['fast'][:, neuron_mask, :],
                                 window_bins, axis=2)
            slow = causal_boxcar(data[block]['slow'][:, neuron_mask, :],
                                 window_bins, axis=2)

            sess_mean[block]['fast'].append(np.nanmean(fast, axis=0))
            sess_mean[block]['slow'].append(np.nanmean(slow, axis=0))
            sess_n[block]['fast'].append(fast.shape[0])
            sess_n[block]['slow'].append(slow.shape[0])

            # time-avg responses across specified windows
            for win in windows:
                wl = window_label(win)
                t_mask = time_mask(t_ax, win)
                sess_tavg[wl][block]['fast'].append(
                    np.nanmean(fast[:, :, t_mask], axis=2))
                sess_tavg[wl][block]['slow'].append(
                    np.nanmean(slow[:, :, t_mask], axis=2))

        included_sessions.append(session.name)
        n_neurons_per_session.append(int(neuron_mask.sum()))
        del data
        gc.collect()

    if not included_sessions:
        print(f'  skipping {animal}: no valid sessions for {suffix}')
        return animal, None

    n_sessions = len(included_sessions)
    n_total = sum(n_neurons_per_session)
    print(f'  {animal} [{suffix}]: {n_sessions} sessions, {n_total} neurons')

    # compute coding directions per block and window (raw + dprime)
    dimensions = {'early': {}, 'late': {}}
    dimensions_dprime = {'early': {}, 'late': {}}
    for block in ['early', 'late']:
        for win in windows:
            wl = window_label(win)
            fast_sess = sess_tavg[wl][block]['fast']
            slow_sess = sess_tavg[wl][block]['slow']

            # 'raw' cd: session-averaged mean(fast) - mean(slow)
            fast_means = [np.nanmean(s, axis=0) for s in fast_sess]
            slow_means = [np.nanmean(s, axis=0) for s in slow_sess]
            w_raw = np.concatenate(fast_means) - np.concatenate(slow_means)
            dimensions[block][wl], _ = l2_normalise(w_raw)

            # dprime variance: per-session unpaired d', concatenated (normed by sd)
            d_parts = [_dprime_unpaired(f, s) for f, s in zip(fast_sess, slow_sess)]
            w_dp = np.concatenate(d_parts)
            dimensions_dprime[block][wl], _ = l2_normalise(w_dp)

    # cross-block projections with separate fast and slow
    pop_mean = {}
    for block in ['early', 'late']:
        pop_mean[block] = {
            'fast': np.concatenate(sess_mean[block]['fast'], axis=0),
            'slow': np.concatenate(sess_mean[block]['slow'], axis=0),
        }
    del sess_mean

    cross_projections = {}
    cross_projections_dprime = {}
    for proj_block in ['early', 'late']:
        cross_projections[proj_block] = {}
        cross_projections_dprime[proj_block] = {}
        for dim_block in ['early', 'late']:
            cross_projections[proj_block][dim_block] = {}
            cross_projections_dprime[proj_block][dim_block] = {}
            for wl in dimensions[dim_block]:
                w = dimensions[dim_block][wl]
                w_dp = dimensions_dprime[dim_block][wl]
                cross_projections[proj_block][dim_block][wl] = {
                    'fast': w @ pop_mean[proj_block]['fast'],
                    'slow': w @ pop_mean[proj_block]['slow'],
                }
                cross_projections_dprime[proj_block][dim_block][wl] = {
                    'fast': w_dp @ pop_mean[proj_block]['fast'],
                    'slow': w_dp @ pop_mean[proj_block]['slow'],
                }
    del pop_mean

    # between-block cosine similarity + null (raw + dprime)
    between_block = {}
    between_block_dprime = {}
    for win in windows:
        wl = window_label(win)
        if wl not in dimensions['early'] or wl not in dimensions['late']:
            continue
        real_cos = cosine_similarity(dimensions['early'][wl],
                                     dimensions['late'][wl])
        real_cos_dp = cosine_similarity(dimensions_dprime['early'][wl],
                                        dimensions_dprime['late'][wl])
        null_cos = np.full(n_perm, np.nan)
        null_cos_dp = np.full(n_perm, np.nan)

        sess_pooled_fast = []
        sess_pooled_slow = []
        n_early_fast = []
        n_early_slow = []
        for s in range(n_sessions):
            sess_pooled_fast.append(np.concatenate(
                [sess_tavg[wl]['early']['fast'][s],
                 sess_tavg[wl]['late']['fast'][s]], axis=0))
            sess_pooled_slow.append(np.concatenate(
                [sess_tavg[wl]['early']['slow'][s],
                 sess_tavg[wl]['late']['slow'][s]], axis=0))
            n_early_fast.append(sess_n['early']['fast'][s])
            n_early_slow.append(sess_n['early']['slow'][s])

        rng = np.random.default_rng(0)
        for p in range(n_perm):
            raw_a, raw_b = [], []
            dp_a, dp_b = [], []
            for pf, ps, nef, nes in zip(
                    sess_pooled_fast, sess_pooled_slow,
                    n_early_fast, n_early_slow):
                idx_f = rng.permutation(pf.shape[0])
                f_shuf = pf[idx_f]
                fa_half, fb_half = f_shuf[:nef], f_shuf[nef:]
                idx_s = rng.permutation(ps.shape[0])
                s_shuf = ps[idx_s]
                sa_half, sb_half = s_shuf[:nes], s_shuf[nes:]

                raw_a.append(np.nanmean(fa_half, axis=0) - np.nanmean(sa_half, axis=0))
                raw_b.append(np.nanmean(fb_half, axis=0) - np.nanmean(sb_half, axis=0))
                dp_a.append(_dprime_unpaired(fa_half, sa_half))
                dp_b.append(_dprime_unpaired(fb_half, sb_half))

            null_cos[p] = cosine_similarity(np.concatenate(raw_a),
                                            np.concatenate(raw_b))
            null_cos_dp[p] = cosine_similarity(np.concatenate(dp_a),
                                               np.concatenate(dp_b))

        between_block[wl] = {'real': real_cos, 'null': null_cos}
        between_block_dprime[wl] = {'real': real_cos_dp, 'null': null_cos_dp}

    result = {
        'tf_t_ax': t_ax,
        'dimensions': dimensions,
        'dimensions_dprime': dimensions_dprime,
        'between_block_cosine': between_block,
        'between_block_cosine_dprime': between_block_dprime,
        'cross_projections': cross_projections,
        'cross_projections_dprime': cross_projections_dprime,
        'included_sessions': included_sessions,
        'n_neurons_per_session': n_neurons_per_session,
        'unit_ids': unit_ids,
        'sess_tavg': sess_tavg,
        'sess_n': sess_n,
    }

    with open(save_dir / f'tf_dimensions_{animal}_cd_{suffix}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return animal, result


def extract_tf_dimensions(npx_dir=PATHS['npx_dir_local'],
                          ops=ANALYSIS_OPTIONS,
                          cd_ops=CODING_DIM_OPS,
                          area: str | None = None,
                          unit_filter: list[str] | None = None,
                          method: str = 'cd',
                          n_jobs: int | None = None):
    """
    extract TF coding directions (fast vs slow) at defined time windows,
    per block, per animal (pseudo-population).
    area: brain region (AREA_GROUPS key), or None for all neurons
    unit_filter: list of GLM classification names, OR logic (e.g. ['tf', 'lick_prep'])
    n_jobs: number of parallel workers (None = all cores)
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)
    save_dir = Path(npx_dir) / 'coding_dims'
    save_dir.mkdir(exist_ok=True)
    suffix = file_suffix(area, unit_filter)

    args = [(animal, sess_dirs, ops, cd_ops, area, unit_filter, save_dir)
            for animal, sess_dirs in animal_sessions.items()]

    with Pool(n_jobs) as pool:
        results = pool.starmap(_process_tf_animal, args)

    all_results = {animal: res for animal, res in results if res is not None}

    out_path = save_dir / f'tf_dimensions_{method}_{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved TF dimensions ({method}) to {out_path}')

    return all_results


#%% premotor coding dimensions

def _process_motor_animal(animal, sess_dirs, ops, cd_ops, lick_type,
                          area, unit_filter, save_dir):
    """process a single animal for motor coding dimensions"""
    # PSTHs are sliced from the high-res FR matrix at sp_bin_width
    window_bins = get_window_bins(cd_ops, ops['sp_bin_width'])
    windows = cd_ops['motor_prelick_windows']
    bl_win = cd_ops['motor_baseline_window']
    n_perm = cd_ops['n_permutations']
    min_n = cd_ops.get('min_neurons', 5)
    suffix = file_suffix(area, unit_filter)

    # (n_events, n_neurons) per prelick window + baseline, for permutation
    sess_tavg_win = {window_label(w): {'early': [], 'late': []} for w in windows}
    sess_tavg_bl = {'early': [], 'late': []}
    # (n_neurons, n_time) event-averaged, for cross-projections
    sess_mean = {'early': [], 'late': []}
    # event counts per block, for permutation split
    sess_n = {'early': [], 'late': []}
    included_sessions = []
    n_neurons_per_session = []
    unit_ids = []

    for sess_dir in sess_dirs:
        data, lick_t_ax = _load_lick_resps_by_block(sess_dir, lick_type)
        if not _session_valid_for_motor(data):
            continue

        neuron_mask = get_neuron_mask(sess_dir, area, unit_filter)
        if neuron_mask.sum() < min_n:
            del data
            gc.collect()
            continue

        session = Session.load(str(sess_dir / 'session.pkl'))
        cluster_ids = session.unit_info['cluster_id'].values[neuron_mask]
        unit_ids.extend([(sess_dir.name, int(cid)) for cid in cluster_ids])

        bl_mask = time_mask(lick_t_ax, bl_win)

        for block in ['early', 'late']:
            resp = causal_boxcar(data[block][:, neuron_mask, :],
                                 window_bins, axis=-1)

            sess_mean[block].append(np.nanmean(resp, axis=0))
            sess_n[block].append(resp.shape[0])
            sess_tavg_bl[block].append(np.nanmean(resp[:, :, bl_mask], axis=2))

            for win in windows:
                wl = window_label(win)
                t_mask = time_mask(lick_t_ax, win)
                sess_tavg_win[wl][block].append(
                    np.nanmean(resp[:, :, t_mask], axis=2))

        included_sessions.append(sess_dir.name)
        n_neurons_per_session.append(int(neuron_mask.sum()))
        del data
        gc.collect()

    if not included_sessions:
        print(f'  skipping {animal}: insufficient lick data for {suffix}')
        return animal, None

    n_sessions = len(included_sessions)
    n_total = sum(n_neurons_per_session)
    print(f'  {animal} [{suffix}]: {n_sessions} sessions, {n_total} neurons')

    # compute coding directions per block and window (raw + paired dprime)
    dimensions = {'early': {}, 'late': {}}
    dimensions_dprime = {'early': {}, 'late': {}}
    for block in ['early', 'late']:
        for win in windows:
            wl = window_label(win)
            if not sess_tavg_win[wl][block]:
                continue
            win_sess = sess_tavg_win[wl][block]
            bl_sess = sess_tavg_bl[block]

            # raw: mean(prelick) - mean(baseline) per neuron
            win_means = [np.nanmean(s, axis=0) for s in win_sess]
            bl_means = [np.nanmean(s, axis=0) for s in bl_sess]
            w_raw = np.concatenate(win_means) - np.concatenate(bl_means)
            dimensions[block][wl], _ = l2_normalise(w_raw)

            # dprime: per-session paired d' (same lick events in both windows)
            d_parts = [_dprime_paired(w, b) for w, b in zip(win_sess, bl_sess)]
            w_dp = np.concatenate(d_parts)
            dimensions_dprime[block][wl], _ = l2_normalise(w_dp)

    # cross-block projections
    pop_mean = {}
    for block in ['early', 'late']:
        pop_mean[block] = np.concatenate(sess_mean[block], axis=0)
    del sess_mean

    cross_projections = {}
    cross_projections_dprime = {}
    for proj_block in ['early', 'late']:
        cross_projections[proj_block] = {}
        cross_projections_dprime[proj_block] = {}
        for dim_block in ['early', 'late']:
            cross_projections[proj_block][dim_block] = {}
            cross_projections_dprime[proj_block][dim_block] = {}
            for wl in dimensions[dim_block]:
                w = dimensions[dim_block][wl]
                w_dp = dimensions_dprime[dim_block][wl]
                cross_projections[proj_block][dim_block][wl] = w @ pop_mean[proj_block]
                cross_projections_dprime[proj_block][dim_block][wl] = w_dp @ pop_mean[proj_block]
    del pop_mean

    # between-block cosine similarity + null (raw + dprime)
    between_block = {}
    between_block_dprime = {}

    for win in windows:
        wl = window_label(win)
        if wl not in dimensions['early'] or wl not in dimensions['late']:
            continue
        real_cos = cosine_similarity(dimensions['early'][wl],
                                     dimensions['late'][wl])
        real_cos_dp = cosine_similarity(dimensions_dprime['early'][wl],
                                        dimensions_dprime['late'][wl])
        null_cos = np.full(n_perm, np.nan)
        null_cos_dp = np.full(n_perm, np.nan)

        sess_pooled_win = []
        sess_pooled_bl = []
        n_early = []
        for s in range(n_sessions):
            sess_pooled_win.append(np.concatenate(
                [sess_tavg_win[wl]['early'][s],
                 sess_tavg_win[wl]['late'][s]], axis=0))
            sess_pooled_bl.append(np.concatenate(
                [sess_tavg_bl['early'][s],
                 sess_tavg_bl['late'][s]], axis=0))
            n_early.append(sess_n['early'][s])

        rng = np.random.default_rng(0)
        for p in range(n_perm):
            raw_a, raw_b = [], []
            dp_a, dp_b = [], []
            for pw, pb, n_e in zip(sess_pooled_win, sess_pooled_bl, n_early):
                idx = rng.permutation(pw.shape[0])
                w_shuf = pw[idx]
                b_shuf = pb[idx]
                wa_half, wb_half = w_shuf[:n_e], w_shuf[n_e:]
                ba_half, bb_half = b_shuf[:n_e], b_shuf[n_e:]

                raw_a.append(np.nanmean(wa_half, axis=0) - np.nanmean(ba_half, axis=0))
                raw_b.append(np.nanmean(wb_half, axis=0) - np.nanmean(bb_half, axis=0))
                dp_a.append(_dprime_paired(wa_half, ba_half))
                dp_b.append(_dprime_paired(wb_half, bb_half))

            null_cos[p] = cosine_similarity(np.concatenate(raw_a),
                                            np.concatenate(raw_b))
            null_cos_dp[p] = cosine_similarity(np.concatenate(dp_a),
                                               np.concatenate(dp_b))

        between_block[wl] = {'real': real_cos, 'null': null_cos}
        between_block_dprime[wl] = {'real': real_cos_dp, 'null': null_cos_dp}

    result = {
        'lick_t_ax': lick_t_ax,
        'dimensions': dimensions,
        'dimensions_dprime': dimensions_dprime,
        'between_block_cosine': between_block,
        'between_block_cosine_dprime': between_block_dprime,
        'cross_projections': cross_projections,
        'cross_projections_dprime': cross_projections_dprime,
        'included_sessions': included_sessions,
        'n_neurons_per_session': n_neurons_per_session,
        'unit_ids': unit_ids,
        'sess_tavg_win': sess_tavg_win,
        'sess_tavg_bl': sess_tavg_bl,
        'sess_n': sess_n,
    }

    with open(save_dir / f'motor_dimensions_{animal}_cd_{suffix}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return animal, result


def extract_motor_dimensions(npx_dir=PATHS['npx_dir_local'],
                             ops=ANALYSIS_OPTIONS,
                             cd_ops=CODING_DIM_OPS,
                             lick_type='fa',
                             area: str | None = None,
                             unit_filter: list[str] | None = None,
                             method: str = 'cd',
                             n_jobs=None):
    """
    extract premotor coding directions (pre-lick window vs baseline) at
    defined time windows, per block, per animal (pseudo-population).
    lick_type: 'fa' for false alarms only (motor-only), 'hit', or 'all'
    area: brain region, AREA_GROUPS key, or None/'all' for all neurons
    unit_filter: list of GLM classification names, OR logic (e.g. ['tf', 'lick_prep'])
    n_jobs: number of parallel workers (None = all cores)
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)
    save_dir = Path(npx_dir) / 'coding_dims'
    save_dir.mkdir(exist_ok=True)
    suffix = file_suffix(area, unit_filter)

    args = [(animal, sess_dirs, ops, cd_ops, lick_type, area, unit_filter, save_dir)
            for animal, sess_dirs in animal_sessions.items()]

    with Pool(n_jobs) as pool:
        results = pool.starmap(_process_motor_animal, args)

    all_results = {animal: res for animal, res in results if res is not None}

    out_path = save_dir / f'motor_dimensions_{method}_{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved motor dimensions ({method}) to {out_path}')

    return all_results


#%% block coding dimensions — extraction

def extract_block_dimensions(npx_dir=PATHS['npx_dir_local'],
                             ops=ANALYSIS_OPTIONS,
                             cd_ops=CODING_DIM_OPS,
                             area: str | None = None,
                             unit_filter: list[str] | None = None,
                             method: str = 'cd',
                             n_jobs: int | None = None):
    """
    extract block coding directions (early vs late block) at defined time
    windows, per animal (pseudo-population).
    area: brain region, AREA_GROUPS key, or None/'all' for all neurons
    unit_filter: list of GLM classification names, OR logic (e.g. ['tf', 'lick_prep'])
    n_jobs: number of parallel workers (None = all cores)
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)
    save_dir = Path(npx_dir) / 'coding_dims'
    save_dir.mkdir(exist_ok=True)
    suffix = file_suffix(area, unit_filter)

    # per-animal
    args = [(animal, sess_dirs, ops, cd_ops, area, unit_filter, save_dir)
            for animal, sess_dirs in animal_sessions.items()]

    with Pool(n_jobs) as pool:
        results = pool.starmap(_process_block_animal, args)

    all_results = {animal: res for animal, res in results if res is not None}

    out_path = save_dir / f'block_dimensions_{method}_{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved block dimensions ({method}) to {out_path}')

    return all_results
