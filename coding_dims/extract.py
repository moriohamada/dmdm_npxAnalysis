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


def _mean_diff(a, b):
    # a, b: (nEv, nN) -> (nN,)
    return np.nanmean(a, axis=0) - np.nanmean(b, axis=0)


def _dprime_unpaired(a, b):
    # a, b: (nEv, nN) -> (nN,)
    mu_diff = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)
    sigma = np.sqrt((np.nanvar(a, axis=0) + np.nanvar(b, axis=0)) / 2)
    d = np.zeros_like(mu_diff)
    valid = sigma > 0
    d[valid] = mu_diff[valid] / sigma[valid]
    return d


def _dprime_paired(a, b):
    # a, b: (nEv, nN), matched events -> (nN,)
    diff = a - b
    mu = np.nanmean(diff, axis=0)
    sigma = np.nanstd(diff, axis=0)
    d = np.zeros_like(mu)
    valid = sigma > 0
    d[valid] = mu[valid] / sigma[valid]
    return d


def _lda_unpaired(a, b):
    # a, b: (nEv, nN) -> (nN,); Ledoit-Wolf shrinkage LDA
    from sklearn.covariance import LedoitWolf
    mu_diff = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)
    X = np.concatenate([a - np.nanmean(a, axis=0),
                        b - np.nanmean(b, axis=0)], axis=0)
    X = X[~np.any(np.isnan(X), axis=1)]
    if X.shape[0] < 2:
        return np.zeros_like(mu_diff)
    cov = LedoitWolf().fit(X).covariance_
    try:
        return np.linalg.solve(cov, mu_diff)
    except np.linalg.LinAlgError:
        return np.zeros_like(mu_diff)


def _lda_paired(a, b):
    # a, b: (nEv, nN), matched events -> (nN,); Hotelling's T² direction
    from sklearn.covariance import LedoitWolf
    diff = a - b
    mu = np.nanmean(diff, axis=0)
    diff = diff[~np.any(np.isnan(diff), axis=1)]
    if diff.shape[0] < 2:
        return np.zeros_like(mu)
    cov = LedoitWolf().fit(diff).covariance_
    try:
        return np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        return np.zeros_like(mu)


TF_DIM_FNS    = {'cd': _mean_diff, 'dprime_cd': _dprime_unpaired, 'lda': _lda_unpaired}
MOTOR_DIM_FNS = {'cd': _mean_diff, 'dprime_cd': _dprime_paired,   'lda': _lda_paired}
BLOCK_DIM_FNS = {'cd': _mean_diff, 'dprime_cd': _dprime_unpaired, 'lda': _lda_unpaired}


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
    # returns {dim_name: (w_normed, norm)} per dim_name in BLOCK_DIM_FNS.
    # entry can be (None, 0.0) if not enough data
    sess_early = []  # list of (n_early_trials, nN) per session
    sess_late  = []  # list of (n_late_trials, nN) per session
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
        if early_vecs and late_vecs:
            sess_early.append(np.stack(early_vecs))
            sess_late.append(np.stack(late_vecs))

    out = {v: (None, 0.0) for v in BLOCK_DIM_FNS}
    if not sess_early:
        return out

    for dim_name, fn in BLOCK_DIM_FNS.items():
        w = np.concatenate([fn(e, l) for e, l in zip(sess_early, sess_late)])
        w, norm = l2_normalise(w)
        if norm > 0:
            out[dim_name] = (w, norm)
    return out


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

    # real direction and AUC per window, per dim_name
    real_fit_labels = _get_fit_labels(sess_trial_lists, sess_fit_idx)

    dimensions     = {v: {} for v in BLOCK_DIM_FNS}
    direction_norm = {v: {} for v in BLOCK_DIM_FNS}
    real_aucs      = {v: {} for v in BLOCK_DIM_FNS}
    null_aucs = {v: {window_label(w): np.full(n_perm, np.nan) for w in windows}
                 for v in BLOCK_DIM_FNS}
    p_values = {v: {} for v in BLOCK_DIM_FNS}

    for win in windows:
        wl = window_label(win)
        dirs = _compute_block_directions(
            sess_trial_lists, sess_fit_idx, real_fit_labels, wl)
        for dim_name, (w, norm) in dirs.items():
            if w is None:
                continue
            dimensions[dim_name][wl] = w
            direction_norm[dim_name][wl] = norm
            real_aucs[dim_name][wl] = _project_test_auc(
                sess_trial_lists, sess_test_idx, w, neuron_offsets, wl)

    # circular-shift null
    rng = np.random.default_rng(0)
    for p in range(n_perm):
        shifted = [circular_shift_labels(al, rng) for al in sess_all_labels]
        shifted_fit_labels = _get_fit_labels(
            sess_trial_lists, sess_fit_idx, sess_all_labels=shifted)

        for win in windows:
            wl = window_label(win)
            dirs = _compute_block_directions(
                sess_trial_lists, sess_fit_idx, shifted_fit_labels, wl)
            for dim_name, (w_null, _) in dirs.items():
                if w_null is None or wl not in dimensions[dim_name]:
                    continue
                null_aucs[dim_name][wl][p] = _project_test_auc(
                    sess_trial_lists, sess_test_idx, w_null, neuron_offsets, wl)

    for dim_name in BLOCK_DIM_FNS:
        for wl, real_auc in real_aucs[dim_name].items():
            null = null_aucs[dim_name][wl]
            valid = null[~np.isnan(null)]
            p_values[dim_name][wl] = np.mean(valid >= real_auc) if len(valid) > 0 else np.nan

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
        'direction_norm': direction_norm,
        'real_aucs': real_aucs,
        'null_aucs': null_aucs,
        'p_values': p_values,
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

    # compute coding directions per dim_name, block, window
    dimensions = {v: {'early': {}, 'late': {}} for v in TF_DIM_FNS}
    for block in ['early', 'late']:
        for win in windows:
            wl = window_label(win)
            fast_sess = sess_tavg[wl][block]['fast']
            slow_sess = sess_tavg[wl][block]['slow']
            for dim_name, fn in TF_DIM_FNS.items():
                w = np.concatenate([fn(f, s) for f, s in zip(fast_sess, slow_sess)])
                dimensions[dim_name][block][wl], _ = l2_normalise(w)

    # cross-block projections with separate fast and slow, per dim_name
    pop_mean = {}
    for block in ['early', 'late']:
        pop_mean[block] = {
            'fast': np.concatenate(sess_mean[block]['fast'], axis=0),
            'slow': np.concatenate(sess_mean[block]['slow'], axis=0),
        }
    del sess_mean

    cross_projections = {v: {} for v in TF_DIM_FNS}
    for dim_name in TF_DIM_FNS:
        for proj_block in ['early', 'late']:
            cross_projections[dim_name][proj_block] = {}
            for dim_block in ['early', 'late']:
                cross_projections[dim_name][proj_block][dim_block] = {}
                for wl, w in dimensions[dim_name][dim_block].items():
                    cross_projections[dim_name][proj_block][dim_block][wl] = {
                        'fast': w @ pop_mean[proj_block]['fast'],
                        'slow': w @ pop_mean[proj_block]['slow'],
                    }
    del pop_mean

    # between-block cosine similarity + null
    between_block = {v: {} for v in TF_DIM_FNS}
    for win in windows:
        wl = window_label(win)
        if not all(wl in dimensions[v]['early'] and wl in dimensions[v]['late']
                   for v in TF_DIM_FNS):
            continue

        real_cos = {v: cosine_similarity(dimensions[v]['early'][wl],
                                          dimensions[v]['late'][wl])
                    for v in TF_DIM_FNS}
        null_cos = {v: np.full(n_perm, np.nan) for v in TF_DIM_FNS}

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
            fast_a, fast_b, slow_a, slow_b = [], [], [], []
            for pf, ps, nef, nes in zip(
                    sess_pooled_fast, sess_pooled_slow,
                    n_early_fast, n_early_slow):
                idx_f = rng.permutation(pf.shape[0])
                f_shuf = pf[idx_f]
                fast_a.append(f_shuf[:nef])
                fast_b.append(f_shuf[nef:])
                idx_s = rng.permutation(ps.shape[0])
                s_shuf = ps[idx_s]
                slow_a.append(s_shuf[:nes])
                slow_b.append(s_shuf[nes:])

            for dim_name, fn in TF_DIM_FNS.items():
                w_a = np.concatenate([fn(f, s) for f, s in zip(fast_a, slow_a)])
                w_b = np.concatenate([fn(f, s) for f, s in zip(fast_b, slow_b)])
                null_cos[dim_name][p] = cosine_similarity(w_a, w_b)

        for dim_name in TF_DIM_FNS:
            between_block[dim_name][wl] = {'real': real_cos[dim_name],
                                           'null': null_cos[dim_name]}

    result = {
        'tf_t_ax': t_ax,
        'dimensions': dimensions,
        'between_block_cosine': between_block,
        'cross_projections': cross_projections,
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
    # compute coding directions per dim_name, block, window
    dimensions = {v: {'early': {}, 'late': {}} for v in MOTOR_DIM_FNS}
    for block in ['early', 'late']:
        for win in windows:
            wl = window_label(win)
            if not sess_tavg_win[wl][block]:
                continue
            win_sess = sess_tavg_win[wl][block]
            bl_sess = sess_tavg_bl[block]
            for dim_name, fn in MOTOR_DIM_FNS.items():
                w = np.concatenate([fn(w_ev, b_ev)
                                    for w_ev, b_ev in zip(win_sess, bl_sess)])
                dimensions[dim_name][block][wl], _ = l2_normalise(w)

    # cross-block projections, per dim_name
    pop_mean = {}
    for block in ['early', 'late']:
        pop_mean[block] = np.concatenate(sess_mean[block], axis=0)
    del sess_mean

    cross_projections = {v: {} for v in MOTOR_DIM_FNS}
    for dim_name in MOTOR_DIM_FNS:
        for proj_block in ['early', 'late']:
            cross_projections[dim_name][proj_block] = {}
            for dim_block in ['early', 'late']:
                cross_projections[dim_name][proj_block][dim_block] = {}
                for wl, w in dimensions[dim_name][dim_block].items():
                    cross_projections[dim_name][proj_block][dim_block][wl] = (
                        w @ pop_mean[proj_block])
    del pop_mean

    # between-block cosine similarity + null
    between_block = {v: {} for v in MOTOR_DIM_FNS}
    for win in windows:
        wl = window_label(win)
        if not all(wl in dimensions[v]['early'] and wl in dimensions[v]['late']
                   for v in MOTOR_DIM_FNS):
            continue

        real_cos = {v: cosine_similarity(dimensions[v]['early'][wl],
                                          dimensions[v]['late'][wl])
                    for v in MOTOR_DIM_FNS}
        null_cos = {v: np.full(n_perm, np.nan) for v in MOTOR_DIM_FNS}

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
            win_a, win_b, bl_a, bl_b = [], [], [], []
            for pw, pb, n_e in zip(sess_pooled_win, sess_pooled_bl, n_early):
                idx = rng.permutation(pw.shape[0])
                w_shuf = pw[idx]
                b_shuf = pb[idx]
                win_a.append(w_shuf[:n_e])
                win_b.append(w_shuf[n_e:])
                bl_a.append(b_shuf[:n_e])
                bl_b.append(b_shuf[n_e:])

            for dim_name, fn in MOTOR_DIM_FNS.items():
                w_a = np.concatenate([fn(w, b) for w, b in zip(win_a, bl_a)])
                w_b = np.concatenate([fn(w, b) for w, b in zip(win_b, bl_b)])
                null_cos[dim_name][p] = cosine_similarity(w_a, w_b)

        for dim_name in MOTOR_DIM_FNS:
            between_block[dim_name][wl] = {'real': real_cos[dim_name],
                                           'null': null_cos[dim_name]}

    result = {
        'lick_t_ax': lick_t_ax,
        'dimensions': dimensions,
        'between_block_cosine': between_block,
        'cross_projections': cross_projections,
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
