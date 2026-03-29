"""coding dimension extraction: TF and premotor coding directions by block"""

import numpy as np
import pickle
import gc
from pathlib import Path
from sklearn.decomposition import PCA

from config import PATHS, ANALYSIS_OPTIONS, CODING_DIM_OPS
from data.load_responses import load_psth
from data.session import Session
from utils.filing import get_session_dirs_by_animal
from utils.smoothing import causal_boxcar


#%% shared utilities

def _get_window_bins(bm_ops):
    dt = ANALYSIS_OPTIONS['pop_bin_width']
    return max(1, int(round(bm_ops['sliding_window_ms'] / 1000 / dt)))


def _window_label(win):
    return f'{win[0]:.2f}_{win[1]:.2f}'


def _time_mask(t_ax, win):
    """boolean mask for time bins within window [win[0], win[1])"""
    return (t_ax >= win[0]) & (t_ax < win[1])


def coding_direction(resp_a, resp_b):
    """
    normalised difference of means between two sets of responses.
    resp_a, resp_b: (n_trials, n_dims) or (n_timepoints, n_dims)
    returns unit vector (n_dims,)
    """
    w = np.nanmean(resp_a, axis=0) - np.nanmean(resp_b, axis=0)
    norm = np.linalg.norm(w)
    if norm > 0:
        w /= norm
    return w


def cosine_similarity(a, b):
    """cosine similarity between two vectors"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return np.clip(np.dot(a, b) / (na * nb), -1, 1)


#%% data loading

def _load_tf_trials_by_block(sess_dir, ops=ANALYSIS_OPTIONS, bm_ops=CODING_DIM_OPS):
    """
    load per-trial outlier TF responses, split by block and fast/slow.
    returns {block: {'fast': (nEv, nN, nT), 'slow': (nEv, nN, nT)}}, t_ax
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


def _load_lick_trials_by_block(sess_dir, lick_type='fa'):
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
    """both blocks need fast and slow TF trials"""
    return all(tf_data[b]['fast'].shape[0] > 0 and tf_data[b]['slow'].shape[0] > 0
               for b in ['early', 'late'])


def _session_valid_for_motor(lick_data, min_trials=2):
    """both blocks need sufficient lick trials"""
    return all(lick_data[b] is not None and lick_data[b].shape[0] >= min_trials
               for b in ['early', 'late'])


#%% TF coding dimensions

def extract_tf_dimensions(npx_dir=PATHS['npx_dir_local'],
                          ops=ANALYSIS_OPTIONS,
                          bm_ops=CODING_DIM_OPS):
    """
    extract TF coding directions (fast vs slow) at defined time windows,
    per block, per animal (pseudo-population).
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)
    window_bins = _get_window_bins(bm_ops)
    windows = bm_ops['tf_coding_windows']
    n_perm = bm_ops['n_permutations']
    all_results = {}
    save_dir = Path(npx_dir) / 'coding_dims'
    save_dir.mkdir(exist_ok=True)

    for animal, sess_dirs in animal_sessions.items():
        print(f'TF coding dimensions: {animal}')

        sess_fast = {'early': [], 'late': []}
        sess_slow = {'early': [], 'late': []}
        included_sessions = []
        n_neurons_per_session = []
        tf_t_ax = None

        for sess_dir in sess_dirs:
            data, t_ax = _load_tf_trials_by_block(sess_dir, ops, bm_ops)
            if tf_t_ax is None:
                tf_t_ax = t_ax
            if not _session_valid_for_tf(data):
                continue
            for block in ['early', 'late']:
                fast = causal_boxcar(data[block]['fast'], window_bins, axis=-1)
                slow = causal_boxcar(data[block]['slow'], window_bins, axis=-1)
                sess_fast[block].append(fast)
                sess_slow[block].append(slow)
            included_sessions.append(sess_dir.name)
            n_neurons_per_session.append(data['early']['fast'].shape[1])

        if not sess_fast['early']:
            print(f'  Skipping {animal}: no valid sessions')
            continue

        n_sessions = len(included_sessions)
        n_total = sum(n_neurons_per_session)
        print(f'  {n_sessions} sessions, {n_total} neurons')

        # compute coding directions per block and window
        dimensions = {}
        for block in ['early', 'late']:
            dimensions[block] = {}
            for win in windows:
                wl = _window_label(win)
                t_mask = _time_mask(tf_t_ax, win)
                if t_mask.sum() == 0:
                    continue

                fast_means = []
                slow_means = []
                for s_fast, s_slow in zip(sess_fast[block], sess_slow[block]):
                    fast_means.append(np.nanmean(s_fast[:, :, t_mask], axis=(0, 2)))
                    slow_means.append(np.nanmean(s_slow[:, :, t_mask], axis=(0, 2)))

                fast_pop = np.concatenate(fast_means)
                slow_pop = np.concatenate(slow_means)

                w = fast_pop - slow_pop
                norm = np.linalg.norm(w)
                if norm > 0:
                    w /= norm
                dimensions[block][wl] = w

        # cross-block projections with separate fast and slow
        cross_projections = {}
        for proj_block in ['early', 'late']:
            cross_projections[proj_block] = {}
            for dim_block in ['early', 'late']:
                cross_projections[proj_block][dim_block] = {}
                for wl, w in dimensions[dim_block].items():
                    fast_resp = []
                    slow_resp = []
                    for s_fast, s_slow in zip(sess_fast[proj_block],
                                               sess_slow[proj_block]):
                        fast_resp.append(np.nanmean(s_fast, axis=0))
                        slow_resp.append(np.nanmean(s_slow, axis=0))
                    fast_pop_mean = np.concatenate(fast_resp, axis=0)
                    slow_pop_mean = np.concatenate(slow_resp, axis=0)
                    cross_projections[proj_block][dim_block][wl] = {
                        'fast': w @ fast_pop_mean,
                        'slow': w @ slow_pop_mean,
                    }

        # between-block cosine similarity + null
        between_block = {}
        for win in windows:
            wl = _window_label(win)
            if wl not in dimensions['early'] or wl not in dimensions['late']:
                continue
            real_cos = cosine_similarity(dimensions['early'][wl],
                                         dimensions['late'][wl])

            t_mask = _time_mask(tf_t_ax, win)
            null_cos = np.full(n_perm, np.nan)

            # pool trials across blocks per session
            sess_pooled_fast = []
            sess_pooled_slow = []
            sess_n_early_fast = []
            sess_n_early_slow = []
            for sf_e, sf_l, ss_e, ss_l in zip(
                    sess_fast['early'], sess_fast['late'],
                    sess_slow['early'], sess_slow['late']):
                sess_pooled_fast.append(np.concatenate([sf_e, sf_l], axis=0))
                sess_pooled_slow.append(np.concatenate([ss_e, ss_l], axis=0))
                sess_n_early_fast.append(sf_e.shape[0])
                sess_n_early_slow.append(ss_e.shape[0])

            rng = np.random.default_rng(0)
            for p in range(n_perm):
                fast_means_a, fast_means_b = [], []
                slow_means_a, slow_means_b = [], []
                for pooled_f, pooled_s, nef, nes in zip(
                        sess_pooled_fast, sess_pooled_slow,
                        sess_n_early_fast, sess_n_early_slow):
                    idx_f = rng.permutation(pooled_f.shape[0])
                    f_shuf = pooled_f[idx_f]
                    fast_means_a.append(np.nanmean(f_shuf[:nef, :, :][:, :, t_mask], axis=(0, 2)))
                    fast_means_b.append(np.nanmean(f_shuf[nef:, :, :][:, :, t_mask], axis=(0, 2)))
                    idx_s = rng.permutation(pooled_s.shape[0])
                    s_shuf = pooled_s[idx_s]
                    slow_means_a.append(np.nanmean(s_shuf[:nes, :, :][:, :, t_mask], axis=(0, 2)))
                    slow_means_b.append(np.nanmean(s_shuf[nes:, :, :][:, :, t_mask], axis=(0, 2)))

                fa = np.concatenate(fast_means_a)
                sa = np.concatenate(slow_means_a)
                fb = np.concatenate(fast_means_b)
                sb = np.concatenate(slow_means_b)

                w_a = fa - sa
                w_b = fb - sb
                na, nb = np.linalg.norm(w_a), np.linalg.norm(w_b)
                if na > 0 and nb > 0:
                    null_cos[p] = cosine_similarity(w_a / na, w_b / nb)

            between_block[wl] = {'real': real_cos, 'null': null_cos}

        all_results[animal] = {
            'tf_t_ax': tf_t_ax,
            'dimensions': dimensions,
            'between_block_cosine': between_block,
            'cross_projections': cross_projections,
            'included_sessions': included_sessions,
            'n_neurons_per_session': n_neurons_per_session,
        }

        del sess_fast, sess_slow
        gc.collect()

    with open(save_dir / 'tf_dimensions.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved TF dimensions to {save_dir / "tf_dimensions.pkl"}')

    return all_results


#%% premotor coding dimensions

def extract_motor_dimensions(npx_dir=PATHS['npx_dir_local'],
                             ops=ANALYSIS_OPTIONS,
                             bm_ops=CODING_DIM_OPS,
                             lick_type='fa'):
    """
    extract premotor coding directions (pre-lick window vs baseline) at
    defined time windows, per block, per animal (pseudo-population).
    PCA denoising before coding direction extraction.
    lick_type: 'fa' for false alarms only (motor-only), 'hit', or 'all'
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)
    window_bins = _get_window_bins(bm_ops)
    windows = bm_ops['motor_coding_windows']
    bl_win = bm_ops['motor_baseline_window']
    n_pcs = bm_ops['motor_denoise_pcs']
    n_perm = bm_ops['n_permutations']
    all_results = {}
    save_dir = Path(npx_dir) / 'coding_dims'
    save_dir.mkdir(exist_ok=True)

    for animal, sess_dirs in animal_sessions.items():
        print(f'Motor coding dimensions: {animal}')

        sess_lick = {'early': [], 'late': []}
        included_sessions = []
        n_neurons_per_session = []
        lick_t_ax = None

        for sess_dir in sess_dirs:
            data, t_ax = _load_lick_trials_by_block(sess_dir, lick_type)
            if t_ax is not None and lick_t_ax is None:
                lick_t_ax = t_ax
            # require both blocks to have sufficient trials
            if not _session_valid_for_motor(data):
                continue
            for block in ['early', 'late']:
                resp = causal_boxcar(data[block], window_bins, axis=-1)
                sess_lick[block].append(resp)
            included_sessions.append(sess_dir.name)
            n_neurons_per_session.append(data['early'].shape[1])

        if not sess_lick['early'] or lick_t_ax is None:
            print(f'  Skipping {animal}: insufficient lick data')
            continue

        n_sessions = len(included_sessions)
        n_total = sum(n_neurons_per_session)
        print(f'  {n_sessions} sessions, {n_total} neurons')

        # trial-average per session, concatenate neurons -> (nN_total, nT)
        lick_mean = {}
        for block in ['early', 'late']:
            means = [np.nanmean(r, axis=0) for r in sess_lick[block]]
            lick_mean[block] = np.concatenate(means, axis=0)

        bl_mask = _time_mask(lick_t_ax, bl_win)

        # PCA denoising: fit on pooled lick trajectories (both blocks)
        pooled_traj = np.concatenate([lick_mean['early'].T, lick_mean['late'].T], axis=0)
        pca_mean = np.nanmean(pooled_traj, axis=0)
        pooled_centred = pooled_traj - pca_mean
        actual_pcs = min(n_pcs, min(pooled_centred.shape))
        pca = PCA(n_components=actual_pcs)
        pca.fit(pooled_centred)

        # extract coding directions per block
        dimensions = {}
        dimensions_neuron = {}

        for block in ['early', 'late']:
            dimensions[block] = {}
            dimensions_neuron[block] = {}
            traj = lick_mean[block].T  # (nT, nN)
            scores = (traj - pca_mean) @ pca.components_.T  # (nT, k)

            bl_scores = scores[bl_mask]

            for win in windows:
                wl = _window_label(win)
                t_mask = _time_mask(lick_t_ax, win)
                if t_mask.sum() == 0:
                    continue
                win_scores = scores[t_mask]

                w_pc = coding_direction(win_scores, bl_scores)
                dimensions[block][wl] = w_pc

                w_neuron = pca.components_.T @ w_pc
                norm = np.linalg.norm(w_neuron)
                if norm > 0:
                    w_neuron = w_neuron / norm
                dimensions_neuron[block][wl] = w_neuron

        # between-block cosine similarity + null (trial-level block label shuffling)
        between_block = {}

        # pool per-trial data per session for null
        sess_pooled_lick = []
        sess_n_early_lick = []
        for s_early, s_late in zip(sess_lick['early'], sess_lick['late']):
            sess_pooled_lick.append(np.concatenate([s_early, s_late], axis=0))
            sess_n_early_lick.append(s_early.shape[0])

        for win in windows:
            wl = _window_label(win)
            if wl not in dimensions_neuron['early'] or wl not in dimensions_neuron['late']:
                continue
            real_cos = cosine_similarity(dimensions_neuron['early'][wl],
                                         dimensions_neuron['late'][wl])

            t_mask = _time_mask(lick_t_ax, win)
            rng = np.random.default_rng(0)
            null_cos = np.full(n_perm, np.nan)

            for p in range(n_perm):
                means_a, means_b = [], []
                for pooled, n_e in zip(sess_pooled_lick, sess_n_early_lick):
                    idx = rng.permutation(pooled.shape[0])
                    shuf = pooled[idx]
                    means_a.append(np.nanmean(shuf[:n_e], axis=0))   # (nN, nT)
                    means_b.append(np.nanmean(shuf[n_e:], axis=0))

                pop_a = np.concatenate(means_a, axis=0).T  # (nT, nN_total)
                pop_b = np.concatenate(means_b, axis=0).T

                sc_a = (pop_a - pca_mean) @ pca.components_.T
                sc_b = (pop_b - pca_mean) @ pca.components_.T

                w_a_pc = coding_direction(sc_a[t_mask], sc_a[bl_mask])
                w_b_pc = coding_direction(sc_b[t_mask], sc_b[bl_mask])

                w_a_n = pca.components_.T @ w_a_pc
                w_b_n = pca.components_.T @ w_b_pc
                na, nb = np.linalg.norm(w_a_n), np.linalg.norm(w_b_n)
                if na > 0 and nb > 0:
                    null_cos[p] = cosine_similarity(w_a_n, w_b_n)

            between_block[wl] = {'real': real_cos, 'null': null_cos}

        # time-resolved projections onto motor dimensions (in neuron space)
        projections_lick = {}
        for proj_block in ['early', 'late']:
            projections_lick[proj_block] = {}
            for dim_block in ['early', 'late']:
                projections_lick[proj_block][dim_block] = {}
                for wl, w in dimensions_neuron[dim_block].items():
                    projections_lick[proj_block][dim_block][wl] = (
                        w @ lick_mean[proj_block])

        all_results[animal] = {
            'lick_t_ax': lick_t_ax,
            'dimensions_pc': dimensions,
            'dimensions_neuron': dimensions_neuron,
            'pca_components': pca.components_,
            'pca_mean': pca_mean,
            'between_block_cosine': between_block,
            'projections_lick': projections_lick,
            'included_sessions': included_sessions,
            'n_neurons_per_session': n_neurons_per_session,
        }

        del sess_lick, lick_mean
        gc.collect()

    with open(save_dir / 'motor_dimensions.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved motor dimensions to {save_dir / "motor_dimensions.pkl"}')

    return all_results


#%% cross-type analysis: TF dims vs motor dims

def extract_cross_type_analysis(npx_dir=PATHS['npx_dir_local'],
                                bm_ops=CODING_DIM_OPS):
    """
    compare TF and motor coding directions: cosine similarity between
    all pairs of TF and motor dimensions, per block.
    only compares animals where both analyses used the same sessions.
    """
    save_dir = Path(npx_dir) / 'coding_dims'

    with open(save_dir / 'tf_dimensions.pkl', 'rb') as f:
        tf_results = pickle.load(f)
    with open(save_dir / 'motor_dimensions.pkl', 'rb') as f:
        motor_results = pickle.load(f)

    animals = set(tf_results.keys()) & set(motor_results.keys())
    all_results = {}

    for animal in sorted(animals):
        tf_r = tf_results[animal]
        motor_r = motor_results[animal]

        # check that both analyses used the same sessions
        tf_sessions = tf_r.get('included_sessions', [])
        motor_sessions = motor_r.get('included_sessions', [])
        if tf_sessions != motor_sessions:
            print(f'  {animal}: session mismatch '
                  f'(TF: {len(tf_sessions)}, motor: {len(motor_sessions)}) - skipping')
            continue

        # cross-type cosine similarity per block
        cross_cosine = {}
        for block in ['early', 'late']:
            tf_dims = tf_r['dimensions'].get(block, {})
            motor_dims = motor_r['dimensions_neuron'].get(block, {})
            cross_cosine[block] = {}
            for tf_wl, tf_w in tf_dims.items():
                for motor_wl, motor_w in motor_dims.items():
                    key = f'tf_{tf_wl}_x_motor_{motor_wl}'
                    cross_cosine[block][key] = cosine_similarity(tf_w, motor_w)

        all_results[animal] = {
            'cross_cosine': cross_cosine,
        }

    with open(save_dir / 'cross_type_analysis.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved cross-type analysis to {save_dir / "cross_type_analysis.pkl"}')

    return all_results
