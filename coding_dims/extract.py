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
from utils.filing import get_session_dirs_by_animal
from utils.smoothing import causal_boxcar
from utils.time import window_label, time_mask
from utils.rois import AREA_GROUPS, in_group


#%% shared utilities

def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return np.clip(np.dot(a, b) / (na * nb), -1, 1)

def _get_window_bins(bm_ops):
    dt = ANALYSIS_OPTIONS['pop_bin_width']
    return max(1, int(round(bm_ops['sliding_window_ms'] / 1000 / dt)))


def _get_neuron_mask(sess_dir, area=None, unit_filter=None):
    """boolean mask for neurons matching area and/or GLM classification (OR logic)"""
    session = Session.load(str(sess_dir / 'session.pkl'))
    regions = session.unit_info['brain_region_comb'].values
    n = len(regions)

    if area is None or area == 'all':
        mask = np.ones(n, dtype=bool)
    elif area in AREA_GROUPS:
        mask = in_group(regions, area)
    else:
        mask = np.array([r == area for r in regions])

    if unit_filter is not None:
        glm_path = sess_dir / 'glm_classifications.csv'
        if not glm_path.exists():
            return np.zeros(n, dtype=bool)
        glm = pd.read_csv(glm_path)
        glm_mask = np.zeros(n, dtype=bool)
        n_glm = min(len(glm), n)
        for f in unit_filter:
            col = f'{f}_sig'
            if col in glm.columns:
                glm_mask[:n_glm] |= glm[col].values[:n_glm].astype(bool)
        mask &= glm_mask

    return mask


def _file_suffix(area=None, unit_filter=None):
    """build filename suffix from area and unit filter"""
    parts = [area if area and area != 'all' else 'all']
    if unit_filter:
        parts.append('-'.join(unit_filter))
    return '_'.join(parts)


#%% data loading

def _load_tf_resps_by_block(sess_dir, ops=ANALYSIS_OPTIONS, bm_ops=CODING_DIM_OPS):
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
    """both blocks need sufficient lick trials"""
    return all(lick_data[b] is not None and lick_data[b].shape[0] >= min_trials
               for b in ['early', 'late'])


#%% TF coding dimensions

def _process_tf_animal(animal, sess_dirs, ops, bm_ops, area, unit_filter, save_dir):
    """process a single animal to extract tf coding dim"""
    window_bins = _get_window_bins(bm_ops)
    windows = bm_ops['tf_coding_windows']
    n_perm = bm_ops['n_permutations']
    min_n = bm_ops.get('min_neurons', 5)
    suffix = _file_suffix(area, unit_filter)

    # (n_events, n_neurons) per window, for permutation
    sess_tavg = {window_label(w): {'early': {'fast': [], 'slow': []},
                                    'late':  {'fast': [], 'slow': []}}
                 for w in windows}
    # (n_neurons, n_time) event-averaged, for cross-projections
    sess_mean = {'early': {'fast': [], 'slow': []},
                 'late':  {'fast': [], 'slow': []}}
    # event counts per block per group, for permutation split
    sess_n = {'early': {'fast': [], 'slow': []},
              'late':  {'fast': [], 'slow': []}}
    included_sessions = []
    n_neurons_per_session = []
    unit_ids = []  # (session_name, cluster_id) per neuron in concatenation order

    for sess_dir in sess_dirs:
        data, t_ax = _load_tf_resps_by_block(sess_dir, ops, bm_ops)
        if not _session_valid_for_tf(data):
            continue

        neuron_mask = _get_neuron_mask(sess_dir, area, unit_filter)
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
                                 window_bins, axis=-1)
            slow = causal_boxcar(data[block]['slow'][:, neuron_mask, :],
                                 window_bins, axis=-1)

            sess_mean[block]['fast'].append(np.nanmean(fast, axis=0))
            sess_mean[block]['slow'].append(np.nanmean(slow, axis=0))
            sess_n[block]['fast'].append(fast.shape[0])
            sess_n[block]['slow'].append(slow.shape[0])

            for win in windows:
                wl = window_label(win)
                t_mask = time_mask(t_ax, win)
                sess_tavg[wl][block]['fast'].append(
                    np.nanmean(fast[:, :, t_mask], axis=2))
                sess_tavg[wl][block]['slow'].append(
                    np.nanmean(slow[:, :, t_mask], axis=2))

        included_sessions.append(sess_dir.name)
        n_neurons_per_session.append(int(neuron_mask.sum()))
        del data
        gc.collect()

    if not included_sessions:
        print(f'  skipping {animal}: no valid sessions for {suffix}')
        return animal, None

    n_sessions = len(included_sessions)
    n_total = sum(n_neurons_per_session)
    print(f'  {animal} [{suffix}]: {n_sessions} sessions, {n_total} neurons')

    # compute coding directions per block and window
    dimensions = {}
    for block in ['early', 'late']:
        dimensions[block] = {}
        for win in windows:
            wl = window_label(win)
            fast_means = [np.nanmean(s, axis=0)
                          for s in sess_tavg[wl][block]['fast']]
            slow_means = [np.nanmean(s, axis=0)
                          for s in sess_tavg[wl][block]['slow']]

            w = np.concatenate(fast_means) - np.concatenate(slow_means)
            norm = np.linalg.norm(w)
            if norm > 0:
                w /= norm
            dimensions[block][wl] = w

    # cross-block projections with separate fast and slow
    pop_mean = {}
    for block in ['early', 'late']:
        pop_mean[block] = {
            'fast': np.concatenate(sess_mean[block]['fast'], axis=0),
            'slow': np.concatenate(sess_mean[block]['slow'], axis=0),
        }
    del sess_mean

    cross_projections = {}
    for proj_block in ['early', 'late']:
        cross_projections[proj_block] = {}
        for dim_block in ['early', 'late']:
            cross_projections[proj_block][dim_block] = {}
            for wl, w in dimensions[dim_block].items():
                cross_projections[proj_block][dim_block][wl] = {
                    'fast': w @ pop_mean[proj_block]['fast'],
                    'slow': w @ pop_mean[proj_block]['slow'],
                }
    del pop_mean

    # between-block cosine similarity + null
    between_block = {}
    for win in windows:
        wl = window_label(win)
        if wl not in dimensions['early'] or wl not in dimensions['late']:
            continue
        real_cos = cosine_similarity(dimensions['early'][wl],
                                     dimensions['late'][wl])
        null_cos = np.full(n_perm, np.nan)

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
            fast_means_a, fast_means_b = [], []
            slow_means_a, slow_means_b = [], []
            for pf, ps, nef, nes in zip(
                    sess_pooled_fast, sess_pooled_slow,
                    n_early_fast, n_early_slow):
                idx_f = rng.permutation(pf.shape[0])
                f_shuf = pf[idx_f]
                fast_means_a.append(np.nanmean(f_shuf[:nef], axis=0))
                fast_means_b.append(np.nanmean(f_shuf[nef:], axis=0))
                idx_s = rng.permutation(ps.shape[0])
                s_shuf = ps[idx_s]
                slow_means_a.append(np.nanmean(s_shuf[:nes], axis=0))
                slow_means_b.append(np.nanmean(s_shuf[nes:], axis=0))

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

    result = {
        'tf_t_ax': t_ax,
        'dimensions': dimensions,
        'between_block_cosine': between_block,
        'cross_projections': cross_projections,
        'included_sessions': included_sessions,
        'n_neurons_per_session': n_neurons_per_session,
        'unit_ids': unit_ids,
    }

    with open(save_dir / f'tf_dimensions_{animal}_{suffix}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return animal, result


def extract_tf_dimensions(npx_dir=PATHS['npx_dir_local'],
                          ops=ANALYSIS_OPTIONS,
                          bm_ops=CODING_DIM_OPS,
                          area: str | None = None,
                          unit_filter: list[str] | None = None,
                          n_jobs: int | None =None):
    """
    extract TF coding directions (fast vs slow) at defined time windows,
    per block, per animal (pseudo-population).
    area: brain region, AREA_GROUPS key, or None/'all' for all neurons
    unit_filter: list of GLM classification names, OR logic (e.g. ['tf', 'lick_prep'])
    n_jobs: number of parallel workers (None = all cores)
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)
    save_dir = Path(npx_dir) / 'coding_dims'
    save_dir.mkdir(exist_ok=True)
    suffix = _file_suffix(area, unit_filter)

    args = [(animal, sess_dirs, ops, bm_ops, area, unit_filter, save_dir)
            for animal, sess_dirs in animal_sessions.items()]

    with Pool(n_jobs) as pool:
        results = pool.starmap(_process_tf_animal, args)

    all_results = {animal: res for animal, res in results if res is not None}

    out_path = save_dir / f'tf_dimensions_{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved TF dimensions to {out_path}')

    return all_results


#%% premotor coding dimensions

def _process_motor_animal(animal, sess_dirs, ops, bm_ops, lick_type,
                          area, unit_filter, save_dir):
    """process a single animal for motor coding dimensions"""
    window_bins = _get_window_bins(bm_ops)
    windows = bm_ops['motor_prelick_windows']
    bl_win = bm_ops['motor_baseline_window']
    n_perm = bm_ops['n_permutations']
    min_n = bm_ops.get('min_neurons', 5)
    suffix = _file_suffix(area, unit_filter)

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

        neuron_mask = _get_neuron_mask(sess_dir, area, unit_filter)
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

    # compute coding directions per block and window
    dimensions = {}
    for block in ['early', 'late']:
        dimensions[block] = {}
        for win in windows:
            wl = window_label(win)
            if not sess_tavg_win[wl][block]:
                continue
            win_means = [np.nanmean(s, axis=0)
                         for s in sess_tavg_win[wl][block]]
            bl_means = [np.nanmean(s, axis=0)
                        for s in sess_tavg_bl[block]]

            w = np.concatenate(win_means) - np.concatenate(bl_means)
            norm = np.linalg.norm(w)
            if norm > 0:
                w /= norm
            dimensions[block][wl] = w

    # cross-block projections
    pop_mean = {}
    for block in ['early', 'late']:
        pop_mean[block] = np.concatenate(sess_mean[block], axis=0)
    del sess_mean

    cross_projections = {}
    for proj_block in ['early', 'late']:
        cross_projections[proj_block] = {}
        for dim_block in ['early', 'late']:
            cross_projections[proj_block][dim_block] = {}
            for wl, w in dimensions[dim_block].items():
                cross_projections[proj_block][dim_block][wl] = (
                    w @ pop_mean[proj_block])
    del pop_mean

    # between-block cosine similarity + null
    between_block = {}

    for win in windows:
        wl = window_label(win)
        if wl not in dimensions['early'] or wl not in dimensions['late']:
            continue
        real_cos = cosine_similarity(dimensions['early'][wl],
                                     dimensions['late'][wl])
        null_cos = np.full(n_perm, np.nan)

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
            win_means_a, win_means_b = [], []
            bl_means_a, bl_means_b = [], []
            for pw, pb, n_e in zip(sess_pooled_win, sess_pooled_bl, n_early):
                idx = rng.permutation(pw.shape[0])
                w_shuf = pw[idx]
                b_shuf = pb[idx]
                win_means_a.append(np.nanmean(w_shuf[:n_e], axis=0))
                win_means_b.append(np.nanmean(w_shuf[n_e:], axis=0))
                bl_means_a.append(np.nanmean(b_shuf[:n_e], axis=0))
                bl_means_b.append(np.nanmean(b_shuf[n_e:], axis=0))

            wa = np.concatenate(win_means_a) - np.concatenate(bl_means_a)
            wb = np.concatenate(win_means_b) - np.concatenate(bl_means_b)
            na, nb = np.linalg.norm(wa), np.linalg.norm(wb)
            if na > 0 and nb > 0:
                null_cos[p] = cosine_similarity(wa / na, wb / nb)

        between_block[wl] = {'real': real_cos, 'null': null_cos}

    result = {
        'lick_t_ax': lick_t_ax,
        'dimensions': dimensions,
        'between_block_cosine': between_block,
        'cross_projections': cross_projections,
        'included_sessions': included_sessions,
        'n_neurons_per_session': n_neurons_per_session,
        'unit_ids': unit_ids,
    }

    with open(save_dir / f'motor_dimensions_{animal}_{suffix}.pkl', 'wb') as f:
        pickle.dump(result, f)

    return animal, result


def extract_motor_dimensions(npx_dir=PATHS['npx_dir_local'],
                             ops=ANALYSIS_OPTIONS,
                             bm_ops=CODING_DIM_OPS,
                             lick_type='fa',
                             area: str | None = None,
                             unit_filter: list[str] | None = None,
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
    suffix = _file_suffix(area, unit_filter)

    args = [(animal, sess_dirs, ops, bm_ops, lick_type, area, unit_filter, save_dir)
            for animal, sess_dirs in animal_sessions.items()]

    with Pool(n_jobs) as pool:
        results = pool.starmap(_process_motor_animal, args)

    all_results = {animal: res for animal, res in results if res is not None}

    out_path = save_dir / f'motor_dimensions_{suffix}.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved motor dimensions to {out_path}')

    return all_results
