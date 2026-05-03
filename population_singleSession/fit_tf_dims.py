"""
Get baseline-period activity (by area, by session) and stimulus log2(TF) to compute
TF-coding and TF-null spaces - single sessions.

Regression-based, analogous to movement potent/null space (fit_move_potent_null.py),
but: continuous (single-trial baseline period, not event-aligned), 1D target
(log2(TF)), and TF leads neural in the lag sweep.

These are extracted independently of movement space, but can be rotated to be confined
to null space/aligned to motor etc.
"""

from config import PATHS, ANALYSIS_OPTIONS, TFDIM_OPTIONS
from utils.rois import AREA_GROUPS
AREA_NAMES = list(AREA_GROUPS.keys())
from utils.filing import get_session_dirs_by_animal, load_fr_matrix
from data.session import Session
from data.stimulus import build_stim_vector
from data.responses import compute_psth

import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
from scipy.linalg import null_space


PULSE_PSTH_WIN  = (0.0, 0.5)   # s; tf-pulse window for null-space rotation
N_NULL          = 2            # null-space dims kept after rotation
LASSO_CV_FOLDS  = 5
LASSO_N_ALPHAS  = 20


def _baseline_window(row, ops):
    """(t_start, t_end) for the baseline period of a trial, or None if invalid"""
    bl_start = row['Baseline_ON_rise'] + ops['rmv_time_around_bl']
    if row['IsHit'] or row['IsMiss']:
        bl_end = row['Change_ON_rise']
    elif row['IsFA']:
        bl_end = row['motion_onset'] - ops['rmv_time_around_move']
    elif row['IsAbort']:
        bl_end = (row['Baseline_ON_rise'] + row['rt_abort']
                  - ops['rmv_time_around_move'])
    else:
        return None
    if pd.isna(bl_start) or pd.isna(bl_end):
        return None
    return bl_start, bl_end


def get_baseline_tf_by_trial(session: Session,
                             ops: dict,
                             tf_ops: dict) -> list:
    """
    Per-trial baseline-period log2(TF) on the FR-matrix time grid.
    Returns list of dicts (invalid trials skipped) with:
    ['tf']: log2(TF) trace, length nT_bl
    ['t_idx']: indices into FR matrix columns spanning the baseline window
    ['trial']: original trial index
    ['block'], ['tr_in_block']: metadata for block masking
    """
    fr_t = session.fr_matrix.columns.values
    U = build_stim_vector(session, fr_t)[0]  # full-session log2(TF)

    out = []
    for tr, row in session.trials.iterrows():
        win = _baseline_window(row, ops)
        if win is None:
            continue
        bl_start, bl_end = win
        if bl_end - bl_start < tf_ops['min_baseline_dur']:
            continue
        t_idx = np.where((fr_t >= bl_start) & (fr_t < bl_end))[0]
        if len(t_idx) < 2:
            continue

        out.append({
            'tf': U[t_idx],
            't_idx': t_idx,
            'trial': tr,
            'block': row['hazardblock'],
            'tr_in_block': row['tr_in_block'],
        })
    return out


def get_baseline_activity_by_area(session: Session,
                                  bl_per_trial: list,
                                  areas: list[str]) -> dict:
    """
    Iterate through areas and extract baseline-period activity per trial.
    Returns dict keyed by area, each value also a dict with:
    ['cids']: list of unit ids,
    ['X']: list of nN x nT_bl np arrays (one per valid trial, aligned to bl_per_trial)
    """
    bl_activity = dict()
    for group in areas:
        in_area = session.area_mask(AREA_GROUPS[group])
        if not any(in_area):
            continue
        fr = session.fr_matrix.values[in_area, :]
        bl_activity[group] = {
            'cids': session.unit_info.cluster_id.values[in_area],
            'X': [fr[:, e['t_idx']] for e in bl_per_trial],
        }
    return bl_activity


def _centre_tf(tf_by_block):
    mu = np.nanmean(np.concatenate(tf_by_block['all']['tf']))
    for block in tf_by_block:
        tf_by_block[block]['tf_centred'] = [arr - mu
                                            for arr in tf_by_block[block]['tf']]
    return tf_by_block


def _centre_neural(sp_by_block):
    for area, d in sp_by_block['all'].items():
        mu = np.nanmean(np.concatenate(d['X'], axis=1), axis=1, keepdims=True)
        for block in sp_by_block:
            if area not in sp_by_block[block]:
                continue
            sp_by_block[block][area]['X_centred'] = [
                x - mu for x in sp_by_block[block][area]['X']
            ]
    return sp_by_block


def _centre_signals(tf_by_block, sp_by_block):
    # centre log2(TF) by mean over 'all'
    tf_by_block = _centre_tf(tf_by_block)
    # subtract per-neuron baseline-period mean (computed on 'all')
    sp_by_block = _centre_neural(sp_by_block)
    return tf_by_block, sp_by_block


def _lag_concat(X_per_tr: list, tf_per_tr: list, k: int):
    """
    Within each trial, drop first k samples of neural and last k samples of TF, then
    concatenate across trials. TF[t-k] is paired with neural[t] (TF leads neural).
    Also returns groups: trial index per concatenated sample (for grouped CV).
    """
    Xs, ys, groups = [], [], []
    for tr_idx, (X_tr, tf_tr) in enumerate(zip(X_per_tr, tf_per_tr)):
        n = len(tf_tr)
        if n <= k:
            continue
        Xs.append(X_tr[k:])
        ys.append(tf_tr[:n-k])
        groups.append(np.full(n - k, tr_idx))
    if not Xs:
        return None, None, None
    return (np.concatenate(Xs, axis=0),
            np.concatenate(ys),
            np.concatenate(groups))


def _tf_pulse_means_by_area_block(session: Session,
                                  areas: list[str],
                                  ops: dict) -> dict:
    """mean tf-pulse-aligned activity per (area, block), pos and neg means
    concatenated along time. window = PULSE_PSTH_WIN.
    returns: {area: {block: (nN, 2 * nT_psth)}}"""
    pulses = session.tf_pulses
    fr = session.fr_matrix
    t_ax = fr.columns.values

    non_trans = (pulses['tr_in_block'] > ops['ignore_first_trials_in_block']).values
    t_to_event = np.fmin(pulses['time_to_lick'].values,
                         pulses['time_to_abort'].values)
    valid = ((pulses['tr_time'].values > ops['rmv_time_around_bl']) &
             (t_to_event > ops['rmv_time_around_move']))
    pos = (pulses['tf'].values > 0)
    block_arr = pulses['block'].values

    block_masks = {
        'all':   non_trans & valid,
        'early': (block_arr == 'early') & non_trans & valid,
        'late':  (block_arr == 'late')  & non_trans & valid,
    }

    out = {}
    for area in areas:
        in_area = session.area_mask(AREA_GROUPS[area])
        if not any(in_area):
            continue
        X = fr.values[in_area, :]
        out[area] = {}
        for block, mask in block_masks.items():
            pos_t = pulses.loc[mask & pos,  'time'].values
            neg_t = pulses.loc[mask & ~pos, 'time'].values
            if len(pos_t) == 0 or len(neg_t) == 0:
                continue
            pos_psth, _ = compute_psth(X, t_ax, pos_t, resp_win=PULSE_PSTH_WIN)
            neg_psth, _ = compute_psth(X, t_ax, neg_t, resp_win=PULSE_PSTH_WIN)
            out[area][block] = np.concatenate(
                [pos_psth.mean(axis=0), neg_psth.mean(axis=0)], axis=1)
    return out


def calculate_tf_dims_by_area(tf_by_block: dict,
                              sp_by_block: dict,
                              bin_size: float,
                              pulse_means: dict,
                              tf_ops: dict = TFDIM_OPTIONS):
    """
    fit TF-coding axis per area/block via LassoCV directly on lagged centred neural
    activity → log2(TF). lag picked on 'all' (per area), reused for early/late.
    null space rotated via PCA on the mean tf-pulse-aligned PSTH (pos and neg pooled,
    PULSE_PSTH_WIN window) — top N_NULL dims kept.
    Returns: tf_dims[area], dict with 'cids', 'delay' (s), and per-block:
        pot:  1 x nN
        null: N_NULL x nN
    """
    blocks = list(tf_by_block.keys())
    delay_bins = np.arange(int(round(tf_ops['max_lag_s'] / bin_size)) + 1)

    tf_dims = dict()
    for area in sp_by_block['all'].keys():

        X_list_all = sp_by_block['all'][area]['X_centred']
        n_neurons = X_list_all[0].shape[0] if X_list_all else 0
        if n_neurons < tf_ops['min_neurons']:
            continue
        if area not in pulse_means:
            continue
        tf_dims[area] = {'cids': sp_by_block['all'][area]['cids']}

        # per-trial neural samples (samples on rows, float64 to avoid LassoCV
        # Gram-matrix precision errors); per-trial centred TF
        Xs = {b: [x.T.astype(np.float64) for x in sp_by_block[b][area]['X_centred']]
              for b in blocks}
        ys = {b: [y.astype(np.float64) for y in tf_by_block[b]['tf_centred']]
              for b in blocks}

        # pick delay on 'all' by max in-sample R2 (TF leads neural).
        # CV folds hold out whole trials (samples within a trial are temporally
        # correlated, so random KFold would leak across folds).
        r2s = []
        for k in delay_bins:
            Xc, yc, gc = _lag_concat(Xs['all'], ys['all'], k)
            if Xc is None:
                r2s.append(-np.inf)
                continue
            cv = list(GroupKFold(n_splits=min(LASSO_CV_FOLDS, len(np.unique(gc))))
                      .split(Xc, yc, groups=gc))
            mdl = LassoCV(cv=cv, alphas=LASSO_N_ALPHAS).fit(Xc, yc)
            r2s.append(mdl.score(Xc, yc))
        best_k = int(delay_bins[np.argmax(r2s)])
        tf_dims[area]['delay'] = best_k * bin_size

        for block in blocks:
            Xc, yc, gc = _lag_concat(Xs[block], ys[block], best_k)
            if Xc is None or len(Xc) < 2:
                continue

            # lasso regression in neural space → w_pot (1 x nN), trial-grouped CV
            cv = list(GroupKFold(n_splits=min(LASSO_CV_FOLDS, len(np.unique(gc))))
                      .split(Xc, yc, groups=gc))
            pot_mdl = LassoCV(cv=cv, alphas=LASSO_N_ALPHAS).fit(Xc, yc)
            w_pot = pot_mdl.coef_.reshape(1, -1)
            if not np.any(w_pot):
                continue

            # full null space basis (nN x nN-1)
            null_basis = null_space(w_pot)

            # rotate null basis to capture variance of mean tf-pulse psth
            mean_psth = pulse_means[area].get(block)
            if mean_psth is None:
                continue
            null_coords = null_basis.T @ mean_psth          # (nN-1, 2*nT_psth)
            n_keep = min(N_NULL, null_coords.shape[0])
            rot = PCA(n_components=n_keep).fit(null_coords.T)

            tf_dims[area][block] = {
                'pot':  w_pot,
                'null': rot.components_ @ null_basis.T,     # (n_keep, nN)
            }

    return tf_dims


def fit_tf_dims_per_session(npx_dir: str = PATHS['npx_dir_local'],
                            ops: dict = ANALYSIS_OPTIONS,
                            tf_ops: dict = TFDIM_OPTIONS,
                            areas: list[str] = AREA_NAMES,
                            ):
    """
    Runner to loop through all sessions, extract baseline-period neural activity and
    stimulus log2(TF), calculate TF-coding/null dims per area.
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)

    for animal in animal_sessions.keys():
        print(f'Fitting TF space for {animal}')
        sessions = animal_sessions[animal]
        for session_dir in sessions:
            print(f'    {session_dir}')

            session = Session.load(session_dir/'session.pkl')
            fr = load_fr_matrix(str(session_dir / 'FR_matrix.parquet'))
            session.fr_matrix = fr

            # per-trial baseline TF + indices into fr_matrix columns
            bl_per_trial = get_baseline_tf_by_trial(session, ops, tf_ops)
            if len(bl_per_trial) < tf_ops['min_trials']:
                print(f'      skipping - {len(bl_per_trial)} valid trials '
                      f'(< {tf_ops["min_trials"]})')
                continue

            # baseline activity per area
            bl_activity = get_baseline_activity_by_area(session, bl_per_trial, areas)

            # trial-axis masks per block; transition trials excluded from early/late
            blocks = np.array([e['block'] for e in bl_per_trial])
            tr_in_block = np.array([e['tr_in_block'] for e in bl_per_trial])
            non_trans = tr_in_block > ops['ignore_first_trials_in_block']
            block_masks = {
                'all':   np.ones(len(bl_per_trial), dtype=bool),
                'early': (blocks == 'early') & non_trans,
                'late':  (blocks == 'late')  & non_trans,
            }
            # skip session if 'all' under threshold; drop early/late if either under
            min_tr = tf_ops['min_trials']
            if block_masks['all'].sum() < min_tr:
                print(f'      skipping - {block_masks["all"].sum()} trials '
                      f'(< {min_tr})')
                continue
            if (block_masks['early'].sum() < min_tr or
                block_masks['late'].sum() < min_tr):
                print(f'      early/late under {min_tr} trials - "all" only')
                block_masks = {'all': block_masks['all']}

            sp_by_block, tf_by_block = dict(), dict()
            for block, mask in block_masks.items():
                idx = np.where(mask)[0]
                tf_by_block[block] = {'tf': [bl_per_trial[i]['tf'] for i in idx]}
                sp_by_block[block] = {
                    area: {'cids': v['cids'],
                           'X':    [v['X'][i] for i in idx]}
                    for area, v in bl_activity.items()
                }
            tf_by_block, sp_by_block = _centre_signals(tf_by_block, sp_by_block)

            bin_size = float(np.median(np.diff(session.fr_matrix.columns.values)))
            pulse_means = _tf_pulse_means_by_area_block(session, areas, ops)
            tf_dims = calculate_tf_dims_by_area(tf_by_block, sp_by_block, bin_size,
                                                pulse_means, tf_ops)

            # save
            save_path = session_dir / 'tf_dims.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(tf_dims, f)

if __name__ == '__main__':
    fit_tf_dims_per_session()