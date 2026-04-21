"""
unregularised poisson glm - no group lasso, plain MLE
"""
import numpy as np
import pandas as pd
import pickle
import os
import gc
from pathlib import Path

from config import PATHS, ANALYSIS_OPTIONS, GLM_OPTIONS
from data.session import Session
from utils.filing import load_fr_matrix
from neuron_prediction.data import (
    load_glm_inputs, get_trial_fold_indices, neuron_seed,
    normalise_design_matrix,
)
from neuron_prediction.evaluate import pearson_r, reduce_design_matrix
from neuron_prediction.results.peth import build_event_spec, fold_peths


#%% spike counts

def build_spike_counts(fr_path, fr_stats, bin_factor=5):
    """un-z-score 10ms FR matrix and sum into 50ms spike counts

    returns (nN x T) uint16 array and (T,) time axis
    """
    fr = load_fr_matrix(fr_path)
    t_ax_10ms = fr.columns.values.astype(float)

    # un-z-score: rate_hz = z * sd + mean
    mu = fr_stats['mean'].values[:, None]
    sd = fr_stats['sd'].values[:, None]
    rate = fr.values * sd + mu

    # convert to spike counts per 10ms bin
    counts_10ms = rate * ANALYSIS_OPTIONS['sp_bin_width']

    # sum groups of bin_factor bins to get 50ms counts
    n_keep = (counts_10ms.shape[1] // bin_factor) * bin_factor
    counts = counts_10ms[:, :n_keep].reshape(counts_10ms.shape[0], -1, bin_factor).sum(axis=2)
    t_ax = t_ax_10ms[:n_keep].reshape(-1, bin_factor).mean(axis=1)

    # clip any tiny negative values from floating point and round
    counts = np.clip(counts, 0, None).round().astype(np.uint16)

    return counts, t_ax


#%% time-shift utility

def _time_shift(signal, kernel_win, bin_width=GLM_OPTIONS['bin_width']):
    """expand a (T,) signal into (T, n_lags) by shifting at each lag

    kernel_win: (start, end) in seconds
    e.g. (0, 1.5) means lags 0, 0.05, 0.10, ..., 1.45s into the past
    e.g. (-1.25, 0) means the predictor leads the response by 0 to 1.25s
    """
    T = len(signal)
    lag_start = round(kernel_win[0] / bin_width)
    lag_end = round(kernel_win[1] / bin_width)
    lags = np.arange(lag_start, lag_end)
    n_lags = len(lags)

    shifted = np.zeros((T, n_lags), dtype=np.float32)
    for i, lag in enumerate(lags):
        if lag >= 0:
            shifted[lag:, i] = signal[:T - lag]
        else:
            shifted[:T + lag, i] = signal[-lag:]

    return shifted, lags


#%% predictor builders

def _build_tf_predictor(session, t_ax):
    """log2(TF) at each 50ms bin during baseline period, 0 elsewhere"""
    T = len(t_ax)
    tf_signal = np.zeros(T, dtype=np.float32)

    for _, row in session.trials.iterrows():
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

        # only baseline period (before change)
        bl_end = row.get('Baseline_ON_fall', np.nan)
        if np.isnan(bl_end):
            bl_end = ft_20hz[-1]
        bl_mask = ft_20hz <= bl_end

        mask = (t_ax >= ft_20hz[0]) & (t_ax <= bl_end)
        bin_idx = np.where(mask)[0]
        if len(bin_idx) == 0:
            continue

        insert_idx = np.searchsorted(ft_20hz[bl_mask], t_ax[bin_idx], side='right') - 1
        insert_idx = np.clip(insert_idx, 0, bl_mask.sum() - 1)
        tf_signal[bin_idx] = tf_20hz[bl_mask][insert_idx]

    return tf_signal


def _build_event_predictor(event_times, t_ax):
    """place impulse at bin nearest each event time"""
    signal = np.zeros(len(t_ax), dtype=np.float32)
    for t in event_times:
        if np.isnan(t):
            continue
        idx = np.searchsorted(t_ax, t)
        idx = min(idx, len(t_ax) - 1)
        signal[idx] = 1.0
    return signal


def _build_trial_start_predictor(session, t_ax):
    return _build_event_predictor(session.trials['Baseline_ON_rise'].values, t_ax)


def _build_change_predictors(session, t_ax):
    """one impulse predictor per change magnitude, returns dict"""
    change_tfs = sorted(session.trials['Stim2TF'].dropna().unique())
    predictors = {}
    for ch_tf in change_tfs:
        mask = session.trials['Stim2TF'] == ch_tf
        times = session.trials.loc[mask, 'Change_ON_rise'].dropna().values
        predictors[f'change_tf{ch_tf}'] = _build_event_predictor(times, t_ax)
    return predictors


def _build_lick_prep_predictor(session, t_ax):
    """impulse at motion_onset for hits and FAs"""
    times = session.trials.loc[
        (session.trials['IsHit'] == 1) | (session.trials['IsFA'] == 1),
        'motion_onset'
    ].dropna().values
    return _build_event_predictor(times, t_ax)


def _build_lick_exec_predictor(session, t_ax):
    """impulse at first_lick"""
    times = session.trials['first_lick'].dropna().values
    return _build_event_predictor(times, t_ax)


def _build_air_puff_predictor(session, t_ax):
    return _build_event_predictor(session.trials['Air_puff_rise'].dropna().values, t_ax)


def _build_reward_predictor(session, t_ax):
    return _build_event_predictor(session.trials['Valve_L_rise'].dropna().values, t_ax)


def _build_abort_predictor(session, t_ax):
    """impulse at abort time (Baseline_ON_rise + rt_abort)"""
    abort_mask = session.trials['rt_abort'].notna()
    times = (session.trials.loc[abort_mask, 'Baseline_ON_rise']
             + session.trials.loc[abort_mask, 'rt_abort']).values
    return _build_event_predictor(times, t_ax)


def _build_time_ramp_predictor(session, t_ax):
    """ramp from 0 at each baseline onset, 0 outside trials"""
    signal = np.zeros(len(t_ax), dtype=np.float32)
    for _, row in session.trials.iterrows():
        bl_on = row['Baseline_ON_rise']
        bl_off = row['Baseline_ON_fall']
        if np.isnan(bl_on) or np.isnan(bl_off):
            continue
        tr_end = np.nanmax([bl_off, row.get('Change_ON_fall', np.nan)])
        mask = (t_ax >= bl_on) & (t_ax < tr_end)
        signal[mask] = t_ax[mask] - bl_on
    return signal


def _build_block_predictor(session, t_ax):
    """binary: 1 during late block trials, 0 during early block / outside trials"""
    signal = np.zeros(len(t_ax), dtype=np.float32)
    for _, row in session.trials.iterrows():
        if row['hazardblock'] != 'late':
            continue
        bl_on = row['Baseline_ON_rise']
        tr_end = np.nanmax([row['Baseline_ON_fall'],
                            row.get('Change_ON_fall', np.nan)])
        if np.isnan(bl_on) or np.isnan(tr_end):
            continue
        mask = (t_ax >= bl_on) & (t_ax < tr_end)
        signal[mask] = 1.0
    return signal


def _build_phase_predictors(session, t_ax):
    """one-hot grating phase, split by drift direction

    returns dict with 'phase_up' and 'phase_down', each (T, 12)
    """
    n_bins = GLM_OPTIONS['n_phase_bins']
    T = len(t_ax)
    phase_up = np.zeros((T, n_bins), dtype=np.float32)
    phase_down = np.zeros((T, n_bins), dtype=np.float32)

    for _, row in session.trials.iterrows():
        tf_raw = np.array(row['TF'])
        ft_raw = np.array(row['frame_time'])

        tf_seq = tf_raw[tf_raw.nonzero()]
        ft_seq = ft_raw[~np.isnan(ft_raw)]
        n = min(len(tf_seq), len(ft_seq))
        if n == 0:
            continue
        tf_seq, ft_seq = tf_seq[:n], ft_seq[:n]

        # cumulative phase in degrees
        dt = np.median(np.diff(ft_seq))
        phase = np.cumsum(tf_seq * dt) * 360.0
        phase_mod = phase % 360.0

        # assign to bins
        bin_idx = np.clip((phase_mod / (360.0 / n_bins)).astype(int), 0, n_bins - 1)

        # map to FR time axis
        fr_mask = (t_ax >= ft_seq[0]) & (t_ax <= ft_seq[-1])
        fr_idx = np.where(fr_mask)[0]
        if len(fr_idx) == 0:
            continue

        insert_idx = np.searchsorted(ft_seq, t_ax[fr_idx], side='right') - 1
        insert_idx = np.clip(insert_idx, 0, n - 1)

        target = phase_up if row['Stim1Ori'] == 90 else phase_down
        for fi, si in zip(fr_idx, insert_idx):
            target[fi, bin_idx[si]] = 1.0

    return {'phase_up': phase_up, 'phase_down': phase_down}


def _align_to_bins(signal, signal_times, t_ax):
    """align a continuous signal to the 50ms time axis via nearest-sample lookup"""
    idx = np.searchsorted(signal_times, t_ax, side='right') - 1
    idx = np.clip(idx, 0, len(signal) - 1)
    aligned = signal[idx].astype(np.float32)
    aligned[t_ax < signal_times[0]] = 0
    aligned[t_ax > signal_times[-1]] = 0
    return aligned


def _build_face_me_predictor(session, eye_cam_times, t_ax):
    mouth_me = np.array(session.move['video']['mouth_me'])
    n = min(len(mouth_me), len(eye_cam_times))
    return _align_to_bins(mouth_me[:n], eye_cam_times[:n], t_ax)


def _build_running_predictor(session, t_ax):
    speed = np.array(session.move['running']['speed'])
    time = np.array(session.move['running']['time'])
    return _align_to_bins(speed, time, t_ax)


def _build_pupil_predictor(session, eye_cam_times, t_ax):
    pupil = np.array(session.move['video']['pupil_area'])
    n = min(len(pupil), len(eye_cam_times))
    return _align_to_bins(pupil[:n], eye_cam_times[:n], t_ax)


#%% valid mask (exclude transition trials)

def _build_valid_mask(session, t_ax):
    """boolean mask (T,): True for bins inside non-transition trials"""
    mask = np.zeros(len(t_ax), dtype=bool)
    n_ignore = ANALYSIS_OPTIONS['ignore_first_trials_in_block']

    for _, row in session.trials.iterrows():
        if row['tr_in_block'] <= n_ignore:
            continue
        bl_on = row['Baseline_ON_rise']
        tr_end = np.nanmax([row['Baseline_ON_fall'],
                            row.get('Change_ON_fall', np.nan)])
        if np.isnan(bl_on) or np.isnan(tr_end):
            continue
        mask |= (t_ax >= bl_on) & (t_ax < tr_end)

    return mask


#%% design matrix assembly

def build_predictor_spec(session, has_eye_cam=True):
    """returns list of (name, kernel_window_or_None) tuples

    kernel_window is None for predictors that are already multi-column (phase, time ramp, block)
    """
    ops = GLM_OPTIONS
    change_tfs = sorted(session.trials['Stim2TF'].dropna().unique())

    spec = []
    spec.append(('tf', ops['kern_tf']))
    spec.append(('trial_start', ops['kern_trial_start']))
    spec.append(('time_ramp', None))
    spec.append(('block', None))
    for ch_tf in change_tfs:
        spec.append((f'change_tf{ch_tf}', ops['kern_change']))
    spec.append(('lick_prep', ops['kern_lick_prep']))
    spec.append(('lick_exec', ops['kern_lick_exec']))

    return spec


def build_design_matrix(session, t_ax, eye_cam_times):
    """build full (T x P) design matrix for one session

    returns design matrix, predictor col_map dict mapping name -> (col_slice, lags)
    """
    ops = GLM_OPTIONS
    bin_w = ops['bin_width']

    # build all base signals
    base_signals = {}
    base_signals['tf'] = _build_tf_predictor(session, t_ax)
    base_signals['trial_start'] = _build_trial_start_predictor(session, t_ax)
    base_signals['time_ramp'] = _build_time_ramp_predictor(session, t_ax)
    base_signals['block'] = _build_block_predictor(session, t_ax)

    for name, sig in _build_change_predictors(session, t_ax).items():
        base_signals[name] = sig

    base_signals['lick_prep'] = _build_lick_prep_predictor(session, t_ax)
    base_signals['lick_exec'] = _build_lick_exec_predictor(session, t_ax)

    # assemble: time-shift where needed, concatenate
    has_eye_cam = eye_cam_times is not None
    pred_spec = build_predictor_spec(session, has_eye_cam=has_eye_cam)
    blocks = []
    col_map = {}
    col_offset = 0

    for name, kernel_win in pred_spec:
        sig = base_signals[name]

        if kernel_win is not None:
            shifted, lags = _time_shift(sig, kernel_win, bin_w)
            n_cols = shifted.shape[1]
            blocks.append(shifted)
        elif sig.ndim == 2:
            # already multi-column (phase)
            n_cols = sig.shape[1]
            lags = np.arange(n_cols)
            blocks.append(sig)
        else:
            # single column, no shifting (time ramp, block)
            n_cols = 1
            lags = np.array([0])
            blocks.append(sig[:, None])

        col_map[name] = (slice(col_offset, col_offset + n_cols), lags)
        col_offset += n_cols

    X = np.concatenate(blocks, axis=1).astype(np.float32)

    return X, col_map


#%% save/load

def save_glm_inputs(sess_dir, counts, X, col_map, t_ax, valid_mask):
    """save prepped GLM data for one session"""
    sess_dir = Path(sess_dir)
    np.save(sess_dir / 'glm_counts.npy', counts)
    np.save(sess_dir / 'glm_design.npy', X)
    np.save(sess_dir / 'glm_t_ax.npy', t_ax)
    np.save(sess_dir / 'glm_valid.npy', valid_mask)
    with open(sess_dir / 'glm_spec.pkl', 'wb') as f:
        pickle.dump(col_map, f)


#%% local prep

def prepare_session(sess_dir, ceph_dir, overwrite=False):
    """build and save GLM inputs for one session"""
    sess_dir = Path(sess_dir)
    if not overwrite and (sess_dir / 'glm_design.npy').exists():
        print(f'  already prepped, skipping')
        return

    sess = Session.load(str(sess_dir / 'session.pkl'))
    fr_path = sess_dir / 'FR_matrix.parquet'
    if not fr_path.exists():
        print(f'  no FR matrix, skipping')
        return

    print(f'  building spike counts...')
    counts, t_ax = build_spike_counts(str(fr_path), sess.fr_stats)

    # load eye cam timestamps from ceph
    ceph_sess = Path(ceph_dir) / sess.animal / sess.name
    eye_cam_path = ceph_sess / 'daq_Eye_cam.csv'
    eye_cam_times = None
    if eye_cam_path.exists():
        eye_cam_times = pd.read_csv(eye_cam_path)['rise_t'].values

    print(f'  building design matrix...')
    X, col_map = build_design_matrix(sess, t_ax, eye_cam_times)
    valid_mask = _build_valid_mask(sess, t_ax)

    save_glm_inputs(str(sess_dir), counts, X, col_map, t_ax, valid_mask)
    print(f'  saved: counts {counts.shape}, X {X.shape}, '
          f'valid bins: {valid_mask.sum()}/{len(valid_mask)}')

    del counts, X, sess
    gc.collect()


def prepare_all_sessions(npx_dir=PATHS['npx_dir_local'],
                         ceph_dir=PATHS['npx_dir_ceph'],
                         overwrite: bool = False):
    """prep GLM inputs for all sessions"""
    for subj in sorted(os.listdir(npx_dir)):
        subj_dir = os.path.join(npx_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for sess in sorted(os.listdir(subj_dir)):
            sess_dir = os.path.join(subj_dir, sess)
            if not os.path.exists(os.path.join(sess_dir, 'session.pkl')):
                continue
            print(f'{subj}/{sess}')
            prepare_session(sess_dir, ceph_dir, overwrite)


def build_job_map(npx_dir=PATHS['npx_dir_local'], output_path=None):
    """build CSV mapping SLURM array index -> (session_dir, neuron_index, cluster_id)"""
    if output_path is None:
        output_path = os.path.join(npx_dir, 'glm_job_map.csv')

    rows = []
    for subj in sorted(os.listdir(npx_dir)):
        subj_dir = os.path.join(npx_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for sess in sorted(os.listdir(subj_dir)):
            sess_dir = os.path.join(subj_dir, sess)
            counts_path = os.path.join(sess_dir, 'glm_counts.npy')
            if not os.path.exists(counts_path):
                continue
            sess_data = Session.load(os.path.join(sess_dir, 'session.pkl'))
            n_neurons = len(sess_data.fr_stats)
            cluster_ids = sess_data.fr_stats.index.values
            for i in range(n_neurons):
                rows.append({
                    'job_idx': len(rows),
                    'sess_dir': sess_dir,
                    'neuron_idx': i,
                    'cluster_id': cluster_ids[i],
                    'animal': sess_data.animal,
                    'session': sess_data.name,
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f'Job map: {len(df)} neurons across {df["sess_dir"].nunique()} sessions')
    print(f'Saved to {output_path}')
    return df


#%% GLM fitting

def build_event_masks(session, t_ax):
    """build boolean masks marking relevant time windows for each lesion group

    returns dict: group_name -> (T,) bool
    """
    T = len(t_ax)
    masks = {}

    # TF: bins within -0.1 to 0.75s of each TF outlier pulse during baseline
    tf_mask = np.zeros(T, dtype=bool)
    if session.tf_pulses is not None and len(session.tf_pulses) > 0:
        for _, pulse in session.tf_pulses.iterrows():
            t = pulse['time']
            tf_mask |= (t_ax >= t - 0.1) & (t_ax <= t + 0.75)
    masks['tf'] = tf_mask

    # lick_prep: -1.25 to 0s before motion_onset
    lick_prep_mask = np.zeros(T, dtype=bool)
    for _, row in session.trials.iterrows():
        if row['IsHit'] != 1 and row['IsFA'] != 1:
            continue
        t = row.get('motion_onset', np.nan)
        if np.isnan(t):
            continue
        lick_prep_mask |= (t_ax >= t - 1.25) & (t_ax <= t)
    masks['lick_prep'] = lick_prep_mask

    # lick_exec: 0 to 0.5s after first_lick
    lick_exec_mask = np.zeros(T, dtype=bool)
    for _, row in session.trials.iterrows():
        t = row.get('first_lick', np.nan)
        if np.isnan(t):
            continue
        lick_exec_mask |= (t_ax >= t) & (t_ax <= t + 0.5)
    masks['lick_exec'] = lick_exec_mask

    return masks


#%% GLM fitting

def _fit_poisson_glm(X, y, max_iter=500, tol=1e-6):
    """fit poisson GLM via gradient descent (no regularisation)

    log(rate) = X @ w + b, loss = mean(exp(X @ w + b) - y * (X @ w + b))
    step size chosen by backtracking line search.
    """
    n, p = X.shape
    w = np.zeros(p, dtype=np.float64)
    b = np.float64(np.log(max(y.mean(), 1e-8)))

    def _loss(w, b):
        log_rate = np.clip(X @ w + b, -20, 20)
        return (np.exp(log_rate) - y * log_rate).mean()

    step = 1.0
    loss = _loss(w, b)

    for it in range(max_iter):
        log_rate = np.clip(X @ w + b, -20, 20)
        rate = np.exp(log_rate)
        residual = rate - y
        grad_w = X.T @ residual / n
        grad_b = residual.mean()

        # backtracking line search
        for _ in range(20):
            w_new = w - step * grad_w
            b_new = b - step * grad_b
            new_loss = _loss(w_new, b_new)
            if new_loss <= loss - 0.5 * step * (
                    np.dot(grad_w, w - w_new) + grad_b * (b - b_new)):
                break
            step *= 0.5
        else:
            break  # line search failed, stop

        w, b = w_new, b_new

        if abs(loss - new_loss) < tol:
            break
        loss = new_loss
        step = min(step * 1.5, 1.0)  # cautiously grow step

    return w, b


def _predict_glm(X, w, b):
    """predict counts from GLM weights"""
    log_rate = np.clip(X @ w + b, -20, 20)
    return np.exp(log_rate)


def fit_neuron(counts_1d, X, col_map, fold_ids,
               event_spec=None, ops=GLM_OPTIONS):
    """fit unregularised poisson GLM for one neuron

    1. fit per fold for cross-validated evaluation
    2. lesion analysis: zero out each predictor group, evaluate on
       bins where that group is non-zero
    3. refit on all data for kernel extraction

    fold_ids: (T,) int array from get_trial_fold_indices. bins with
        fold_id == -1 are excluded from all fitting and evaluation.
    event_spec: dict {kind: (bin_idx, signs, pre, post)} from
        build_event_spec. if given, per-fold PETHs are computed.
    """
    lesion_groups = ops['lesion_groups']
    group_names = list(lesion_groups.keys())
    n_folds = ops['n_folds']
    fit_kw = {k: ops[k] for k in ('max_iter', 'tol') if k in ops}

    # mask to valid bins only
    valid = fold_ids >= 0
    X_v = X[valid]
    y_v = counts_1d[valid].astype(np.float64)
    folds_v = fold_ids[valid]

    # derive evaluation masks from design matrix: bins where predictor group is non-zero
    group_masks = {}
    for gname, pred_list in lesion_groups.items():
        mask = np.zeros(X_v.shape[0], dtype=bool)
        for pred_name in pred_list:
            if pred_name in col_map:
                col_slice, _ = col_map[pred_name]
                mask |= np.any(X_v[:, col_slice] != 0, axis=1)
        group_masks[gname] = mask

    #%% fit per fold
    full_r = np.full(n_folds, np.nan)
    full_r_group = {g: np.full(n_folds, np.nan) for g in group_names}
    lesioned_r = {g: np.full(n_folds, np.nan) for g in group_names}
    fold_wb = {}

    # full-length CV predictions for PETH computation
    T = len(counts_1d)
    y_full_cv = np.full(T, np.nan)
    y_red_cv = {g: np.full(T, np.nan) for g in group_names}
    valid_T_idx = np.where(valid)[0]

    for k in range(n_folds):
        test_mask = folds_v == k
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_train, X_test, _, _ = normalise_design_matrix(
            X_v[train_mask], X_v[test_mask], col_map)
        y_train, y_test = y_v[train_mask], y_v[test_mask]

        if y_train.sum() == 0 or y_test.sum() == 0:
            continue

        w, b = _fit_poisson_glm(X_train, y_train, **fit_kw)
        fold_wb[k] = (w, b)

        y_pred = _predict_glm(X_test, w, b)
        full_r[k] = pearson_r(y_test, y_pred)

        test_T_idx = valid_T_idx[test_mask]
        y_full_cv[test_T_idx] = y_pred

        # per-group: refit without each predictor group
        for gname, pred_list in lesion_groups.items():
            win = group_masks[gname][test_mask]
            if win.sum() < 5:
                continue

            full_r_group[gname][k] = pearson_r(y_test[win], y_pred[win])

            X_train_red, col_map_red = reduce_design_matrix(
                X_v[train_mask], pred_list, col_map)
            X_test_red, _ = reduce_design_matrix(
                X_v[test_mask], pred_list, col_map)
            X_tr_n, X_te_n, _, _ = normalise_design_matrix(
                X_train_red, X_test_red, col_map_red)
            w_red, b_red = _fit_poisson_glm(X_tr_n, y_train, **fit_kw)
            y_les = _predict_glm(X_te_n, w_red, b_red)
            lesioned_r[gname][k] = pearson_r(y_test[win], y_les[win])
            y_red_cv[gname][test_T_idx] = y_les

    if not fold_wb:
        return None

    print(f'  mean r={np.nanmean(full_r):.4f}')

    #%% per-fold PETHs for paper-style classification
    peth_data = {}
    if event_spec is not None:
        counts_f = counts_1d.astype(np.float64)
        for kind, (bin_idx, signs, pre, post) in event_spec.items():
            if kind not in y_red_cv:
                continue
            n_bins = pre + post
            pa_fast = np.full((n_folds, n_bins), np.nan)
            pa_slow = np.full((n_folds, n_bins), np.nan)
            pf_fast = np.full((n_folds, n_bins), np.nan)
            pf_slow = np.full((n_folds, n_bins), np.nan)
            pr_fast = np.full((n_folds, n_bins), np.nan)
            pr_slow = np.full((n_folds, n_bins), np.nan)

            for k in range(n_folds):
                (pa_fast[k], pa_slow[k],
                 pf_fast[k], pf_slow[k],
                 pr_fast[k], pr_slow[k]) = fold_peths(
                    counts_f, y_full_cv, y_red_cv[kind],
                    bin_idx, signs, fold_ids, k, pre, post)

            peth_data[f'peth_{kind}_actual_fast'] = pa_fast
            peth_data[f'peth_{kind}_actual_slow'] = pa_slow
            peth_data[f'peth_{kind}_full_fast'] = pf_fast
            peth_data[f'peth_{kind}_full_slow'] = pf_slow
            peth_data[f'peth_{kind}_reduced_fast'] = pr_fast
            peth_data[f'peth_{kind}_reduced_slow'] = pr_slow

    #%% refit on all valid data for kernel extraction
    X_all, _, _, _ = normalise_design_matrix(X_v, X_v, col_map)
    w_final, b_final = _fit_poisson_glm(X_all, y_v, **fit_kw)

    #%% assemble results
    result = {
        'weights': w_final,
        'bias': np.array([b_final]),
        'fold_ids': fold_ids,
        'full_r': full_r,
    }
    for gname in group_names:
        result[f'full_r_group_{gname}'] = full_r_group[gname]
        result[f'lesioned_r_{gname}'] = lesioned_r[gname]
    result.update(peth_data)

    return result


def fit_neuron_from_disk(sess_dir, neuron_idx, ops=GLM_OPTIONS):
    """load prepped data, fit one neuron, save results to glm_unreg_results/"""
    sess_dir = Path(sess_dir)
    counts, X, col_map, t_ax, valid_mask = load_glm_inputs(str(sess_dir))

    sess = Session.load(str(sess_dir / 'session.pkl'))

    fold_ids = get_trial_fold_indices(
        sess.trials, t_ax, ops['n_folds'],
        seed=neuron_seed(str(sess_dir), neuron_idx),
        ignore_first_n=ANALYSIS_OPTIONS['ignore_first_trials_in_block'])

    print(f'Fitting neuron {neuron_idx} '
          f'({(fold_ids >= 0).sum()} valid bins, '
          f'{len(set(fold_ids[fold_ids >= 0]))} folds)')

    event_spec = build_event_spec(
        sess,
        kinds=['tf', 'lick_prep', 'lick_exec'],
        t_ax=t_ax,
        bin_width=ops['bin_width'],
        tf_sd_threshold=ops['tf_sd_threshold'])

    result = fit_neuron(counts[neuron_idx], X, col_map, fold_ids,
                        event_spec=event_spec, ops=ops)

    if result is None:
        print(f'  skipped (no valid predictions)')
        return

    results_dir = sess_dir / 'glm_unreg_results'
    results_dir.mkdir(exist_ok=True)
    np.savez(results_dir / f'neuron_{neuron_idx}.npz', **result)

    # save col_map once per session (same for all neurons)
    col_map_path = results_dir / 'col_map.pkl'
    if not col_map_path.exists():
        with open(col_map_path, 'wb') as f:
            pickle.dump(col_map, f)

    print(f'Saved to {results_dir / f"neuron_{neuron_idx}.npz"}')


# classification and kernel extraction moved to neuron_prediction.results
