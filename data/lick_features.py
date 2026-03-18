"""
Feature extraction + target generation for lick prediction model, generating per-bin
feature vectors and lick targets
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from data.session import Session
from config import LICK_PRED_OPS


def _get_trial_stimulus(row):
    """Extract aligned log2 TF and frame times at 20Hz for one trial."""
    tf_raw = np.array(row['TF'])
    ft_raw = np.array(row['frame_time'])
    tf_seq = tf_raw[tf_raw.nonzero()]
    ch_fr = round(row['stimT'] * 60)

    tf_20hz = np.log2(tf_seq[:ch_fr:3])
    ft_20hz = ft_raw[~np.isnan(ft_raw)][:ch_fr:3]

    n = min(len(tf_20hz), len(ft_20hz))
    return tf_20hz[:n], ft_20hz[:n]


def _compute_motion_lick_delay(trials):
    """Median delay between motion_onset and first_lick for trials where both exist."""
    mo = trials['motion_onset']
    fl = trials['first_lick']
    both = mo.notna() & fl.notna() & (trials['IsHit'] | trials['IsFA'])
    delays = fl[both] - mo[both]
    if len(delays) == 0:
        return 0.0
    return delays.median()


def _trial_lick_time(row, motion_lick_delay=0.0):
    """Use motion_onset if available, otherwise infer from first_lick."""
    if not (row['IsHit'] or row['IsFA']):
        return np.nan
    if not pd.isna(row['motion_onset']):
        return row['motion_onset']
    if not pd.isna(row.get('first_lick', np.nan)):
        return row['first_lick'] - motion_lick_delay
    if row['IsFA']:
        return row['Baseline_ON_rise'] + row['rt_FA']
    return np.nan


def _trial_outcome_code(row):
    if row['IsHit']:
        return 'hit'
    elif row['IsMiss']:
        return 'miss'
    elif row['IsFA']:
        return 'fa'
    elif row['IsAbort']:
        return 'abort'
    else:
        # 'Ref' trials - lick too fast after change, treat as FA
        return 'fa'


def _trial_event_time(row, motion_lick_delay=0.0):
    """Time of outcome event relative to baseline onset."""
    bl_on = row['Baseline_ON_rise']
    if row['IsHit'] or row['IsFA']:
        lick = _trial_lick_time(row, motion_lick_delay)
        if not np.isnan(lick):
            return lick - bl_on
    if row['IsMiss']:
        return row['stimT'] + 2.15
    if row['IsAbort']:
        return row['rt_abort']
    return 0.0


def _build_tf_history(tf_20hz, n_bins, n_hist):
    """build (n_bins, n_hist) sliding window of TF history, zero-padded at start."""
    padded = np.concatenate([np.zeros(n_hist - 1), tf_20hz[:n_bins]])
    idx = np.arange(n_hist)[None, :] + np.arange(n_bins)[:, None]
    return padded[idx]


OUTCOME_MAP = {'hit': 0, 'miss': 1, 'fa': 2, 'abort': 3}

# feature column layout - defines which columns in X correspond to which features
N_TF_HIST = 40
FEATURE_COLS = {
    'stimulus':          list(range(0, N_TF_HIST)),
    'time_in_trial':     [N_TF_HIST],
    'block':             [N_TF_HIST + 1],
    'prev_outcome':      list(range(N_TF_HIST + 2, N_TF_HIST + 6)),
    'prev_event_time':   [N_TF_HIST + 6],
    'time_since_reward': [N_TF_HIST + 7],
    'trial_num':         [N_TF_HIST + 8],
}
CONTINUOUS_COLS = (FEATURE_COLS['stimulus'] + FEATURE_COLS['time_in_trial']
                   + FEATURE_COLS['prev_event_time'] + FEATURE_COLS['time_since_reward']
                   + FEATURE_COLS['trial_num'])
N_FEATURES = N_TF_HIST + 9


def build_trial_features(row, prev_outcome, prev_event_time,
                         time_since_reward, trial_num,
                         motion_lick_delay=0.0, ops=LICK_PRED_OPS):
    """
    Build feature matrix X and target vector y for one trial.

    Returns (X, y, n_bins) or (None, None, 0) if trial should be skipped.
    """
    n_hist = ops['tf_history_bins']
    sigma = ops['lick_sigma_bins']
    extend = ops['lick_extend_bins']
    resp_win = ops['response_window']

    if row['IsAbort']:
        return None, None, 0

    # exclude Ref trials (lick too fast after change to be a real detection)
    if row.get('trialoutcome') == 'Ref':
        return None, None, 0

    bl_on = row['Baseline_ON_rise']
    lick_t = _trial_lick_time(row, motion_lick_delay)
    has_lick = not np.isnan(lick_t)

    tf_20hz, ft_20hz = _get_trial_stimulus(row)
    if len(tf_20hz) == 0:
        return None, None, 0

    # determine number of bins
    if has_lick:
        trial_end = lick_t - bl_on + extend * ops['bin_width']
    elif row['IsHit'] or row['IsMiss']:
        trial_end = row['stimT'] + resp_win
    else:
        trial_end = len(tf_20hz) * ops['bin_width']

    n_bins = max(1, int(trial_end / ops['bin_width']))
    if n_bins < 1:
        return None, None, 0

    # pad TF if trial extends past available stimulus (post-lick extension)
    n_stim = len(tf_20hz)
    if n_bins > n_stim:
        tf_20hz = np.concatenate([tf_20hz, np.zeros(n_bins - n_stim)])

    tf_history = _build_tf_history(tf_20hz, n_bins, n_hist)
    time_in_trial = np.arange(n_bins) * ops['bin_width']
    block = 1.0 if row['hazardblock'] == 'late' else 0.0

    outcome_onehot = np.zeros(4)
    if prev_outcome in OUTCOME_MAP:
        outcome_onehot[OUTCOME_MAP[prev_outcome]] = 1.0

    # get 'static' features constant over all bins in a trial
    static = np.concatenate([[block], outcome_onehot,
                              [prev_event_time, time_since_reward, trial_num]])

    X = np.column_stack([
        tf_history,
        time_in_trial,
        np.tile(static, (n_bins, 1)),
    ])

    # soft target - smooth lick time with gaussian
    y = np.zeros(n_bins)
    if has_lick:
        lick_bin = int((lick_t - bl_on) / ops['bin_width'])
        lick_bin = np.clip(lick_bin, 0, n_bins - 1)
        y[lick_bin] = 1.0
        y = gaussian_filter1d(y, sigma=sigma)
        if y.max() > 0:
            y = y / y.max()

    return X, y, n_bins


def build_session_features(session, ops=LICK_PRED_OPS):
    """
    build feature matrices and targets for all non-abort trials in a session.

    Returns:
        X: (total_bins, 49) feature matrix
        y: (total_bins,) target vector (lick/no lick, smoothed)
        trial_ids: (total_bins,) trial index for each bin
    """
    Xs, ys, ids = [], [], []
    motion_lick_delay = _compute_motion_lick_delay(session.trials)

    prev_outcome = None
    prev_event_time = 0.0
    time_since_reward = ops['max_time_since_reward']
    last_reward_time = None

    for tr, row in session.trials.iterrows():
        if last_reward_time is not None:
            time_since_reward = row['Baseline_ON_rise'] - last_reward_time
        else:
            time_since_reward = ops['max_time_since_reward']

        X, y_tr, n_bins = build_trial_features(
            row, prev_outcome, prev_event_time,
            time_since_reward, float(tr), motion_lick_delay, ops)

        if X is not None and n_bins > 0:
            Xs.append(X)
            ys.append(y_tr)
            ids.append(np.full(n_bins, tr))

        # update history for next trial (including aborts)
        prev_outcome = _trial_outcome_code(row)
        prev_event_time = _trial_event_time(row, motion_lick_delay)
        if row['IsHit']:
            last_reward_time = _trial_lick_time(row, motion_lick_delay)

    if not Xs:
        return np.empty((0, 49)), np.empty(0), np.empty(0, dtype=int)

    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ids)


def build_mouse_features(session_paths, ops=LICK_PRED_OPS):
    """
    build features for all sessions of one mouse.

    returns
        sessions_data: list of (X, y, trial_ids) tuples per sess
        session_names: list of session names
    """
    sessions_data = []
    session_names = []

    for path in session_paths:
        sess = Session.load(path)
        X, y, trial_ids = build_session_features(sess, ops)
        if len(X) > 0:
            sessions_data.append((X, y, trial_ids))
            session_names.append(sess.name)

    return sessions_data, session_names
