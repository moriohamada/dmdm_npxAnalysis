"""
leaky integrator model fit per animal per block on baseline-only data.
fits (tau, gain, threshold) by feature matching against three observables:
    P(lick|TF)         -> gain (sensitivity)
    baseline FA hazard -> threshold (criterion)
    lick-triggered TF  -> tau (integration time)
the three-feature objective breaks the gain x threshold degeneracy that arises
when fitting only to FA times.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed

from config import ANALYSIS_OPTIONS
from behaviour.extraction import strip_and_convert_tf


#%% grid + feature targets

# tau: dense log-spacing around the paper's behavioural estimate (~0.27s);
#      sentinels [0] (no integration) and [inf] (perfect integration) bracket the search.
# gain: factor-25 around 1 in log octaves (sensitivity scaling of log2(TF) input).
# threshold: log-spaced; the integrator state SS-SD is ~gain*0.43 octaves at tau=0.25s,
#            so thresholds 0.2-8 span ~0.5x to ~20x the noise floor at gain=1.
SEARCH_PARAMS = dict(
    tau       = np.concatenate([[0], np.logspace(np.log10(0.05), np.log10(3), 14), [np.inf]]),
    gain      = np.logspace(np.log10(0.2), np.log10(5), 10),
    threshold = np.logspace(np.log10(0.2), np.log10(8), 14),
)

LICK_WIN    = (0.2, 1.0)                       # s; window for P(lick|TF)
TF_BINS     = np.arange(-0.8, 0.85, 0.1)       # octaves
HAZARD_BINS = np.arange(2.0, 16.0, 0.5)        # s; centres
HAZARD_HALF = 0.25                             # s; half-width
KERNEL_LAGS = np.arange(-1.0, 0.0 + 0.05, 0.05)  # s before FA


#%% preprocessing

def clean_baseline_trials(df, min_t=ANALYSIS_OPTIONS['ignore_trial_start']):
    """keep FA trials with rt_FA past min_t and non-FA trials whose baseline lasted past min_t"""
    fa_ok = df['IsFA'] & (df['rt_FA'] > min_t)
    non_fa_ok = ~df['IsFA'] & (df['stimT'] > min_t)
    return df[fa_ok | non_fa_ok].reset_index(drop=True)


def precompute_tf(df, config=ANALYSIS_OPTIONS, fa_extend_bins=0, mode='baseline'):
    """per-trial TF in log2 octaves, subsampled, with `bl_end` marking the per-trial
    cutoff.

    mode='baseline': bl_end = FA bin (FA trials) or change-onset bin (non-FA), with
        optional fa_extend_bins past the FA. tf_mat is NaN'd past bl_end.
    mode='full_trial': non-FA trials extend through the response window so the model
        sees the change pulse and learns to (not) lick. FA path unchanged."""
    frame_step  = config['tf_sample_step']
    sample_rate = config['tf_sample_rate']
    dt = 1.0 / sample_rate

    # strip grey-screen and convert before subsampling so times align with rt_FA/stimT
    tf_seqs = [strip_and_convert_tf(tf)[::frame_step] for tf in df['stim_TF']]
    seq_lens = np.array([len(t) for t in tf_seqs], dtype=int)
    max_t = int(seq_lens.max()) if len(seq_lens) else 0
    n = len(df)

    tf_mat = np.full((n, max_t), np.nan)
    for i, t in enumerate(tf_seqs):
        tf_mat[i, :len(t)] = t

    is_fa  = df['IsFA'].to_numpy(dtype=bool)
    fa_t   = df['rt_FA'].to_numpy(dtype=float)
    stim_t = df['stimT'].to_numpy(dtype=float)

    bl_end_t = np.where(is_fa, fa_t, stim_t)
    bl_end = np.minimum(np.ceil(bl_end_t * sample_rate).astype(int), seq_lens)
    if fa_extend_bins:
        bl_end = np.where(is_fa, np.minimum(bl_end + fa_extend_bins, seq_lens), bl_end)
    if mode == 'full_trial':
        rw_bins = int(round(config['response_window'] * sample_rate))
        non_fa_end = np.minimum(np.ceil(stim_t * sample_rate).astype(int) + rw_bins,
                                seq_lens)
        bl_end = np.where(is_fa, bl_end, non_fa_end)
    past = np.arange(max_t)[None] >= bl_end[:, None]
    tf_mat[past] = np.nan

    fa_time = np.where(is_fa, fa_t, np.nan)
    return tf_mat, bl_end, fa_time, dt


#%% simulation

def simulate_integrator(tf_mat, bl_end, tau, gain, threshold, dt,
                        min_t=ANALYSIS_OPTIONS['ignore_trial_start']):
    """deterministic leaky integrator: v_{t+1} = decay * v_t + gain * tf_t.
    returns predicted FA time per trial, nan if no threshold crossing within baseline"""
    n_trials, max_t = tf_mat.shape
    if tau == 0:
        decay = 0.0
    elif np.isinf(tau):
        decay = 1.0
    else:
        decay = float(np.exp(-dt / tau))
    min_sample = int(min_t / dt)

    v = np.zeros(n_trials)
    pred_fa = np.full(n_trials, np.nan)
    running = np.ones(n_trials, dtype=bool)

    for t in range(max_t):
        active = running & (t < bl_end)
        if not active.any():
            break
        x = np.where(np.isnan(tf_mat[:, t]), 0.0, tf_mat[:, t])
        v = np.where(active, decay * v + gain * x, v)
        if t >= min_sample:
            crossed = active & (v >= threshold)
            pred_fa[crossed] = t * dt
            running[crossed] = False
    return pred_fa


#%% features

def _p_lick_tf(tf_mat, bl_end, fa_time, dt,
               min_t=ANALYSIS_OPTIONS['ignore_trial_start']):
    """P(FA within LICK_WIN s after each TF sample), binned by TF value"""
    n, max_t = tf_mat.shape
    t_grid = np.arange(max_t) * dt
    win_lo, win_hi = LICK_WIN

    lick_t  = np.where(np.isfinite(fa_time), fa_time, np.inf)
    horizon = bl_end * dt - win_hi
    valid = (t_grid[None] >= min_t) & (t_grid[None] <= horizon[:, None]) & ~np.isnan(tf_mat)
    diff = lick_t[:, None] - t_grid[None]
    licked = (diff >= win_lo) & (diff <= win_hi)

    tfv = tf_mat[valid]
    lk  = licked[valid].astype(float)
    p = np.full(len(TF_BINS) - 1, np.nan)
    for b in range(len(TF_BINS) - 1):
        m = (tfv >= TF_BINS[b]) & (tfv < TF_BINS[b + 1])
        if m.sum() > 20:
            p[b] = lk[m].mean()
    return p


def _hazard(bl_end, fa_time, dt):
    """FA hazard rate per time-in-trial bin (life-table)"""
    bl_t = bl_end * dt
    fa_finite = np.isfinite(fa_time)
    haz = np.full(len(HAZARD_BINS), np.nan)
    for b, c in enumerate(HAZARD_BINS):
        lo, hi = c - HAZARD_HALF, c + HAZARD_HALF
        at_risk = (bl_t > lo).sum()
        if at_risk == 0:
            continue
        n_fa = (fa_finite & (fa_time >= lo) & (fa_time < hi)).sum()
        haz[b] = n_fa / at_risk
    return haz


def _kernel(tf_mat, bl_end, fa_time, dt):
    """mean TF (octaves) at each lag before FA, vectorised over lags and FA trials"""
    sample_rate = 1.0 / dt
    fa_idx = np.where(np.isfinite(fa_time))[0]
    n_lags = len(KERNEL_LAGS)
    if len(fa_idx) == 0:
        return np.full(n_lags, np.nan)

    fa_samples = (fa_time[fa_idx] * sample_rate).astype(int)
    lag_samples = np.round(KERNEL_LAGS * sample_rate).astype(int)
    idx = fa_samples[:, None] + lag_samples[None, :]
    valid = (idx >= 0) & (idx < bl_end[fa_idx, None])

    vals = np.full(idx.shape, np.nan)
    rows = np.broadcast_to(fa_idx[:, None], idx.shape)
    vals[valid] = tf_mat[rows[valid], idx[valid]]

    n_per_lag = np.isfinite(vals).sum(axis=0)
    kernel = np.where(n_per_lag >= 10, np.nanmean(vals, axis=0), np.nan)
    return kernel


def compute_features(tf_mat, bl_end, fa_time, dt):
    return dict(
        p_lick_tf = _p_lick_tf(tf_mat, bl_end, fa_time, dt),
        hazard    = _hazard(bl_end, fa_time, dt),
        kernel    = _kernel(tf_mat, bl_end, fa_time, dt),
    )


#%% fitting

def feature_loss(real, synth, weights=None):
    """variance-normalised mse across the three observables, averaged"""
    weights = weights or dict(p_lick_tf=1.0, hazard=1.0, kernel=1.0)
    parts, ws = [], []
    for k, w in weights.items():
        r, s = real[k], synth[k]
        valid = np.isfinite(r) & np.isfinite(s)
        if valid.sum() < 3:
            continue
        scale = np.nanvar(r[valid]) + 1e-9
        parts.append(np.mean((r[valid] - s[valid]) ** 2) / scale)
        ws.append(w)
    if not parts:
        return np.inf
    return float(np.average(parts, weights=ws))


def grid_search(df_block, search_params=SEARCH_PARAMS, weights=None,
                n_jobs=-1, verbose=True):
    """fit (tau, gain, threshold) for one (subject, block) via 3D grid"""
    tf_mat, bl_end, fa_time, dt = precompute_tf(df_block)
    real = compute_features(tf_mat, bl_end, fa_time, dt)

    keys = list(search_params)
    combos = list(product(*search_params.values()))
    if verbose:
        print(f'grid: {len(combos)} combos over {keys}')

    def _one(vals):
        params = dict(zip(keys, vals))
        pred = simulate_integrator(tf_mat, bl_end, dt=dt, **params)
        synth = compute_features(tf_mat, bl_end, pred, dt)
        return feature_loss(real, synth, weights), params, pred, synth

    res = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_one)(v) for v in combos)
    losses = np.array([r[0] for r in res])
    best = int(np.argmin(losses))

    if verbose:
        print(f'best: {res[best][1]} (loss={losses[best]:.4f})')

    return dict(
        params           = [r[1] for r in res],
        losses           = losses,
        real_feats       = real,
        best_params      = res[best][1],
        best_pred_fa     = res[best][2],
        best_synth_feats = res[best][3],
    )


def fit_per_subj(dfs, save_path=None, overwrite=False, min_trials=50, **kwargs):
    """fit per animal per block, baseline-only data"""
    if save_path and Path(save_path).exists() and not overwrite:
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    results = {}
    for subj, df in dfs.items():
        df_clean = clean_baseline_trials(df)
        results[subj] = {}
        for block in ['early', 'late']:
            df_b = df_clean[df_clean['hazardblock'] == block].reset_index(drop=True)
            if len(df_b) < min_trials:
                print(f'{subj} | {block}: only {len(df_b)} trials, skipping')
                continue
            print(f'{subj} | {block} | n={len(df_b)}')
            results[subj][block] = grid_search(df_b, **kwargs)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f'saved to {save_path}')
    return results


def load_results(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
