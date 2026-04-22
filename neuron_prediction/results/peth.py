"""
event-triggered PETH helpers for paper-style unit classification

follows Khilkevich & Lohse et al. 2024 methods:
- define fast/slow TF pulses from session.tf_pulses at 0.5 s.d. threshold
- compute per-fold PETHs of actual spikes, full-model prediction,
  reduced-model prediction around events
- feeds into criterion-1 (mean Pearson's r > 0.2) and criterion-2
  (residual-prediction t-test) in classify.py
"""
import numpy as np


# log2(TF) std during baseline (from stimulus: σ = 0.25 octaves)
TF_BASELINE_LOG2_SD = 0.25

# PETH windows (seconds). kept at current code values as requested
EVENT_WINDOWS = {
    'tf':        (-0.1, 0.75),
    'lick_prep': (-1.25, 0.0),
    'lick_exec': (0.0, 0.5),
}


def get_event_times(session, kind, tf_sd_threshold=0.5):
    """return (times, signs) for one event kind

    signs: +1 / -1 for fast / slow TF pulses; +1 for all lick events
    """
    if kind == 'tf':
        if session.tf_pulses is None or len(session.tf_pulses) == 0:
            return np.array([]), np.array([], dtype=int)
        tf_log2 = session.tf_pulses['tf'].values
        thresh = tf_sd_threshold * TF_BASELINE_LOG2_SD
        keep = np.abs(tf_log2) >= thresh
        times = session.tf_pulses['time'].values[keep]
        signs = np.sign(tf_log2[keep]).astype(int)
        return times, signs

    if kind in ('lick_prep', 'lick_exec'):
        from neuron_prediction.data import lick_times
        times = lick_times(session)
        return times, np.ones(len(times), dtype=int)

    raise ValueError(f'unknown event kind: {kind}')


def window_bins(window_s, bin_width):
    """(pre_bins, post_bins) for a (t_start, t_end) window in seconds

    n_bins = pre + post; window spans [-pre*bw, +post*bw) around event bin
    """
    pre = int(np.round(-window_s[0] / bin_width))
    post = int(np.round(window_s[1] / bin_width))
    return pre, post


def build_event_spec(session, kinds, t_ax, bin_width, tf_sd_threshold=0.5):
    """pre-compute per-kind (bin_idx, signs, pre, post)

    bin_idx: bin index of each event's centre in t_ax
    signs: +1 / -1 per event
    pre, post: window width in bins
    """
    spec = {}
    for kind in kinds:
        times, signs = get_event_times(session, kind,
                                       tf_sd_threshold=tf_sd_threshold)
        if len(times) == 0:
            pre, post = window_bins(EVENT_WINDOWS[kind], bin_width)
            spec[kind] = (np.array([], dtype=int),
                          np.array([], dtype=int), pre, post)
            continue
        bin_idx = np.searchsorted(t_ax, times)
        bin_idx = np.clip(bin_idx, 0, len(t_ax) - 1)
        pre, post = window_bins(EVENT_WINDOWS[kind], bin_width)
        spec[kind] = (bin_idx, signs, pre, post)
    return spec


def fold_peths(counts, y_full_cv, y_red_cv,
               bin_idx, signs, fold_ids, fold_k, pre, post):
    """compute per-fold PETHs for fast/slow events in fold_k

    counts: (T,) actual spike counts
    y_full_cv: (T,) full-model CV predictions, NaN on train/invalid bins
    y_red_cv:  (T,) reduced-model CV predictions, NaN on train/invalid bins
    bin_idx: (n_events,) event centre bin indices
    signs: (n_events,) +1 fast / -1 slow / +1 for non-tf kinds
    fold_ids: (T,) fold index per bin (>= 0 valid)
    fold_k: target fold
    pre, post: window bins

    returns 6 (n_bins,) arrays:
        actual_fast, actual_slow, full_fast, full_slow, red_fast, red_slow
    any can be NaN if no events of that sign are valid in this fold
    """
    n_bins = pre + post
    nan_out = np.full(n_bins, np.nan)

    # keep only events whose centre is in fold_k, and whose whole window
    # lies in valid bins (fold_ids >= 0). trial-aligned folds mean this
    # is usually automatic but we check defensively
    in_fold = fold_ids[bin_idx] == fold_k
    lo = bin_idx - pre
    hi = bin_idx + post
    in_bounds = (lo >= 0) & (hi <= len(fold_ids))
    keep_init = in_fold & in_bounds

    keep = []
    for i, (b, ok) in enumerate(zip(bin_idx, keep_init)):
        if not ok:
            keep.append(False)
            continue
        if np.any(fold_ids[b - pre:b + post] < 0):
            keep.append(False)
        else:
            keep.append(True)
    keep = np.array(keep, dtype=bool)

    if not keep.any():
        return (nan_out, nan_out, nan_out, nan_out, nan_out, nan_out)

    kept_bins = bin_idx[keep]
    kept_signs = signs[keep]

    fast_bins = kept_bins[kept_signs > 0]
    slow_bins = kept_bins[kept_signs < 0]

    def peth(signal, bins):
        if len(bins) == 0:
            return nan_out.copy()
        stack = np.stack([signal[b - pre:b + post] for b in bins])
        return np.nanmean(stack, axis=0)

    return (peth(counts, fast_bins), peth(counts, slow_bins),
            peth(y_full_cv, fast_bins), peth(y_full_cv, slow_bins),
            peth(y_red_cv, fast_bins), peth(y_red_cv, slow_bins))
