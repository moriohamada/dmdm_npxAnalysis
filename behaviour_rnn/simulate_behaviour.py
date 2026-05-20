"""
forward the trained RNN on real per-trial data and compute predicted versions
of the three mouse behavioural observables (hazard, pulse-aligned lick prob,
lick-triggered TF) for side-by-side comparison with the empirical analyses
in behaviour/extraction.py.
"""
import numpy as np
import torch

from config import ANALYSIS_OPTIONS, BEHAVIOUR_RNN_OPS
from behaviour_rnn.train import build_tensors, _rt_kernel


#%% forward pass

@torch.no_grad()
def predict_p_lick(model, inputs, mask, pos_weight, batch_size=64, device='cpu'):
    """
    per-bin calibrated P(lick) for valid bins, NaN elsewhere.
    we undo the pos_weight rescaling by subtracting log(pos_weight) from
    the model's logits before sigmoid.
    """
    model.eval().to(device)
    n, max_t = inputs.shape[:2]
    p_lick = np.full((n, max_t), np.nan, dtype=np.float32)
    log_w = float(np.log(pos_weight))
    for i in range(0, n, batch_size):
        b = slice(i, min(i + batch_size, n))
        logit = model(inputs[b].to(device)).cpu().numpy() - log_w
        p_calib = 1.0 / (1.0 + np.exp(-logit))
        m = mask[b].cpu().numpy().astype(bool)
        chunk = np.full(logit.shape, np.nan, dtype=np.float32)
        chunk[m] = p_calib[m]
        p_lick[i:b.stop] = chunk
    return p_lick


def _apply_motor_delay(p_lick, rt_samples, dt, ops=BEHAVIOUR_RNN_OPS):
    """shift p_lick forward by the mouse's motor delay so simulated lick times
    line up with empirical ones. inverts the backward shift applied to the
    training target. point / gaussian -> shift by median (or mean) RT;
    rt_convolved -> forward-convolve with the (sum-normalised) RT kernel."""
    p_safe = np.where(np.isnan(p_lick), 0.0, p_lick).astype(np.float32)
    n, T = p_safe.shape
    kernel = ops.get('target_kernel', 'point')

    if kernel == 'rt_convolved' and len(rt_samples) > 0:
        h = _rt_kernel(rt_samples, dt)
        h = h / h.sum()
        out = np.zeros_like(p_safe)
        for i in range(n):
            out[i] = np.convolve(p_safe[i], h, mode='full')[:T]
        return out

    if len(rt_samples) == 0:
        return p_safe
    stat = ops.get('motor_shift_stat', 'median')
    rt_central = float(np.median(rt_samples) if stat == 'median' else np.mean(rt_samples))
    shift = int(round(rt_central / dt))
    if shift <= 0:
        return p_safe
    out = np.zeros_like(p_safe)
    out[:, shift:] = p_safe[:, :T - shift]
    return out


def _strip_iti(p_lick, tf_in, meta):
    """drop ITI bins so downstream analyses see trial-aligned indices"""
    iti = meta.get('iti_bins', 0)
    if iti == 0:
        return p_lick, tf_in, meta
    p_lick = p_lick[:, iti:]
    tf_in  = tf_in[:, iti:]
    meta = {**meta,
            'bl_end':       meta['bl_end'] - iti,
            'baseline_end': np.maximum(meta['baseline_end'] - iti, 0),
            'fa_bin':       np.where(meta['fa_bin']  >= 0, meta['fa_bin']  - iti, -1),
            'hit_bin':      np.where(meta['hit_bin'] >= 0, meta['hit_bin'] - iti, -1),
            'iti_bins':     0}
    return p_lick, tf_in, meta


def predict_for_df(model, df, pos_weight, motor_delay=True, **kwargs):
    """build_tensors + predict_p_lick. returns (p_lick, tf_in, meta) where
    tf_in is the TF input channel as a numpy array of shape (n_trials, T).
    motor_delay=True shifts p_lick forward to mirror the training-time target shift."""
    inputs, _, mask, meta = build_tensors(df)
    p_lick = predict_p_lick(model, inputs, mask, pos_weight, **kwargs)
    tf_in = inputs[:, :, 0].numpy()
    p_lick, tf_in, meta = _strip_iti(p_lick, tf_in, meta)
    if motor_delay:
        p_lick = _apply_motor_delay(p_lick, meta['rt_samples'], meta['dt'])
    return p_lick, tf_in, meta


def simulate_all(model, df, pos_weight, config=ANALYSIS_OPTIONS):
    """forward + run all mirror analyses for one mouse"""
    p_lick, tf_in, meta = predict_for_df(model, df, pos_weight)
    return dict(
        hazard  = predicted_hazard(p_lick, meta, config),
        pulse   = predicted_pulse_lick_prob(p_lick, tf_in, meta, config),
        kernel  = predicted_lick_kernel(p_lick, tf_in, meta, config),
        outcome = predicted_outcome_dist(p_lick, meta, config),
    )


#%% predicted hazard rate (mirrors calculate_el_hazard)

def predicted_hazard(p_lick, meta, config=ANALYSIS_OPTIONS):
    """predicted FA hazard per time-in-trial bin, per block. mirrors
    calculate_el_hazard: P(any lick in bin_size window | at risk at start of window)."""
    bin_size = config.get('hazard_bin_size', 0.5)
    bin_step = config.get('hazard_bin_step', 0.1)
    half = bin_size / 2
    max_t_s = config.get('hazard_max_time', 15.5)
    centres = np.arange(half, max_t_s - half + bin_step, bin_step)

    dt = meta['dt']
    baseline_end = meta.get('baseline_end', meta['bl_end'])
    blocks = meta['blocks']
    n_trials, max_t = p_lick.shape

    one_minus = 1.0 - np.where(np.isnan(p_lick), 0.0, p_lick)
    log_surv = np.log(np.clip(one_minus, 1e-9, 1.0))
    cumlog = np.concatenate([np.zeros((n_trials, 1)),
                             np.cumsum(log_surv, axis=1)], axis=1)

    out = {'binCentres': centres}
    for block_name, block_val in [('earlyBlock', 1.0), ('lateBlock', -1.0)]:
        sel = (blocks == block_val)
        if not sel.any():
            out[block_name] = np.full(len(centres), np.nan)
            continue
        bl_sub = baseline_end[sel]
        cumlog_sub = cumlog[sel]
        hazard = np.full(len(centres), np.nan)

        for c_i, c in enumerate(centres):
            lo_b = int(round((c - half) / dt))
            hi_b = min(int(round((c + half) / dt)), max_t)
            if hi_b <= lo_b:
                continue
            at_risk = bl_sub > lo_b
            if not at_risk.any():
                continue
            p_in_window = 1.0 - np.exp(cumlog_sub[at_risk, hi_b] - cumlog_sub[at_risk, lo_b])
            hazard[c_i] = float(np.mean(p_in_window))
        out[block_name] = hazard
    return out


#%% predicted pulse-aligned lick prob (mirrors calculate_pulse_lick_prob)

def predicted_pulse_lick_prob(p_lick, tf_inputs, meta, config=ANALYSIS_OPTIONS):
    """
    predicted P(lick in lick_win after a TF sample), binned by TF value, by block
    and sliding time-in-trial window. matches the 1D part of calculate_pulse_lick_prob.
    analytic: at (trial, t), P(lick in [t+lo, t+hi]) = 1 - prod_{tau in window} (1 - p_tau)
    """
    bin_centres = np.array(config.get('tf_pulse_bin_centres',
                                      np.arange(-1.0, 1.0, 0.05)))
    half_w   = config.get('tf_pulse_bin_width', 0.1) / 2
    lick_win = config.get('tf_pulse_lick_win', [0.2, 1.5])
    time_win = config.get('tf_pulse_time_win', 3)
    time_step = config.get('tf_pulse_time_step', 1)

    dt = meta['dt']
    baseline_end = meta.get('baseline_end', meta['bl_end'])
    blocks = meta['blocks']
    n_trials, max_t = p_lick.shape
    t_bin = np.arange(max_t) * dt

    # per-bin survival-weighted probability of lick within window
    win_lo_b = int(round(lick_win[0] / dt))
    win_hi_b = int(round(lick_win[1] / dt))
    one_minus = 1.0 - np.where(np.isnan(p_lick), 0.0, p_lick)
    log_survival = np.log(np.clip(one_minus, 1e-9, 1.0))
    cumlog = np.concatenate([np.zeros((n_trials, 1)), np.cumsum(log_survival, axis=1)], axis=1)

    # restrict to bins inside baseline only; post-change bins distort TF binning
    valid = ~np.isnan(p_lick) & (np.arange(max_t)[None] < baseline_end[:, None])
    tf = tf_inputs.numpy() if hasattr(tf_inputs, 'numpy') else tf_inputs

    # iterate over time windows and blocks
    max_time = t_bin[baseline_end - 1].max() if len(baseline_end) else 0
    time_starts = np.arange(0, max_time - time_win + time_step, time_step)

    results = {'binCentres': bin_centres, 'time_starts': time_starts, 'time_win': time_win}
    for block_name, block_val in [('early', 1.0), ('late', -1.0)]:
        sel = (blocks == block_val)
        if not sel.any():
            continue
        cumlog_sub = cumlog[sel]
        lo = np.clip(np.arange(max_t) + win_lo_b, 0, max_t)
        hi = np.clip(np.arange(max_t) + win_hi_b, 0, max_t)
        log_s = cumlog_sub[:, hi] - cumlog_sub[:, lo]
        win_p_sub = 1.0 - np.exp(log_s)
        tf_sub = tf[sel]
        valid_sub = valid[sel]

        for t_start in time_starts:
            t_end = t_start + time_win
            t_mask = (t_bin >= t_start) & (t_bin < t_end)
            cond_name = f'{block_name}Block_{t_start:.0f}-{t_end:.0f}s'

            tf_vals = tf_sub[:, t_mask][valid_sub[:, t_mask]]
            win_vals = win_p_sub[:, t_mask][valid_sub[:, t_mask]]

            lick_prob = np.full(len(bin_centres), np.nan)
            n_stim = np.zeros(len(bin_centres), dtype=int)
            for b, bc in enumerate(bin_centres):
                in_bin = (tf_vals >= bc - half_w) & (tf_vals < bc + half_w)
                n_stim[b] = in_bin.sum()
                if in_bin.any():
                    lick_prob[b] = win_vals[in_bin].mean()
            results[cond_name] = {'lickProb': lick_prob, 'n': n_stim}
    return results


#%% predicted lick-triggered TF (mirrors extract_elts + calculate_elta)

def predicted_lick_kernel(p_lick, tf_inputs, meta, config=ANALYSIS_OPTIONS):
    """
    analytic predicted lick-triggered TF (kernel).
    contribution of bin t to the kernel = P(first lick at bin t) = p_t * survival_{t-1}.
    kernel[l] = sum_t P(first lick at t) * TF[t-l]  /  sum_t P(first lick at t)
    computed separately per condition (earlyBlock_early, lateBlock_early, lateBlock_late).
    """
    dt = meta['dt']
    baseline_end = meta.get('baseline_end', meta['bl_end'])
    blocks = meta['blocks']
    n_trials, max_t = p_lick.shape
    n_pre = config.get('n_pre_lick_samples', 40)
    t_split = config.get('tr_split_time', 8)
    t_early = config.get('ignore_trial_start', 2)

    tf = tf_inputs.numpy() if hasattr(tf_inputs, 'numpy') else tf_inputs

    # zero p_lick past the baseline period so post-change bins don't contribute
    # to the FA kernel
    bl_mask = np.arange(max_t)[None] < baseline_end[:, None]
    p_safe = np.where(np.isnan(p_lick) | ~bl_mask, 0.0, p_lick)
    log_survival = np.log(np.clip(1.0 - p_safe, 1e-9, 1.0))
    cumlog_prev = np.concatenate([np.zeros((n_trials, 1)),
                                  np.cumsum(log_survival, axis=1)[:, :-1]], axis=1)
    p_first = p_safe * np.exp(cumlog_prev)  # P(first lick at bin t), baseline only

    t_bin = np.arange(max_t) * dt

    conds = {
        'earlyBlock_early': (blocks == 1.0,  (t_bin > t_early) & (t_bin <= t_split)),
        'lateBlock_early':  (blocks == -1.0, (t_bin > t_early) & (t_bin <= t_split)),
        'lateBlock_late':   (blocks == -1.0, t_bin > t_split),
    }

    lags = np.arange(n_pre)            # lag in bins, 0 = lick bin
    out = {}
    for cond_name, (block_sel, time_mask) in conds.items():
        p_sub  = p_first[block_sel]
        tf_sub = tf[block_sel]
        n_b    = p_sub.shape[0]
        if n_b == 0:
            out[cond_name] = np.full(n_pre, np.nan)
            continue

        # weighted sum of TF[t-l] across (trial, t) where t is in time_mask
        kernel = np.full(n_pre, np.nan)
        for l in lags:
            t_indices = np.arange(max_t)
            t_valid = time_mask & (t_indices - l >= 0)
            if not t_valid.any():
                continue
            t_idx = np.where(t_valid)[0]
            tf_lagged = tf_sub[:, t_idx - l]
            weights   = p_sub[:, t_idx]
            num = (weights * tf_lagged).sum()
            den = weights.sum()
            if den > 0:
                kernel[l] = num / den
        # kernel currently indexed by lag (0 = lick bin). reverse so index 0 = earliest
        out[cond_name] = kernel[::-1]
    return out


#%% predicted outcome distribution (Hit / FA / Miss)

def predicted_outcome_dist(p_lick, meta, config=ANALYSIS_OPTIONS):
    """per-block model vs mouse proportions of FA / Hit / Miss.
    FA is unconditional. Hit/Miss are conditional on having seen the change
    (no FA before change_bin), so P(hit|saw change) + P(miss|saw change) = 1."""
    dt = meta['dt']
    rw_bins = int(round(config.get('response_window', 2.15) / dt))
    df = meta['df']
    n_trials, max_t = p_lick.shape

    p_safe = np.where(np.isnan(p_lick), 0.0, p_lick)
    log_surv = np.log(np.clip(1.0 - p_safe, 1e-9, 1.0))
    cumlog_prev = np.concatenate([np.zeros((n_trials, 1)),
                                  np.cumsum(log_surv, axis=1)[:, :-1]], axis=1)
    p_first = p_safe * np.exp(cumlog_prev)

    stim_t = df['stimT'].to_numpy(dtype=float)
    has_change = np.isfinite(stim_t)
    change_bin = np.where(has_change,
                          np.clip(np.ceil(stim_t / dt).astype(int) - 1, 0, max_t - 1),
                          max_t)

    P_fa = np.zeros(n_trials)
    P_hit_uncond = np.zeros(n_trials)
    P_saw_change = np.zeros(n_trials)
    for i in range(n_trials):
        cb = int(change_bin[i])
        P_fa[i] = p_first[i, :cb].sum() if cb > 0 else 0.0
        if has_change[i]:
            P_saw_change[i] = float(np.exp(cumlog_prev[i, cb])) if cb < max_t else \
                              float(np.exp(cumlog_prev[i, -1] + log_surv[i, -1]))
            if cb < max_t:
                P_hit_uncond[i] = p_first[i, cb:min(cb + rw_bins, max_t)].sum()

    # condition hit/miss on no FA before change
    with np.errstate(invalid='ignore', divide='ignore'):
        P_hit_cond  = np.where(has_change & (P_saw_change > 1e-9),
                               P_hit_uncond / P_saw_change, np.nan)
    P_miss_cond = np.where(has_change, 1.0 - P_hit_cond, np.nan)

    blocks = meta['blocks']
    is_fa_m  = df['IsFA'].to_numpy(dtype=bool)
    is_hit_m = df['IsHit'].to_numpy(dtype=bool)
    is_miss_m = df['IsMiss'].to_numpy(dtype=bool)
    saw_change_m = is_hit_m | is_miss_m  # mouse: any non-FA trial with a change

    out = {}
    for block_name, block_val in [('early', 1.0), ('late', -1.0)]:
        sel = blocks == block_val
        if not sel.any():
            continue
        sel_sc = sel & saw_change_m
        out[block_name] = dict(
            model = dict(
                fa   = float(np.nanmean(P_fa[sel])),
                hit  = float(np.nanmean(P_hit_cond[sel])),
                miss = float(np.nanmean(P_miss_cond[sel])),
            ),
            mouse = dict(
                fa   = float(is_fa_m[sel].mean()),
                hit  = float(is_hit_m[sel_sc].mean()) if sel_sc.any() else np.nan,
                miss = float(is_miss_m[sel_sc].mean()) if sel_sc.any() else np.nan,
            ),
            n = int(sel.sum()),
        )
    return out
