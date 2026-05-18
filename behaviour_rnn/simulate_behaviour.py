"""
forward the trained RNN on real per-trial data and compute predicted versions
of the three mouse behavioural observables (hazard, pulse-aligned lick prob,
lick-triggered TF) for side-by-side comparison with the empirical analyses
in behaviour/extraction.py.
"""
import numpy as np
import torch

from config import ANALYSIS_OPTIONS
from behaviour_rnn.train import build_tensors


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


def predict_for_df(model, df, pos_weight, **kwargs):
    """build_tensors + predict_p_lick. returns (p_lick, tf_in, meta) where
    tf_in is the TF input channel as a numpy array of shape (n_trials, T)."""
    inputs, _, mask, meta = build_tensors(df)
    p_lick = predict_p_lick(model, inputs, mask, pos_weight, **kwargs)
    tf_in = inputs[:, :, 0].numpy()
    return p_lick, tf_in, meta


def simulate_all(model, df, pos_weight, config=ANALYSIS_OPTIONS):
    """forward + run all three mirror analyses for one mouse"""
    p_lick, tf_in, meta = predict_for_df(model, df, pos_weight)
    return dict(
        hazard = predicted_hazard(p_lick, meta, config),
        pulse  = predicted_pulse_lick_prob(p_lick, tf_in, meta, config),
        kernel = predicted_lick_kernel(p_lick, tf_in, meta, config),
    )


#%% predicted hazard rate (mirrors calculate_el_hazard)

def predicted_hazard(p_lick, meta, config=ANALYSIS_OPTIONS):
    """
    predicted FA hazard per time-in-trial bin, per block.
    p_lick is already the model's per-bin P(lick | at risk), so we average
    it across trials at-risk at each time-in-trial bin.
    """
    bin_size = config.get('hazard_bin_size', 0.5)
    bin_step = config.get('hazard_bin_step', 0.1)
    half = bin_size / 2
    max_t_s = 15.5
    centres = np.arange(half, max_t_s - half + bin_step, bin_step)

    dt = meta['dt']
    bl_end = meta['bl_end']
    blocks = meta['blocks']
    t_bin = np.arange(p_lick.shape[1]) * dt

    out = {'binCentres': centres}
    for block_name, block_val in [('earlyBlock', 1.0), ('lateBlock', -1.0)]:
        sel = (blocks == block_val)
        if not sel.any():
            out[block_name] = np.full(len(centres), np.nan)
            continue
        p_sub = p_lick[sel]              # (n_trials, T)
        bl_sub = bl_end[sel]
        at_risk_n = np.zeros(len(centres))
        hazard_sum = np.zeros(len(centres))
        for c_i, c in enumerate(centres):
            lo, hi = c - half, c + half
            bin_mask = (t_bin >= lo) & (t_bin < hi)
            if not bin_mask.any():
                continue
            at_risk = (bl_sub[:, None] > np.where(bin_mask)[0][None]).all(axis=1)
            # average predicted hazard across at-risk trials and across bins in window
            p_window = p_sub[at_risk][:, bin_mask]
            valid = ~np.isnan(p_window)
            if valid.sum() == 0:
                continue
            hazard_sum[c_i] = np.nanmean(p_window) * bin_size / dt  # scale to per-window prob
            at_risk_n[c_i] = at_risk.sum()
        out[block_name] = np.where(at_risk_n > 0, hazard_sum, np.nan)
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
    bl_end = meta['bl_end']
    blocks = meta['blocks']
    n_trials, max_t = p_lick.shape
    t_bin = np.arange(max_t) * dt

    # per-bin survival-weighted probability of lick within window
    win_lo_b = int(round(lick_win[0] / dt))
    win_hi_b = int(round(lick_win[1] / dt))
    one_minus = 1.0 - np.where(np.isnan(p_lick), 0.0, p_lick)
    log_survival = np.log(np.clip(one_minus, 1e-9, 1.0))
    cumlog = np.concatenate([np.zeros((n_trials, 1)), np.cumsum(log_survival, axis=1)], axis=1)

    # masked TF values: only valid (not NaN) per-bin entries
    valid = ~np.isnan(p_lick)
    tf = tf_inputs.numpy() if hasattr(tf_inputs, 'numpy') else tf_inputs

    # iterate over time windows and blocks
    max_time = t_bin[bl_end - 1].max() if len(bl_end) else 0
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
    bl_end = meta['bl_end']
    blocks = meta['blocks']
    n_trials, max_t = p_lick.shape
    n_pre = config.get('n_pre_lick_samples', 40)
    t_split = config.get('tr_split_time', 8)
    t_early = config.get('ignore_trial_start', 2)

    tf = tf_inputs.numpy() if hasattr(tf_inputs, 'numpy') else tf_inputs

    p_safe = np.where(np.isnan(p_lick), 0.0, p_lick)
    log_survival = np.log(np.clip(1.0 - p_safe, 1e-9, 1.0))
    cumlog_prev = np.concatenate([np.zeros((n_trials, 1)),
                                  np.cumsum(log_survival, axis=1)[:, :-1]], axis=1)
    p_first = p_safe * np.exp(cumlog_prev)  # P(first lick at bin t)

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
