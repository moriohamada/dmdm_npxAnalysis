"""
train a small vanilla RNN per mouse to predict per-bin P(lick) from
(TF, time-in-trial, block). supervised on baseline-only data.
also runs the cohort-level train + simulate + cache loop (train_rnns_all_subj).
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from config import ANALYSIS_OPTIONS, BEHAVIOUR_RNN_OPS, PATHS
from behaviour.integrator import clean_baseline_trials, precompute_tf


RNN_DATA_DIR  = os.path.join(PATHS['npx_dir_local'], 'behaviour_rnn')
MODELS_SUBDIR = 'models'
SIM_NAME      = 'rnn_sim.pkl'


#%% dataset

FAST_CHANGE_TFS = (2.0, 4.0)


def _motor_rt_samples(df):
    """hit reaction times (s) at fast TF changes - dominated by motor delay"""
    hits = df['IsHit'].astype(bool) & df['Stim2TF'].isin(FAST_CHANGE_TFS)
    return df.loc[hits, 'rt_RT'].dropna().to_numpy(dtype=float)


def _rt_kernel(rt_samples, dt, max_bins=None):
    """peak-normalised histogram of motor RT in dt-sized bins.
    returns h where h[k] >= 0 is the (normalised) frequency of RT == k bins."""
    rt = rt_samples[rt_samples >= 0]
    if len(rt) == 0:
        return np.array([1.0], dtype=np.float32)
    offsets = np.round(rt / dt).astype(int)
    max_off = int(offsets.max()) + 1
    if max_bins is not None:
        max_off = min(max_off, max_bins)
    h = np.bincount(np.clip(offsets, 0, max_off - 1), minlength=max_off).astype(np.float32)
    return h / h.max()


def _apply_kernel(target, event_bin, idx, rt_samples, dt, ops):
    """add the configured kernel into target rows in idx, centred at event_bin[idx]"""
    if len(idx) == 0:
        return target
    max_t = target.shape[1]
    kernel = ops.get('target_kernel', 'point')

    if kernel == 'rt_convolved':
        h = _rt_kernel(rt_samples, dt)
        n_h = len(h)
        for i in idx:
            offsets = event_bin[i] - np.arange(max_t)
            valid = (offsets >= 0) & (offsets < n_h)
            target[i, valid] = np.maximum(target[i, valid], h[offsets[valid]])
        return target

    if len(rt_samples) > 0:
        stat = ops.get('motor_shift_stat', 'median')
        rt_central = float(np.median(rt_samples) if stat == 'median' else np.mean(rt_samples))
    else:
        rt_central = 0.0
    shift = int(round(rt_central / dt))
    centres = np.clip(event_bin - shift, 0, max_t - 1)

    sigma = float(ops.get('target_sigma_bins', 0))
    if kernel == 'point' or sigma <= 0:
        target[idx, centres[idx]] = np.maximum(target[idx, centres[idx]], 1.0)
        return target

    t_grid = np.arange(max_t)[None]
    c = centres[idx][:, None]
    gauss = np.exp(-0.5 * ((t_grid - c) / sigma) ** 2).astype(np.float32)
    target[idx] = np.maximum(target[idx], gauss)
    return target


def _build_target(fa_bin, is_fa, hit_bin, is_hit, n_trials, max_t,
                  rt_samples, dt, ops):
    """target = decision kernel at FA bin for FA trials AND at hit bin for hit trials.
    miss trials get zero (model should not lick). kernel set by ops['target_kernel']."""
    target = np.zeros((n_trials, max_t), dtype=np.float32)
    _apply_kernel(target, fa_bin, np.where(is_fa)[0], rt_samples, dt, ops)
    _apply_kernel(target, hit_bin, np.where(is_hit)[0], rt_samples, dt, ops)
    return target


def build_tensors(df, config=ANALYSIS_OPTIONS, ops=BEHAVIOUR_RNN_OPS):
    """
    build (inputs, target, mask, meta) from one mouse's trial df. trial inputs span
    baseline and the response window (mode='full_trial'), so the model sees the
    change pulse on hit/miss trials.

    inputs: (N, T, 6) float32 -- tf in octaves, time/16, block (+/-1),
                                 prev_licked (0/1), prev_lick_t / 20, prev_rewarded (0/1)
    target: (N, T) float32 -- decision kernel at the FA bin (FA trials) and at the
                              hit bin (hit trials), peak 1.0. miss trials are zero.
                              kernel set by ops['target_kernel']; the centre is
                              shifted earlier by the mouse's motor delay.
    mask:   (N, T) float32 -- 1 for every bin up to bl_end (ITI included).
                              non-FA trials extend bl_end through stimT + response_window.
    """
    # exclude aborts/refs
    df = df[~df['trialoutcome'].isin(['abort', 'Ref'])]
    df = clean_baseline_trials(df)
    tf_mat, bl_end, fa_time, dt = precompute_tf(
        df, fa_extend_bins=ops['fa_extend_bins'], mode='full_trial')
    n_trials, T_trial = tf_mat.shape

    # ITI prepended so the RNN settles from h=0 before trial start. mask=1
    # and target=0 over the ITI to teach "no lick during ITI". time channel
    # is negative across the ITI for a smooth clock.
    iti_bins = int(round(ops.get('iti_seconds', 0.0) / dt))
    max_t = T_trial + iti_bins

    blocks = df['hazardblock'].map({'early': 1.0, 'late': -1.0}).to_numpy(dtype=np.float32)

    is_fa  = df['IsFA'].to_numpy(dtype=bool)
    is_hit = df['IsHit'].to_numpy(dtype=bool)
    is_miss = df['IsMiss'].to_numpy(dtype=bool)

    fa_t_safe = np.nan_to_num(fa_time, nan=0.0)
    fa_bin_rel = np.clip(np.ceil(fa_t_safe / dt).astype(int) - 1, 0, T_trial - 1)
    fa_bin = np.where(is_fa, fa_bin_rel + iti_bins, -1)

    stim_t = df['stimT'].to_numpy(dtype=float)
    rt_rt  = df['rt_RT'].to_numpy(dtype=float)
    hit_t  = np.where(is_hit, stim_t + rt_rt, 0.0)
    hit_bin_rel = np.clip(np.ceil(hit_t / dt).astype(int) - 1, 0, T_trial - 1)
    hit_bin = np.where(is_hit, hit_bin_rel + iti_bins, -1)

    # previous-trial features (within session; first trial of each session = 0)
    sess = df['sessionID'].to_numpy()
    same_sess = np.concatenate([[False], sess[1:] == sess[:-1]])
    licked_self = (is_fa | is_hit).astype(np.float32)
    lick_t_self = np.where(is_fa, fa_t_safe,
                           np.where(is_hit, stim_t + np.nan_to_num(rt_rt, nan=0.0),
                                    0.0)).astype(np.float32)
    rewarded_self = is_hit.astype(np.float32)

    def _shift(x):
        out = np.concatenate([[0], x[:-1]])
        return np.where(same_sess, out, 0.0).astype(np.float32)

    prev_licked   = _shift(licked_self)
    prev_lick_t   = _shift(lick_t_self) / 20.0
    prev_rewarded = _shift(rewarded_self)

    tf_in = np.zeros((n_trials, max_t), dtype=np.float32)
    tf_in[:, iti_bins:] = np.nan_to_num(tf_mat, nan=0.0).astype(np.float32)
    t_grid = ((np.arange(max_t) - iti_bins) * dt).astype(np.float32)
    time_in = np.broadcast_to(t_grid / 16.0, (n_trials, max_t))
    block_in = np.broadcast_to(blocks[:, None], (n_trials, max_t))
    pl_in  = np.broadcast_to(prev_licked[:, None],   (n_trials, max_t))
    plt_in = np.broadcast_to(prev_lick_t[:, None],   (n_trials, max_t))
    pr_in  = np.broadcast_to(prev_rewarded[:, None], (n_trials, max_t))
    inputs = np.stack([tf_in, time_in, block_in, pl_in, plt_in, pr_in], axis=-1)

    # baseline_end = where the baseline period actually ends (used by hazard /
    # pulse / kernel analyses). distinct from bl_end which in 'full_trial' mode
    # extends through the response window for non-FA trials.
    baseline_end_t = np.where(is_fa, fa_t_safe, stim_t)
    baseline_end_bin = np.minimum(
        np.ceil(baseline_end_t / dt).astype(int) + iti_bins, max_t)
    bl_end = bl_end + iti_bins

    rt_samples = _motor_rt_samples(df)
    target = _build_target(fa_bin, is_fa, hit_bin, is_hit,
                           n_trials, max_t, rt_samples, dt, ops)

    # loss applies over every bin up to bl_end (ITI included; target=0 there)
    t_idx = np.arange(max_t)[None]
    mask = (t_idx < bl_end[:, None]).astype(np.float32)

    meta = dict(
        df         = df.reset_index(drop=True),
        bl_end     = bl_end,
        baseline_end = baseline_end_bin,
        fa_time    = fa_time,
        fa_bin     = fa_bin,
        hit_bin    = hit_bin,
        is_fa      = is_fa,
        is_hit     = is_hit,
        is_miss    = is_miss,
        dt         = dt,
        blocks     = blocks,
        rt_samples = rt_samples,
        iti_bins   = iti_bins,
    )
    return (
        torch.from_numpy(inputs),
        torch.from_numpy(target),
        torch.from_numpy(mask),
        meta,
    )


def split_train_val(meta, val_frac=0.2, seed=0):
    """stratified by (block, outcome) so val mirrors train across hit/miss/FA"""
    rng = np.random.default_rng(seed)
    outcome = np.where(meta['is_fa'], 'fa',
                       np.where(meta['is_hit'], 'hit',
                                np.where(meta['is_miss'], 'miss', 'other')))
    blk = meta['blocks']
    train_idx, val_idx = [], []
    for b in [1.0, -1.0]:
        for o in ['fa', 'hit', 'miss']:
            ids = np.where((blk == b) & (outcome == o))[0]
            rng.shuffle(ids)
            cut = int(len(ids) * val_frac)
            val_idx.extend(ids[:cut])
            train_idx.extend(ids[cut:])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


#%% model

class BehaviourRNN(nn.Module):
    """vanilla relu RNN with linear sigmoid read-out (logits returned)"""

    def __init__(self, n_hidden=8, n_in=6):
        super().__init__()
        self.n_hidden = n_hidden
        self.rnn = nn.RNN(input_size=n_in, hidden_size=n_hidden,
                          nonlinearity='relu', batch_first=True)
        self.readout = nn.Linear(n_hidden, 1)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.readout(h).squeeze(-1)


#%% training

def compute_pos_weight(target, mask):
    n_pos = (target * mask).sum()
    n_neg = ((1.0 - target) * mask).sum()
    return float((n_neg / n_pos).item())


def masked_bce(logit, target, mask, pos_weight):
    """bce-with-logits, mean over valid bins per trial, then over trials"""
    pw = torch.tensor(pos_weight, device=logit.device)
    per_bin = nn.functional.binary_cross_entropy_with_logits(
        logit, target, pos_weight=pw, reduction='none')
    per_trial = (per_bin * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return per_trial.mean()


def train_one(inputs, target, mask, train_idx, val_idx,
              n_hidden=8, ops=BEHAVIOUR_RNN_OPS, device='cpu', verbose=True):
    """one RNN fit with a fixed n_hidden. returns (model, history, pos_weight)"""
    torch.manual_seed(ops['seed'])
    model = BehaviourRNN(n_hidden=n_hidden, n_in=inputs.shape[-1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=ops['lr'],
                             weight_decay=ops['weight_decay'])

    inputs = inputs.to(device)
    target = target.to(device)
    mask   = mask.to(device)

    pos_weight = compute_pos_weight(target[train_idx], mask[train_idx]) * ops['lick_weight']

    hist = {'train': [], 'val': []}
    best_val, best_state, stagn = np.inf, None, 0
    rng = np.random.default_rng(ops['seed'])

    for epoch in range(ops['max_epochs']):
        model.train()
        perm = rng.permutation(train_idx)
        losses = []
        for i in range(0, len(perm), ops['batch_size']):
            b = perm[i:i + ops['batch_size']]
            logit = model(inputs[b])
            loss = masked_bce(logit, target[b], mask[b], pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ops['grad_clip'])
            opt.step()
            losses.append(loss.item())
        tr_loss = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            v_logit = model(inputs[val_idx])
            v_loss  = masked_bce(v_logit, target[val_idx], mask[val_idx],
                                 pos_weight).item()

        hist['train'].append(tr_loss)
        hist['val'].append(v_loss)

        improved = v_loss < best_val - 1e-5
        if improved:
            best_val   = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stagn = 0
        else:
            stagn += 1

        if verbose and epoch % 10 == 0:
            tag = ' *' if improved else ''
            print(f'  epoch {epoch:3d} | train {tr_loss:.4f} | val {v_loss:.4f}{tag}')

        if epoch >= ops['min_epochs'] and stagn >= ops['patience']:
            if verbose:
                print(f'  early stop @ epoch {epoch}, best val {best_val:.4f}')
            break

    model.load_state_dict(best_state)
    return model, hist, pos_weight


def fit_subj(df, n_hidden=None, ops=BEHAVIOUR_RNN_OPS, device='cpu', verbose=True,
             subj=None, sweep_plot_dir=None, real_cached=None):
    """
    fit one mouse. n_hidden=None -> sweep ops['n_hidden_sweep'] and pick
    the smallest n_h within 1% of the best val loss.

    if subj and sweep_plot_dir are provided, real-vs-RNN mirror plots are written
    to sweep_plot_dir/<subj>/n_h_<n_h>/ after each sweep entry.
    """
    inputs, target, mask, meta = build_tensors(df)
    train_idx, val_idx = split_train_val(meta, val_frac=ops['val_frac'], seed=ops['seed'])

    if n_hidden is not None:
        model, hist, pw = train_one(
            inputs, target, mask, train_idx, val_idx,
            n_hidden=n_hidden, ops=ops, device=device, verbose=verbose)
        if subj and sweep_plot_dir:
            _plot_sweep_entry(model, df, pw, subj, n_hidden,
                              sweep_plot_dir, real_cached)
        return dict(
            model=model, history=hist, pos_weight=pw, n_hidden=n_hidden,
            meta=meta, train_idx=train_idx, val_idx=val_idx,
        )

    sweep = {}
    for n_h in ops['n_hidden_sweep']:
        if verbose:
            print(f'--- n_hidden = {n_h} ---')
        m, h, pw = train_one(
            inputs, target, mask, train_idx, val_idx,
            n_hidden=n_h, ops=ops, device=device, verbose=verbose)
        sweep[n_h] = dict(model=m, history=h, pos_weight=pw, best_val=min(h['val']))
        if subj and sweep_plot_dir:
            _plot_sweep_entry(m, df, pw, subj, n_h, sweep_plot_dir, real_cached)

    best_loss = min(s['best_val'] for s in sweep.values())
    chosen = min(n_h for n_h, s in sweep.items() if s['best_val'] <= best_loss * 1.01)
    if verbose:
        print(f'chosen n_hidden = {chosen} (best val {sweep[chosen]["best_val"]:.4f})')

    return dict(
        sweep=sweep, n_hidden=chosen, meta=meta,
        train_idx=train_idx, val_idx=val_idx,
        model=sweep[chosen]['model'], history=sweep[chosen]['history'],
        pos_weight=sweep[chosen]['pos_weight'],
    )


#%% save / load

def save_model(result, path):
    """save state dict + training meta. meta dict (with df) is not saved here."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = dict(
        state_dict = result['model'].state_dict(),
        history    = result['history'],
        pos_weight = result['pos_weight'],
        n_hidden   = result['n_hidden'],
        n_in       = result['model'].rnn.input_size,
        train_idx  = result['train_idx'],
        val_idx    = result['val_idx'],
    )
    if 'sweep' in result:
        obj['sweep_val'] = {n_h: s['best_val'] for n_h, s in result['sweep'].items()}
    torch.save(obj, path)
    print(f'saved to {path}')


def load_model(path):
    obj = torch.load(path, weights_only=False)
    model = BehaviourRNN(n_hidden=obj['n_hidden'], n_in=obj.get('n_in', 6))
    model.load_state_dict(obj['state_dict'])
    model.eval()
    return model, obj


def _plot_sweep_entry(model, df, pos_weight, subj, n_h, plot_dir, real_cached):
    """write real-vs-RNN mirror plots + example trials + outcome dist for one mouse"""
    from behaviour_rnn.simulate_behaviour import simulate_all
    from behaviour_rnn.plotting import (plot_subject_mirrors, plot_example_trials,
                                        plot_outcome_dist_single)
    from utils.figures import save_fig

    out_dir = Path(plot_dir) / subj / f'n_h_{n_h}'
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = simulate_all(model, df, pos_weight)
    plot_subject_mirrors(subj, sim, out_dir, real_cached=real_cached)
    save_fig(plot_example_trials(model, df, pos_weight), str(out_dir / 'examples'))
    save_fig(plot_outcome_dist_single(sim['outcome']), str(out_dir / 'outcome'))


#%% cohort-level: train every mouse, cache simulated behavioural mirrors

def _model_path(subj, data_dir=RNN_DATA_DIR):
    return Path(data_dir) / MODELS_SUBDIR / f'rnn_{subj}.pt'


def train_rnns_all_subj(dfs, overwrite=False, data_dir=RNN_DATA_DIR,
                        sweep_plot_dir=None, **fit_kwargs):
    """
    train an RNN per mouse, save weights, compute and save the simulated
    behavioural mirrors (hazard, pulse-lick, kernel) for the whole cohort.
    skips mice whose model already exists unless overwrite=True.

    sweep_plot_dir, if given, receives per-mouse per-n_hidden diagnostic plots
    (one subdir per (mouse, n_h)) generated as each sweep entry finishes.
    """
    from behaviour_rnn.simulate_behaviour import simulate_all
    data_dir = Path(data_dir)
    (data_dir / MODELS_SUBDIR).mkdir(parents=True, exist_ok=True)

    real_cached = None
    if sweep_plot_dir is not None:
        from behaviour_rnn.plotting import load_real_observables
        real_cached = load_real_observables()

    for subj, df in dfs.items():
        path = _model_path(subj, data_dir)
        if path.exists() and not overwrite:
            print(f'{subj}: cached, skipping')
            continue
        print(f'\n=== training RNN for {subj} ===')
        result = fit_subj(df, verbose=True, subj=subj,
                          sweep_plot_dir=sweep_plot_dir, real_cached=real_cached,
                          **fit_kwargs)
        save_model(result, path)

    sim_path = data_dir / SIM_NAME
    if sim_path.exists() and not overwrite:
        print(f'simulated mirrors cached at {sim_path}')
        return

    sim = {'hazard': {}, 'pulse': {}, 'kernel': {}, 'outcome': {}}
    for subj, df in dfs.items():
        path = _model_path(subj, data_dir)
        if not path.exists():
            continue
        model, obj = load_model(path)
        out = simulate_all(model, df, obj['pos_weight'])
        for k in sim:
            sim[k][subj] = out[k]

    with open(sim_path, 'wb') as f:
        pickle.dump(sim, f)
    print(f'saved simulated mirrors to {sim_path}')


def load_rnn_results(data_dir=RNN_DATA_DIR):
    """return (rnn_meta_by_subj, sim_by_observable)"""
    data_dir = Path(data_dir)
    rnn = {}
    for path in (data_dir / MODELS_SUBDIR).glob('rnn_*.pt'):
        subj = path.stem.replace('rnn_', '')
        _, obj = load_model(path)
        rnn[subj] = obj
    with open(data_dir / SIM_NAME, 'rb') as f:
        sim = pickle.load(f)
    return rnn, sim
