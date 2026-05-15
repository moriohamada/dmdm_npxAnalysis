"""
train a small vanilla RNN per mouse to predict per-bin P(lick) from
(TF, time-in-trial, block). supervised on baseline-only data.
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from config import ANALYSIS_OPTIONS, BEHAVIOUR_RNN_OPS
from behaviour.integrator import clean_baseline_trials, precompute_tf


#%% dataset

def build_tensors(df, config=ANALYSIS_OPTIONS):
    """
    build (inputs, target, mask, meta) from one mouse's trial df.
    inputs: (N, T, 3) float32 -- tf in octaves, time/16, block (+/-1)
    target: (N, T) float32 -- 1 at FA bin for FA trials, else 0
    mask:   (N, T) float32 -- 1 for bins in [ignore_trial_start, bl_end-1]
    """
    df = clean_baseline_trials(df)
    tf_mat, bl_end, fa_time, dt = precompute_tf(df)
    n_trials, max_t = tf_mat.shape

    blocks = df['hazardblock'].map({'early': 1.0, 'late': -1.0}).to_numpy(dtype=np.float32)

    tf_in = np.nan_to_num(tf_mat, nan=0.0).astype(np.float32)
    t_grid = (np.arange(max_t) * dt).astype(np.float32)
    time_in = np.broadcast_to(t_grid / 16.0, (n_trials, max_t))
    block_in = np.broadcast_to(blocks[:, None], (n_trials, max_t))
    inputs = np.stack([tf_in, time_in, block_in], axis=-1)

    target = np.zeros((n_trials, max_t), dtype=np.float32)
    is_fa = df['IsFA'].to_numpy(dtype=bool)
    fa_bin = (bl_end - 1).clip(min=0)
    target[np.where(is_fa)[0], fa_bin[is_fa]] = 1.0

    min_bin = int(round(config['ignore_trial_start'] / dt))
    t_idx = np.arange(max_t)[None]
    mask = ((t_idx >= min_bin) & (t_idx < bl_end[:, None])).astype(np.float32)

    meta = dict(
        df       = df.reset_index(drop=True),
        bl_end   = bl_end,
        fa_time  = fa_time,
        dt       = dt,
        blocks   = blocks,
        min_bin  = min_bin,
    )
    return (
        torch.from_numpy(inputs),
        torch.from_numpy(target),
        torch.from_numpy(mask),
        meta,
    )


def split_train_val(meta, val_frac=0.2, seed=0):
    """stratified by (block, IsFA) so val mirrors train"""
    rng = np.random.default_rng(seed)
    is_fa = meta['df']['IsFA'].to_numpy(dtype=bool)
    blk   = meta['blocks']
    train_idx, val_idx = [], []
    for b in [1.0, -1.0]:
        for fa in [True, False]:
            ids = np.where((blk == b) & (is_fa == fa))[0]
            rng.shuffle(ids)
            cut = int(len(ids) * val_frac)
            val_idx.extend(ids[:cut])
            train_idx.extend(ids[cut:])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


#%% model

class BehaviourRNN(nn.Module):
    """vanilla tanh RNN with linear sigmoid read-out (logits returned)"""

    def __init__(self, n_hidden=8, n_in=3):
        super().__init__()
        self.n_hidden = n_hidden
        self.rnn = nn.RNN(input_size=n_in, hidden_size=n_hidden,
                          nonlinearity='tanh', batch_first=True)
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
    model = BehaviourRNN(n_hidden=n_hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=ops['lr'],
                             weight_decay=ops['weight_decay'])

    inputs = inputs.to(device)
    target = target.to(device)
    mask   = mask.to(device)

    pos_weight = compute_pos_weight(target[train_idx], mask[train_idx])

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

        if verbose and (epoch % 10 == 0 or improved):
            tag = ' *' if improved else ''
            print(f'  epoch {epoch:3d} | train {tr_loss:.4f} | val {v_loss:.4f}{tag}')

        if stagn >= ops['patience']:
            if verbose:
                print(f'  early stop @ epoch {epoch}, best val {best_val:.4f}')
            break

    model.load_state_dict(best_state)
    return model, hist, pos_weight


def fit_subj(df, n_hidden=None, ops=BEHAVIOUR_RNN_OPS, device='cpu', verbose=True):
    """
    fit one mouse. n_hidden=None -> sweep ops['n_hidden_sweep'] and pick
    the smallest n_h within 1% of the best val loss.
    """
    inputs, target, mask, meta = build_tensors(df)
    train_idx, val_idx = split_train_val(meta, val_frac=ops['val_frac'], seed=ops['seed'])

    if n_hidden is not None:
        model, hist, pw = train_one(
            inputs, target, mask, train_idx, val_idx,
            n_hidden=n_hidden, ops=ops, device=device, verbose=verbose)
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
        train_idx  = result['train_idx'],
        val_idx    = result['val_idx'],
    )
    if 'sweep' in result:
        obj['sweep_val'] = {n_h: s['best_val'] for n_h, s in result['sweep'].items()}
    torch.save(obj, path)
    print(f'saved to {path}')


def load_model(path):
    obj = torch.load(path, weights_only=False)
    model = BehaviourRNN(n_hidden=obj['n_hidden'])
    model.load_state_dict(obj['state_dict'])
    model.eval()
    return model, obj
