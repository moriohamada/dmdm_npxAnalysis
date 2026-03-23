"""
fns for analysis of lick prediction models
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from config import PATHS
from lick_pred.features import (
    FEATURE_COLS, CONTINUOUS_COLS, N_TF_HIST, N_FEATURES,
    build_mouse_features, OUTCOME_MAP, PREV_TIME_MASK, ABLATION_GROUPS,
)
from data.session import Session
from lick_pred.models import (
    LinearLickModel, NetworkLickModel, evaluate, compute_class_weight,
)
from lick_pred.run import _group_sessions_by_mouse

BIN_WIDTH = 0.05
SAVE_DIR = os.path.join(PATHS['plots_dir'], 'lick_pred')
# feature groups for plotting
DRIVER_GROUPS = {
    'stimulus':       FEATURE_COLS['stimulus'],
    'time':           FEATURE_COLS['time_in_trial'],
    'block':          FEATURE_COLS['block'],
    'prev_hit':       FEATURE_COLS['prev_outcome'][:1],
    'prev_miss':      FEATURE_COLS['prev_outcome'][1:2],
    'prev_fa':        FEATURE_COLS['prev_outcome'][2:3],
    'prev_abort':     FEATURE_COLS['prev_outcome'][3:4],
    'prev_hit_time':  FEATURE_COLS['prev_hit_time'],
    'prev_miss_time': FEATURE_COLS['prev_miss_time'],
    'prev_fa_time':   FEATURE_COLS['prev_fa_time'],
    'prev_abort_time':FEATURE_COLS['prev_abort_time'],
    'time_since_rwd': FEATURE_COLS['time_since_reward'],
    'trial_num':      FEATURE_COLS['trial_num'],
}


def _fix_old_ablation(abl):
    """hack: old results have per-feature ablation keys, regroup into ABLATION_GROUPS.
    remove once results are rerun on HPC with grouped ablation"""
    if 'trial_history' in abl:
        return abl
    out = {}
    for group, cols in ABLATION_GROUPS.items():
        matching = [k for k, v in FEATURE_COLS.items()
                    if all(c in cols for c in v)]
        if matching:
            out[group] = np.mean([abl[k] for k in matching if k in abl], axis=0)
    return out


#%% loading
def load_results(results_dir=None):
    """
    load all new-format lick prediction results
    returns dict keyed by animal name
    """
    if results_dir is None:
        results_dir = os.path.join(PATHS['npx_dir_local'], 'lick_prediction')

    all_res = {}
    for f in sorted(os.listdir(results_dir)):
        if not f.endswith('_lick_pred.pkl'):
            continue
        with open(os.path.join(results_dir, f), 'rb') as fh:
            res = pickle.load(fh)
        if 'full_results' not in res:
            continue
        all_res[res['animal']] = res
    return all_res


def _get_trial_outcomes(sess):
    """
    extract trial outcome labels from Session object
    """
    outcomes = {}
    for tr, row in sess.trials.iterrows():
        if row['IsHit']:
            outcomes[tr] = 'Hit'
        elif row['IsFA']:
            outcomes[tr] = 'FA'
        elif row['IsMiss']:
            outcomes[tr] = 'Miss'
        elif row['IsAbort']:
            outcomes[tr] = 'Abort'
        else:
            outcomes[tr] = 'Ref'
    return outcomes


def load_mouse(animal, all_res, npx_dir=None):
    """
    load session data and trial outcomes for one mouse
    returns dict with sessions_data (52-feature), session_names, trial_outcomes
    """
    if npx_dir is None:
        npx_dir = PATHS['npx_dir_local']

    grouped = _group_sessions_by_mouse(npx_dir, npx_only=False)
    sessions_data, session_names = build_mouse_features(grouped[animal],
                                                         truncate_at_change=False)

    trial_outcomes = []
    change_tfs = []
    trial_info = []
    for path in grouped[animal]:
        sess = Session.load(path)
        if sess.name in session_names:
            trial_outcomes.append(_get_trial_outcomes(sess))
            change_tfs.append({tr: row['Stim2TF']
                               for tr, row in sess.trials.iterrows()
                               if not pd.isna(row.get('Stim2TF', np.nan))})
            trial_info.append({tr: dict(block=row.get('hazardblock', '?'),
                                        tr_in_block=row.get('tr_in_block', '?'))
                               for tr, row in sess.trials.iterrows()})

    return dict(
        animal=animal,
        sessions_data=sessions_data,
        session_names=session_names,
        trial_outcomes=trial_outcomes,
        change_tfs=change_tfs,
        trial_info=trial_info,
    )


def _reconstruct_model(state_dict):
    """
    rebuild a linear or network model from its state dict
    """
    if 'linear.weight' in state_dict:
        n_features = state_dict['linear.weight'].shape[1]
        model = LinearLickModel(n_features=n_features)
    else:
        n_features = state_dict['net.0.weight'].shape[1]
        n_hidden = state_dict['net.0.weight'].shape[0]
        model = NetworkLickModel(n_features=n_features, n_hidden=n_hidden)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_session(mouse, all_res, arch, sess_idx):
    """normalise and predict for one session using that fold's model
    returns X_raw, X_norm, y, y_pred, trial_ids
    """
    fold_results = all_res[mouse['animal']]['full_results'][arch]
    X, y, trial_ids = mouse['sessions_data'][sess_idx]
    norm_mean, norm_std = fold_results['fold_norm'][sess_idx]

    X_norm = X.copy()
    X_norm[:, CONTINUOUS_COLS] = (X_norm[:, CONTINUOUS_COLS] - norm_mean) / norm_std
    for time_col, gate_col in PREV_TIME_MASK:
        X_norm[X[:, gate_col] == 0, time_col] = 0.0

    model = _reconstruct_model(fold_results['fold_models'][sess_idx])
    with torch.no_grad():
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        y_pred = torch.sigmoid(model(X_t)).numpy()

    return X, X_norm, y, y_pred, trial_ids


def perfect_model_loss(sessions_data):
    """
    loss floor from gaussian-smoothed targets
    """
    all_y = np.concatenate([d[1] for d in sessions_data])
    pos_weight = (all_y <= 0).sum() / max(1, (all_y > 0).sum())
    y = np.clip(all_y, 1e-7, 1 - 1e-7)
    return -(pos_weight * y * np.log(y) + (1 - y) * np.log(1 - y)).mean()


#%% baseline plots

def _best_network_arch(full_results):
    """find network architecture with lowest mean loss"""
    net_archs = {k: v for k, v in full_results.items() if k != 'linear'}
    if not net_archs:
        return None
    return min(net_archs, key=lambda k: net_archs[k]['mean_loss'])


def compute_chance_loss(sessions_data):
    """per-session weighted BCE for optimal constant prediction"""
    all_y = np.concatenate([d[1] for d in sessions_data])
    pos_weight = (all_y <= 0).sum() / max(1, (all_y > 0).sum())
    y_mean = all_y.mean()
    p_const = pos_weight * y_mean / (pos_weight * y_mean + (1 - y_mean))
    p_const = np.clip(p_const, 1e-7, 1 - 1e-7)

    losses = np.full(len(sessions_data), np.nan)
    for i, (_, y, _) in enumerate(sessions_data):
        losses[i] = -(pos_weight * y * np.log(p_const)
                       + (1 - y) * np.log(1 - p_const)).mean()
    return losses


def plot_model_vs_chance(all_res, mice, save_path='default'):
    """line plot: perfect / chance / linear / each network size per mouse

    thin grey line per mouse, thick black grand average
    """
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, 'model_vs_chance.png')

    animals = sorted(mice.keys())

    # collect all network architectures across mice (sorted by hidden size)
    net_keys = {k for a in animals
                for k in all_res[a]['full_results'] if k != 'linear'}
    # sort by numeric part if present, otherwise alphabetically
    def _arch_sort_key(k):
        digits = ''.join(c for c in k if c.isdigit())
        return int(digits) if digits else 0
    net_archs = sorted(net_keys, key=_arch_sort_key)

    # build labels with config info (e.g. "h8\northo=0.01")
    def _net_label(arch):
        for a in animals:
            fr = all_res[a]['full_results']
            if arch in fr:
                return fr[arch].get('config_key', arch)
        return arch
    net_labels = []
    for arch in net_archs:
        cfg_key = _net_label(arch)
        parts = [arch]
        if '_wd' in cfg_key and '_ortho' in cfg_key:
            wd = cfg_key.split('_wd')[1].split('_ortho')[0]
            lo = cfg_key.split('_ortho')[1]
            parts.append(f'wd={wd} ortho={lo}')
        elif 'ortho' in cfg_key:
            val = cfg_key.split('ortho')[-1]
            parts.append(f'ortho={val}')
        elif 'lambda' in cfg_key:
            val = cfg_key.split('lambda')[-1]
            parts.append(f'wd={val}')
        net_labels.append('\n'.join(parts))
    labels = ['perfect', 'chance', 'linear'] + net_labels
    x = np.arange(len(labels))

    per_mouse = []
    for animal in animals:
        full_results = all_res[animal]['full_results']
        row = [
            perfect_model_loss(mice[animal]['sessions_data']),
            compute_chance_loss(mice[animal]['sessions_data']).mean(),
            full_results['linear']['mean_loss'],
        ]
        for arch in net_archs:
            if arch in full_results:
                row.append(full_results[arch]['mean_loss'])
            else:
                row.append(np.nan)
        per_mouse.append(row)
    per_mouse = np.array(per_mouse)

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(labels)), 5))
    for row in per_mouse:
        ax.plot(x, row, color='grey', alpha=0.3, linewidth=0.8)
    ax.plot(x, np.nanmean(per_mouse, axis=0), color='k', linewidth=2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean test loss (weighted BCE)')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


def _get_linear_weights(all_res, animal):
    """average linear weights across folds"""
    fold_results = all_res[animal]['full_results']['linear']
    all_weights = np.array([state['linear.weight'][0].numpy()
                            for state in fold_results['fold_models']])
    return all_weights.mean(axis=0)


def _non_stimulus_features():
    """feature names and column indices for non-stimulus features, expanded"""
    feats = {}
    for name, cols in FEATURE_COLS.items():
        if name == 'stimulus':
            continue
        elif name == 'prev_outcome':
            for oname, oidx in OUTCOME_MAP.items():
                feats[f'prev_{oname}'] = [cols[oidx]]
        else:
            feats[name] = cols
    return feats


def plot_linear_weights(all_res, mice, save_path='default'):
    """stimulus filter and non-stimulus weights per mouse, with mean at bottom"""
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, 'linear_weights.png')
    animals = sorted(mice.keys())
    n_mice = len(animals)
    all_weights = np.array([_get_linear_weights(all_res, a) for a in animals])

    other_features = _non_stimulus_features()
    other_names = list(other_features.keys())
    t_ax = np.arange(N_TF_HIST) * BIN_WIDTH - 2.0

    fig, axes = plt.subplots(n_mice + 1, 2, figsize=(10, 2.5 * (n_mice + 1)),
                             gridspec_kw={'width_ratios': [2, 1]})
    if n_mice == 1:
        axes = axes[None, :] if axes.ndim == 1 else axes

    for row, animal in enumerate(animals):
        weights = all_weights[row]

        ablation = {}
        if 'ablation' in all_res[animal]['full_results'].get('linear', {}):
            ablation = _fix_old_ablation(
                all_res[animal]['full_results']['linear']['ablation'])

        axes[row, 0].plot(t_ax, weights[:N_TF_HIST])
        axes[row, 0].axhline(0, color='k', linewidth=0.5)
        axes[row, 0].set_ylabel(animal, fontsize=8)

        bar_vals = [weights[other_features[name]].mean() for name in other_names]
        colours = []
        for name in other_names:
            if not ablation:
                colours.append('grey')
                continue
            if name in ablation:
                abl_key = name
            elif name in ('block',):
                abl_key = 'block'
            else:
                abl_key = 'trial_history'
            if abl_key in ablation:
                sig = stats.ttest_1samp(ablation[abl_key], 0).pvalue < 0.05
            else:
                sig = False
            colours.append('red' if sig else 'grey')

        axes[row, 1].barh(range(len(other_names)), bar_vals, color=colours)
        axes[row, 1].axvline(0, color='k', linewidth=0.5)
        axes[row, 1].set_yticks(range(len(other_names)))
        axes[row, 1].set_yticklabels(other_names, fontsize=7)

    # mean row
    mean_weights = all_weights.mean(axis=0)
    sem_weights = all_weights.std(axis=0) / np.sqrt(n_mice)

    axes[-1, 0].plot(t_ax, mean_weights[:N_TF_HIST])
    axes[-1, 0].fill_between(t_ax, mean_weights[:N_TF_HIST] - sem_weights[:N_TF_HIST],
                              mean_weights[:N_TF_HIST] + sem_weights[:N_TF_HIST], alpha=0.3)
    axes[-1, 0].axhline(0, color='k', linewidth=0.5)
    axes[-1, 0].set_ylabel('Mean', fontsize=8, fontweight='bold')
    axes[-1, 0].set_xlabel('Time before current bin (s)')

    mean_vals = [all_weights[:, other_features[name]].mean(axis=1)
                 for name in other_names]
    mean_means = [v.mean() for v in mean_vals]
    _, mean_pvals = zip(*[stats.ttest_1samp(v, 0) for v in mean_vals])
    colours = ['red' if p < 0.05 else 'grey' for p in mean_pvals]
    axes[-1, 1].barh(range(len(other_names)), mean_means, color=colours)
    axes[-1, 1].axvline(0, color='k', linewidth=0.5)
    axes[-1, 1].set_yticks(range(len(other_names)))
    axes[-1, 1].set_yticklabels(other_names, fontsize=7)
    axes[-1, 1].set_xlabel('Weight')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


def plot_feature_ablation(all_res, mice, arch='linear', save_path='default'):
    """feature ablation: thin grey line per mouse, thick black average"""
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, f'feature_ablation_{arch}.png')
    animals = sorted(mice.keys())
    groups = list(ABLATION_GROUPS.keys())
    x = np.arange(len(groups))

    per_mouse = np.full((len(animals), len(groups)), np.nan)
    for i, animal in enumerate(animals):
        if arch not in all_res[animal]['full_results']:
            continue
        if 'ablation' not in all_res[animal]['full_results'][arch]:
            continue
        abl = _fix_old_ablation(all_res[animal]['full_results'][arch]['ablation'])
        for j, g in enumerate(groups):
            per_mouse[i, j] = np.nanmean(abl[g])

    fig, ax = plt.subplots(figsize=(6, 4))
    for row in per_mouse:
        if not np.all(np.isnan(row)):
            ax.plot(x, row, color='grey', alpha=0.3, linewidth=0.8)
    ax.plot(x, np.nanmean(per_mouse, axis=0), color='k', linewidth=2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha='right')
    ax.set_ylabel('Loss increase (ablation)')
    ax.set_title(f'{arch} model')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


#%% session and trial plots

def plot_session_heatmap(mouse, all_res, arch, sess_idx, save_path='default'):
    """heatmap of stimulus, target, and prediction for lick trials in a session"""
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR,
            f'heatmap_{mouse["animal"]}_{mouse["session_names"][sess_idx]}.png')
    X_raw, _, y, y_pred, trial_ids = predict_session(mouse, all_res, arch, sess_idx)

    lick_trials = []
    for tr in np.unique(trial_ids):
        mask = trial_ids == tr
        y_tr = y[mask]
        if y_tr.max() > 0:
            lick_trials.append(dict(
                trial=tr,
                lick_bin=np.argmax(y_tr),
                stim=X_raw[mask, N_TF_HIST - 1],
                target=y_tr,
                pred=y_pred[mask],
            ))

    lick_trials.sort(key=lambda t: t['lick_bin'])
    n_lick = len(lick_trials)
    max_bins = max(len(t['stim']) for t in lick_trials)

    stim_mat = np.full((n_lick, max_bins), np.nan)
    target_mat = np.full((n_lick, max_bins), np.nan)
    pred_mat = np.full((n_lick, max_bins), np.nan)

    for i, t in enumerate(lick_trials):
        n = len(t['stim'])
        stim_mat[i, :n] = t['stim']
        target_mat[i, :n] = t['target']
        pred_mat[i, :n] = t['pred']

    fig, axes = plt.subplots(1, 3, figsize=(14, max(4, n_lick * 0.15)),
                             sharey=True)
    extent = [0, max_bins * BIN_WIDTH, n_lick - 0.5, -0.5]

    axes[0].imshow(stim_mat, aspect='auto', extent=extent, cmap='RdBu_r')
    axes[0].set_title('Stimulus (log2 TF)')
    axes[0].set_ylabel('Trial (sorted by lick time)')

    axes[1].imshow(target_mat, aspect='auto', extent=extent, cmap='Reds')
    axes[1].set_title('Target')

    axes[2].imshow(pred_mat, aspect='auto', extent=extent, cmap='Reds')
    axes[2].set_title(f'Prediction ({arch})')

    for ax in axes:
        ax.set_xlabel('Time in trial (s)')

    fig.suptitle(f'{mouse["animal"]} - {mouse["session_names"][sess_idx]}',
                 fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


ABLATION_COLOURS = {
    'stimulus':      'tab:green',
    'time_in_trial': 'tab:orange',
    'block':         'tab:purple',
    'trial_history': 'tab:brown',
}


def _predict_ablated(model, X_norm, group_cols):
    """predict with a feature group zeroed out"""
    X_abl = X_norm.copy()
    X_abl[:, group_cols] = 0.0
    with torch.no_grad():
        X_t = torch.tensor(X_abl, dtype=torch.float32)
        return torch.sigmoid(model(X_t)).numpy()


def plot_trial_detail(trial_idx, X_raw, X_norm, y,
                      y_pred_linear, y_pred_net, ablated_preds,
                      trial_ids, weights, bias,
                      title=None, net_label='network'):
    """combined trial detail: predictions, linear decomposition, network ablation

    row 1: stimulus trace
    row 2: predictions (target + linear + network)
    row 3: time-varying logit contributions from linear model (stimulus + time)
    row 4: constant logit contributions from linear model (bar)
    row 5: network ablation predictions
    """
    mask = trial_ids == trial_idx
    if mask.sum() < 2:
        return None
    bins = np.arange(mask.sum()) * BIN_WIDTH

    fig, (ax_stim, ax_pred, ax_time, ax_bar, ax_abl) = plt.subplots(
        5, 1, figsize=(8, 10),
        gridspec_kw={'height_ratios': [0.8, 1, 1, 1.2, 1]})

    # row 1: stimulus
    ax_stim.plot(bins, X_raw[mask, N_TF_HIST - 1], color='k')
    ax_stim.set_ylabel('log2 TF')
    if title is None:
        title = f'Trial {int(trial_idx)}'
    ax_stim.set_title(title)
    ax_stim.set_xlim(bins[0], bins[-1])

    # row 2: predictions
    ax_pred.plot(bins, y[mask], label='target', color='k', alpha=0.4)
    ax_pred.plot(bins, y_pred_linear[mask], label='linear', color='tab:blue')
    if y_pred_net is not None:
        ax_pred.plot(bins, y_pred_net[mask], label=net_label, color='tab:red')
    ax_pred.set_ylabel('P(lick)')
    ax_pred.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left',
                   borderaxespad=0)
    ax_pred.set_xlim(bins[0], bins[-1])

    # row 3: time-varying logit contributions (linear)
    stim_cols = DRIVER_GROUPS['stimulus']
    stim_contrib = (weights[stim_cols] * X_norm[mask][:, stim_cols]).sum(axis=1)
    ax_time.plot(bins, stim_contrib, label='stimulus', color='k')

    time_cols = DRIVER_GROUPS['time']
    time_contrib = (weights[time_cols] * X_norm[mask][:, time_cols]).sum(axis=1)
    ax_time.plot(bins, time_contrib, label='time_in_trial', color='tab:orange')

    ax_time.axhline(0, color='grey', linewidth=0.5)
    ax_time.set_ylabel('Logit (linear)')
    ax_time.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left',
                   borderaxespad=0)
    ax_time.set_xlim(bins[0], bins[-1])

    # row 4: constant contributions bar (linear)
    constant_groups = {k: v for k, v in DRIVER_GROUPS.items()
                       if k not in ('stimulus', 'time')}
    names = list(constant_groups.keys()) + ['bias']
    vals = []
    for name, cols in constant_groups.items():
        vals.append((weights[cols] * X_norm[mask][:, cols]).sum(axis=1)[0])
    vals.append(bias)

    bar_colours = ['tab:red' if v > 0 else 'tab:blue' for v in vals]
    ax_bar.barh(range(len(names)), vals, color=bar_colours)
    ax_bar.set_yticks(range(len(names)))
    ax_bar.set_yticklabels(names, fontsize=7)
    ax_bar.axvline(0, color='k', linewidth=0.5)
    ax_bar.set_xlabel('Logit (linear)')

    # row 5: network ablation
    if y_pred_net is not None and ablated_preds:
        ax_abl.plot(bins, y[mask], label='target', color='grey', alpha=0.4)
        ax_abl.plot(bins, y_pred_net[mask], label=net_label, color='tab:red', alpha=0.5)
        for group_name, y_abl in ablated_preds.items():
            ax_abl.plot(bins, y_abl[mask], label=f'no {group_name}',
                        color=ABLATION_COLOURS[group_name], linestyle='--')
        ax_abl.set_ylabel('P(lick)')
        ax_abl.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left',
                      borderaxespad=0)
    else:
        ax_abl.set_visible(False)
    ax_abl.set_xlabel('Time in trial (s)')
    ax_abl.set_xlim(bins[0], bins[-1])

    for ax in [ax_stim, ax_pred, ax_time, ax_bar, ax_abl]:
        sns.despine(ax=ax)

    fig.tight_layout()
    return fig


def plot_all_lick_trials(mouse, all_res, sess_idx,
                         lick_only=True, save_path=None):
    """per-trial predictions with linear decomposition and network ablation

    one pdf per session, one page per trial
    """
    X_raw, X_norm, y, y_pred_linear, trial_ids = predict_session(
        mouse, all_res, 'linear', sess_idx)
    outcomes = mouse['trial_outcomes'][sess_idx]
    change_tfs = mouse.get('change_tfs', [{}] * len(mouse['sessions_data']))[sess_idx]
    trial_info = mouse.get('trial_info', [{}] * len(mouse['sessions_data']))[sess_idx]

    # network prediction + ablation
    best_net = _best_network_arch(all_res[mouse['animal']]['full_results'])
    y_pred_net = None
    ablated_preds = {}
    if best_net:
        _, _, _, y_pred_net, _ = predict_session(mouse, all_res, best_net, sess_idx)
        fold_results = all_res[mouse['animal']]['full_results'][best_net]
        net_model = _reconstruct_model(fold_results['fold_models'][sess_idx])
        for group_name, cols in ABLATION_GROUPS.items():
            ablated_preds[group_name] = _predict_ablated(net_model, X_norm, cols)

    # linear weights for contribution decomposition
    lin_results = all_res[mouse['animal']]['full_results']['linear']
    state = lin_results['fold_models'][sess_idx]
    weights = state['linear.weight'][0].numpy()
    bias = state['linear.bias'].item()

    all_trial_ids = np.unique(trial_ids)
    if lick_only:
        all_trial_ids = [tr for tr in all_trial_ids
                         if y[trial_ids == tr].max() > 0]

    if save_path is None:
        save_dir = os.path.join(PATHS['plots_dir'], 'lick_pred')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f'lick_pred_trials_{mouse["animal"]}_{mouse["session_names"][sess_idx]}.pdf')

    with PdfPages(save_path) as pdf:
        for tr in all_trial_ids:
            trial_type = outcomes.get(tr, None)
            ch_tf = change_tfs.get(tr, None)
            info = trial_info.get(tr, {})
            block = info.get('block', '?')
            tr_in_block = info.get('tr_in_block', '?')

            title_parts = [f'Trial {int(tr)}']
            if trial_type:
                title_parts.append(f'({trial_type})')
            if ch_tf is not None:
                title_parts.append(f'({ch_tf}Hz)')
            title_parts.append(f'[{block} block, #{tr_in_block}]')
            title = ' '.join(title_parts)

            fig = plot_trial_detail(tr, X_raw, X_norm, y,
                                    y_pred_linear, y_pred_net, ablated_preds,
                                    trial_ids, weights, bias,
                                    title=title,
                                    net_label=best_net or 'network')
            if fig is None:
                continue
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f'Saved {len(all_trial_ids)} trial plots to {save_path}')


#%% failure analysis

def _per_trial_loss(y, y_pred, trial_ids):
    """weighted BCE per trial"""
    pos_weight = (y <= 0).sum() / max(1, (y > 0).sum())
    y_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
    per_bin = -(pos_weight * y * np.log(y_clip) + (1 - y) * np.log(1 - y_clip))

    trial_losses = {}
    for tr in np.unique(trial_ids):
        mask = trial_ids == tr
        trial_losses[tr] = per_bin[mask].mean()
    return trial_losses


def _collect_trial_losses(mice, all_res, arch='linear'):
    """collect per-trial losses with metadata across all mice

    returns list of dicts with keys: animal, session, trial, loss, outcome, block, position
    """
    records = []
    for animal in sorted(mice.keys()):
        mouse = mice[animal]
        n_sess = len(mouse['sessions_data'])
        for sess_idx in range(n_sess):
            X_raw, _, y, y_pred, trial_ids = predict_session(
                mouse, all_res, arch, sess_idx)
            outcomes = mouse['trial_outcomes'][sess_idx]
            trial_losses = _per_trial_loss(y, y_pred, trial_ids)

            all_trials = np.unique(trial_ids)
            n_trials = len(all_trials)
            for rank, tr in enumerate(all_trials):
                block_val = X_raw[trial_ids == tr, N_TF_HIST + 1][0]
                records.append(dict(
                    animal=animal,
                    session=mouse['session_names'][sess_idx],
                    trial=tr,
                    loss=trial_losses[tr],
                    outcome=outcomes.get(tr, '?'),
                    block='late' if block_val > 0.5 else 'early',
                    position=rank / max(1, n_trials - 1),
                ))
    return records


def plot_loss_by_trial_type(records, save_path='default'):
    """per-trial loss grouped by outcome, with per-mouse lines"""
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, 'loss_by_trial_type.png')
    outcome_types = ['Hit', 'FA', 'Miss']
    x = np.arange(len(outcome_types))

    animals = sorted(set(r['animal'] for r in records))
    per_mouse = np.full((len(animals), len(outcome_types)), np.nan)
    for i, animal in enumerate(animals):
        for j, outcome in enumerate(outcome_types):
            vals = [r['loss'] for r in records
                    if r['animal'] == animal and r['outcome'] == outcome]
            if vals:
                per_mouse[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(5, 4))
    for row in per_mouse:
        ax.plot(x, row, color='grey', alpha=0.3, linewidth=0.8)
    ax.plot(x, np.nanmean(per_mouse, axis=0), color='k', linewidth=2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(outcome_types)
    ax.set_ylabel('Mean trial loss (weighted BCE)')
    ax.set_xlabel('Trial outcome')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


def plot_loss_by_block(records, save_path='default'):
    """per-trial loss split by block, with per-mouse lines"""
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, 'loss_by_block.png')
    blocks = ['early', 'late']
    x = np.arange(len(blocks))

    animals = sorted(set(r['animal'] for r in records))
    per_mouse = np.full((len(animals), len(blocks)), np.nan)
    for i, animal in enumerate(animals):
        for j, block in enumerate(blocks):
            vals = [r['loss'] for r in records
                    if r['animal'] == animal and r['block'] == block]
            if vals:
                per_mouse[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(4, 4))
    for row in per_mouse:
        ax.plot(x, row, color='grey', alpha=0.3, linewidth=0.8)
    ax.plot(x, np.nanmean(per_mouse, axis=0), color='k', linewidth=2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_ylabel('Mean trial loss (weighted BCE)')
    ax.set_xlabel('Block')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


def plot_loss_by_trial_position(records, n_bins=5, save_path='default'):
    """loss vs normalised position within session"""
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, 'loss_by_trial_position.png')
    positions = np.array([r['position'] for r in records])
    losses = np.array([r['loss'] for r in records])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (positions >= bin_edges[i]) & (positions < bin_edges[i + 1])
        if mask.sum() > 0:
            means[i] = losses[mask].mean()
            sems[i] = losses[mask].std() / np.sqrt(mask.sum())

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(bin_centres, means, yerr=sems, marker='o', capsize=3)
    ax.set_xlabel('Position in session (normalised)')
    ax.set_ylabel('Mean trial loss (weighted BCE)')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


def plot_session_loss_scatter(all_res, mice, arch='linear', save_path='default'):
    """per-session test loss, one point per session, coloured by mouse"""
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, 'session_loss_scatter.png')
    animals = sorted(mice.keys())
    colours = plt.cm.tab20(np.linspace(0, 1, len(animals)))

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, animal in enumerate(animals):
        losses = all_res[animal]['full_results'][arch]['test_losses']
        ax.scatter(range(len(losses)), losses, color=colours[i],
                   label=animal, alpha=0.7, s=30)

    ax.set_xlabel('Session (fold) index')
    ax.set_ylabel('Test loss (weighted BCE)')
    ax.legend(fontsize=6, ncol=2, loc='upper right')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


def run_all_lick_model_analyses(results_dir=None):
    """run all lick prediction analyses and generate plots"""
    from lick_pred.hidden_units import (
        analyse_hidden_units, plot_network_schematic, plot_unit_summary,
        plot_unit_responses, plot_unit_ablation_trials,
    )

    all_res = load_results(results_dir)
    mice = {a: load_mouse(a, all_res) for a in all_res}

    plot_model_vs_chance(all_res, mice)
    plot_linear_weights(all_res, mice)

    # feature ablation for all architectures
    all_archs = sorted({k for a in all_res for k in all_res[a]['full_results']})
    for arch in all_archs:
        plot_feature_ablation(all_res, mice, arch=arch)

    records = _collect_trial_losses(mice, all_res)
    plot_loss_by_trial_type(records)
    plot_loss_by_block(records)
    plot_loss_by_trial_position(records)
    plot_session_loss_scatter(all_res, mice)

    for animal in all_res:
        mouse = mice[animal]
        for sess_idx in [0]:#range(len(mouse['sessions_data'])):
            print(f"{animal} - {mouse['session_names'][sess_idx]}")
            plot_all_lick_trials(mouse, all_res, sess_idx=sess_idx)

    # hidden unit analysis
    arch = 'h8'
    for animal in all_res:
        mouse = mice[animal]
        if arch not in all_res[animal]['full_results']:
            continue
        print(f'{animal} — hidden unit analysis ({arch})')
        unit_result = analyse_hidden_units(mouse, all_res, arch)
        plot_network_schematic(unit_result, animal, arch)
        plot_unit_summary(unit_result, animal, arch)
        plot_unit_responses(mouse, all_res, arch, unit_result)
        plot_unit_ablation_trials(mouse, all_res, arch, unit_result)


if __name__ == '__main__':
    #%% run
    all_res = load_results()
    mice = {a: load_mouse(a, all_res) for a in all_res}

    #%% loss comparison
    plot_model_vs_chance(all_res, mice)

    #%% linear weights
    plot_linear_weights(all_res, mice)

    #%% feature ablation
    plot_feature_ablation(all_res, mice)

    #%% failure analysis
    records = _collect_trial_losses(mice, all_res)
    plot_loss_by_trial_type(records)
    plot_loss_by_block(records)
    plot_loss_by_trial_position(records)
    plot_session_loss_scatter(all_res, mice)

    #%% example session
    animal = list(mice.keys())[0]
    plot_session_heatmap(mice[animal], all_res, 'linear', 0)
    plot_all_lick_trials(mice[animal], all_res, sess_idx=0)
