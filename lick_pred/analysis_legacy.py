"""
analysis of lick prediction models - for fits pre 19/03/26, format changed after
"""
import pickle
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from config import PATHS
from lick_pred.features import build_mouse_features
from data.session import Session
from lick_pred.models import NetworkLickModel
from lick_pred.run import _group_sessions_by_mouse


N_TF_HIST = 40
BIN_WIDTH = 0.05

# old feature layout (49 features) - before prev_event_time was split into 4
LEGACY_FEATURE_COLS = {
    'stimulus':          list(range(0, N_TF_HIST)),
    'time_in_trial':     [N_TF_HIST],
    'block':             [N_TF_HIST + 1],
    'prev_outcome':      list(range(N_TF_HIST + 2, N_TF_HIST + 6)),
    'prev_event_time':   [N_TF_HIST + 6],
    'time_since_reward': [N_TF_HIST + 7],
    'trial_num':         [N_TF_HIST + 8],
}
LEGACY_OUTCOME_MAP = {'hit': 0, 'miss': 1, 'fa': 2, 'abort': 3}
LEGACY_CONTINUOUS_COLS = (LEGACY_FEATURE_COLS['stimulus']
                          + LEGACY_FEATURE_COLS['time_in_trial']
                          + LEGACY_FEATURE_COLS['prev_event_time']
                          + LEGACY_FEATURE_COLS['time_since_reward']
                          + LEGACY_FEATURE_COLS['trial_num'])

# feature groups for contribution decomposition (legacy indices)
DRIVER_GROUPS = {
    'stimulus':       LEGACY_FEATURE_COLS['stimulus'],
    'time':           LEGACY_FEATURE_COLS['time_in_trial'],
    'block':          LEGACY_FEATURE_COLS['block'],
    'prev_trial':     (LEGACY_FEATURE_COLS['prev_outcome']
                       + LEGACY_FEATURE_COLS['prev_event_time']),
    'time_since_rwd': LEGACY_FEATURE_COLS['time_since_reward'],
    'trial_num':      LEGACY_FEATURE_COLS['trial_num'],
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def _to_legacy_features(X):
    """collapse current 52-feature vectors to legacy 49-feature layout

    only difference: 4 per-outcome event time cols (46-49) -> 1 summed col (46)
    works because only one outcome is nonzero per trial
    """
    return np.column_stack([
        X[:, :46],
        X[:, 46:50].sum(axis=1, keepdims=True),
        X[:, 50:52],
    ])


#%% loading

def load_legacy_results(results_dir=None):
    """load all legacy lick prediction results

    returns dict keyed by animal name, each entry:
        'animal':           str
        'session_names':    list of str
        'linear_weights':   np.array (49,)
        'linear_bias':      float (from _linear.pt)
        'norm_mu':          np.array (44,)
        'norm_sd':          np.array (44,)
        'linear_ablation':  dict {feature_group: {'loss_increase': array, 'p': float}}
        'sweep_results':    dict {config_key: {'model', 'weight_decay', ...}}
    """
    if results_dir is None:
        results_dir = os.path.join(PATHS['npx_dir_local'], 'lick_prediction')

    all_res = {}
    for f in sorted(os.listdir(results_dir)):
        if not f.endswith('_lick_pred.pkl'):
            continue
        with open(os.path.join(results_dir, f), 'rb') as fh:
            res = pickle.load(fh)
        if 'linear_weights' not in res:
            continue

        animal = res['animal']
        pt_path = os.path.join(results_dir, f'{animal}_linear.pt')
        if os.path.exists(pt_path):
            sd = torch.load(pt_path, map_location='cpu', weights_only=True)
            res['linear_bias'] = sd['linear.bias'].item()
        else:
            res['linear_bias'] = 0.0

        net_path = os.path.join(results_dir, f'{animal}_model.pt')
        if os.path.exists(net_path):
            res['network_sd'] = torch.load(net_path, map_location='cpu',
                                            weights_only=True)
        all_res[animal] = res
    return all_res


def _get_trial_outcomes(sess):
    """extract trial outcome labels from a Session object"""
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
    """load session data for one mouse and combine with saved model

    returns dict with:
        'animal':          str
        'w':               np.array (49,) - linear weights
        'b':               float - linear bias
        'norm_mu':         np.array (44,)
        'norm_sd':         np.array (44,)
        'sessions_data':   list of (X, y, trial_ids) per session, 49-feature layout
        'session_names':   list of str
        'trial_outcomes':  list of {trial_id: 'Hit'/'FA'/'Miss'/...} per session
        'ablation':        dict from saved results
    """
    if npx_dir is None:
        npx_dir = PATHS['npx_dir_local']

    res = all_res[animal]
    grouped = _group_sessions_by_mouse(npx_dir, npx_only=False)
    sessions_data_52, session_names = build_mouse_features(grouped[animal])

    sessions_data = [(_to_legacy_features(X), y, ids)
                     for X, y, ids in sessions_data_52]

    # load trial outcomes (need to re-load sessions to get trial labels)
    all_outcomes = {}
    for path in grouped[animal]:
        sess = Session.load(path)
        all_outcomes[sess.name] = _get_trial_outcomes(sess)
    trial_outcomes = [all_outcomes[name] for name in session_names]

    # reconstruct network model from saved state dict
    net_model = None
    if 'network_sd' in res:
        sd = res['network_sd']
        n_features = sd['net.0.weight'].shape[1]
        n_hidden = sd['net.0.weight'].shape[0]
        net_model = NetworkLickModel(n_features=n_features, n_hidden=n_hidden)
        net_model.load_state_dict(sd)
        net_model.eval()

    return dict(
        animal=animal,
        w=res['linear_weights'],
        b=res['linear_bias'],
        net_model=net_model,
        norm_mu=res['norm_mu'],
        norm_sd=res['norm_sd'],
        sessions_data=sessions_data,
        session_names=session_names,
        trial_outcomes=trial_outcomes,
        ablation=res.get('linear_ablation', {}),
    )


def predict_session(mouse, sess_idx):
    """normalise and predict for one session using saved weights

    returns X_raw, X_norm, y, y_pred_linear, y_pred_net, trial_ids
    y_pred_net is None if no network model was saved
    """
    X, y, trial_ids = mouse['sessions_data'][sess_idx]
    X_norm = X.copy()
    X_norm[:, LEGACY_CONTINUOUS_COLS] = (
        (X_norm[:, LEGACY_CONTINUOUS_COLS] - mouse['norm_mu']) / mouse['norm_sd'])

    logits = X_norm @ mouse['w'] + mouse['b']
    y_pred_linear = _sigmoid(logits)

    y_pred_net = None
    if mouse.get('net_model') is not None:
        with torch.no_grad():
            X_t = torch.tensor(X_norm, dtype=torch.float32)
            y_pred_net = torch.sigmoid(mouse['net_model'](X_t)).numpy()

    return X, X_norm, y, y_pred_linear, y_pred_net, trial_ids


#%% chance level

def compute_chance_loss(sessions_data):
    """per-session weighted BCE loss for optimal constant prediction

    finds the single constant p that minimises weighted BCE across ALL bins,
    then evaluates it per session (matching how the model is evaluated)
    """
    all_y = np.concatenate([d[1] for d in sessions_data])
    pw = (all_y <= 0).sum() / max(1, (all_y > 0).sum())

    # optimal constant sigmoid output under weighted BCE
    y_mean = all_y.mean()
    p = pw * y_mean / (pw * y_mean + (1 - y_mean))
    p = np.clip(p, 1e-7, 1 - 1e-7)

    losses = np.full(len(sessions_data), np.nan)
    for i, (_, y, _) in enumerate(sessions_data):
        losses[i] = -(pw * y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    return losses


def _best_sweep_loss(sweep_results, prefix):
    """mean test loss for the best config matching prefix (e.g. 'linear', 'network')"""
    configs = {k: v for k, v in sweep_results.items() if k.startswith(prefix)}
    if not configs:
        return np.nan
    best_key = min(configs, key=lambda k: configs[k]['mean_loss'])
    return configs[best_key]['mean_loss']


def perfect_model_loss(sessions_data):
    """irreducible loss floor from soft gaussian targets"""
    all_y = np.concatenate([d[1] for d in sessions_data])
    pw = (all_y <= 0).sum() / max(1, (all_y > 0).sum())
    y = np.clip(all_y, 1e-7, 1 - 1e-7)
    return -(pw * y * np.log(y) + (1 - y) * np.log(1 - y)).mean()


def plot_model_vs_chance(all_res, mice, save_path=None):
    """line plot: chance vs best linear vs best network loss per mouse

    each thin line = one mouse, thick line = grand average
    perfect model floor included as a per-mouse point
    """
    animals = sorted(mice.keys())
    labels = ['perfect', 'chance', 'linear', 'network']
    x = np.arange(len(labels))

    per_mouse = []
    for animal in animals:
        floor = perfect_model_loss(mice[animal]['sessions_data'])
        chance = compute_chance_loss(mice[animal]['sessions_data']).mean()
        linear = _best_sweep_loss(all_res[animal]['sweep_results'], 'linear')
        network = _best_sweep_loss(all_res[animal]['sweep_results'], 'network')
        per_mouse.append([floor, chance, linear, network])
    per_mouse = np.array(per_mouse)

    fig, ax = plt.subplots(figsize=(5, 5))
    for row in per_mouse:
        ax.plot(x, row, color='grey', alpha=0.3, linewidth=0.8)
    ax.plot(x, per_mouse.mean(axis=0), color='k', linewidth=2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean test loss (weighted BCE)')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


#%% plotting - weights summary

def _expand_legacy_features():
    """expand prev_outcome one-hot into individual columns for plotting"""
    other_features = {}
    for name, cols in LEGACY_FEATURE_COLS.items():
        if name == 'stimulus':
            continue
        elif name == 'prev_outcome':
            for oname, oidx in LEGACY_OUTCOME_MAP.items():
                other_features[f'prev_{oname}'] = [cols[oidx]]
        else:
            other_features[name] = cols
    return other_features


def plot_linear_weights(all_res, save_path=None):
    """stimulus filter and non-stimulus weights per mouse, with mean at bottom"""
    animals = sorted(all_res.keys())
    n_mice = len(animals)
    all_weights = np.array([all_res[a]['linear_weights'] for a in animals])

    other_features = _expand_legacy_features()
    other_names = list(other_features.keys())
    t_ax = np.arange(N_TF_HIST) * BIN_WIDTH - 2.0

    fig, axes = plt.subplots(n_mice + 1, 2, figsize=(10, 2.5 * (n_mice + 1)),
                             gridspec_kw={'width_ratios': [2, 1]})

    for row, animal in enumerate(animals):
        w = all_res[animal]['linear_weights']
        abl = all_res[animal]['linear_ablation']

        axes[row, 0].plot(t_ax, w[:N_TF_HIST])
        axes[row, 0].axhline(0, color='k', linewidth=0.5)
        axes[row, 0].set_ylabel(animal, fontsize=8)
        if abl['stimulus']['p'] < 0.05:
            axes[row, 0].set_title('*', fontsize=10, loc='right')

        vals = [w[other_features[name]].mean() for name in other_names]
        colours = []
        for name in other_names:
            if name.startswith('prev_') and name != 'prev_event_time':
                p = abl['prev_outcome']['p']
            else:
                p = abl[name]['p']
            colours.append('red' if p < 0.05 else 'grey')

        axes[row, 1].barh(range(len(other_names)), vals, color=colours)
        axes[row, 1].axvline(0, color='k', linewidth=0.5)
        if row == 0:
            axes[row, 1].set_yticks(range(len(other_names)))
            axes[row, 1].set_yticklabels(other_names, fontsize=7)
        else:
            axes[row, 1].set_yticks([])

    mu = all_weights.mean(axis=0)
    sem = all_weights.std(axis=0) / np.sqrt(n_mice)

    axes[-1, 0].plot(t_ax, mu[:N_TF_HIST])
    axes[-1, 0].fill_between(t_ax, mu[:N_TF_HIST] - sem[:N_TF_HIST],
                              mu[:N_TF_HIST] + sem[:N_TF_HIST], alpha=0.3)
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


#%% plotting - session and trial level

def plot_session_heatmap(mouse, sess_idx, save_path=None):
    """heatmap of stimulus, target, and prediction for all lick trials in a session"""
    X_raw, _, y, y_pred, _, trial_ids = predict_session(mouse, sess_idx)

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
    lick_times = []

    for i, t in enumerate(lick_trials):
        n = len(t['stim'])
        stim_mat[i, :n] = t['stim']
        target_mat[i, :n] = t['target']
        pred_mat[i, :n] = t['pred']
        lick_times.append(t['lick_bin'] * BIN_WIDTH)

    fig, axes = plt.subplots(1, 3, figsize=(14, max(4, n_lick * 0.15)),
                             sharey=True)
    extent = [0, max_bins * BIN_WIDTH, n_lick - 0.5, -0.5]

    axes[0].imshow(stim_mat, aspect='auto', extent=extent, cmap='RdBu_r')
    axes[0].set_title('Stimulus (log2 TF)')
    axes[0].set_ylabel('Trial (sorted by lick time)')

    axes[1].imshow(target_mat, aspect='auto', extent=extent, cmap='Reds')
    axes[1].set_title('Target')

    axes[2].imshow(pred_mat, aspect='auto', extent=extent, cmap='Reds')
    axes[2].set_title('Prediction (linear)')

    for i, lt in enumerate(lick_times):
        for ax in axes:
            ax.plot(lt, i, 'k|', markersize=4)
    for ax in axes:
        ax.set_xlabel('Time in trial (s)')

    fig.suptitle(f'{mouse["animal"]} - {mouse["session_names"][sess_idx]}',
                 fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


def plot_trial_detail(trial_idx, X_raw, X_norm, y, y_pred_linear, y_pred_net,
                      trial_ids, w, b, trial_type=None):
    """stimulus, prediction, and feature group contributions for one trial"""
    mask = trial_ids == trial_idx
    bins = np.arange(mask.sum()) * BIN_WIDTH

    fig, (ax_stim, ax_pred, ax_contrib) = plt.subplots(
        3, 1, figsize=(8, 6), sharex=True)

    ax_stim.plot(bins, X_raw[mask, N_TF_HIST - 1])
    ax_stim.set_ylabel('log2 TF')
    title = f'Trial {int(trial_idx)}'
    if trial_type:
        title += f' ({trial_type})'
    ax_stim.set_title(title)

    ax_pred.plot(bins, y[mask], label='target', color='k', alpha=0.5)
    ax_pred.plot(bins, y_pred_linear[mask], label='linear', color='tab:blue')
    if y_pred_net is not None:
        ax_pred.plot(bins, y_pred_net[mask], label='network', color='tab:red')
    ax_pred.set_ylabel('P(lick)')
    ax_pred.legend(fontsize=8)

    for name, cols in DRIVER_GROUPS.items():
        contrib = (w[cols] * X_norm[mask][:, cols]).sum(axis=1)
        ax_contrib.plot(bins, contrib, label=name, alpha=0.8)
    ax_contrib.axhline(b, color='grey', linewidth=0.5, linestyle='--',
                       label='bias')
    ax_contrib.set_ylabel('Logit contribution (linear)')
    ax_contrib.set_xlabel('Time in trial (s)')
    ax_contrib.legend(fontsize=7)

    return fig


def plot_all_lick_trials(mouse, sess_idx, lick_only=True, save_path=None):
    """plot_trial_detail for every trial in a session, saved to a single pdf"""
    X_raw, X_norm, y, y_pred_linear, y_pred_net, trial_ids = \
        predict_session(mouse, sess_idx)
    outcomes = mouse['trial_outcomes'][sess_idx]

    all_trial_ids = np.unique(trial_ids)
    if lick_only:
        all_trial_ids = [tr for tr in all_trial_ids
                         if y[trial_ids == tr].max() > 0]

    from collections import Counter
    type_counts = Counter(outcomes.get(tr, '?') for tr in all_trial_ids)
    print(f'{mouse["session_names"][sess_idx]}: {dict(type_counts)} '
          f'({len(all_trial_ids)} trials)')

    if save_path is None:
        save_path = os.path.join(
            PATHS['plots_dir'],
            f'lick_pred_trials_{mouse["animal"]}_{mouse["session_names"][sess_idx]}.pdf')

    with PdfPages(save_path) as pdf:
        for tr in all_trial_ids:
            trial_type = outcomes.get(tr, None)
            fig = plot_trial_detail(tr, X_raw, X_norm, y, y_pred_linear,
                                    y_pred_net, trial_ids,
                                    mouse['w'], mouse['b'],
                                    trial_type=trial_type)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f'Saved {len(all_trial_ids)} trial plots to {save_path}')

#%%
all_res = load_legacy_results()
plots_dir = PATHS['plots_dir']

plot_linear_weights(all_res,
                    save_path=os.path.join(plots_dir, 'lick_pred_linear_weights.png'))

#%%
mice = {a: load_mouse(a, all_res) for a in all_res}
plot_model_vs_chance(all_res, mice,
                     save_path=os.path.join(plots_dir, 'lick_pred_model_vs_chance.png'))

#%%
mouse = mice['1116760']
plot_session_heatmap(mouse, sess_idx=0,
                     save_path=os.path.join(plots_dir, 'lick_pred_example_session.png'))
plot_all_lick_trials(mouse, sess_idx=0)