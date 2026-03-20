"""
hidden unit analysis for lick prediction network models

# TODO: ideally this should use a model fit on all data (no CV) to avoid
# unit ordering ambiguity across folds. For now I pick a single fold
# (median test loss) as the representative model.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from config import PATHS
from lick_pred.features import FEATURE_COLS, CONTINUOUS_COLS, N_TF_HIST, OUTCOME_MAP
from lick_pred.models import NetworkLickModel, evaluate, compute_class_weight
from lick_pred.analysis import (
    load_results, load_mouse, predict_session, BIN_WIDTH, SAVE_DIR,
    _reconstruct_model, _non_stimulus_features, _best_network_arch,
)


def _pick_fold(all_res, animal, arch):
    """fold with median test loss — representative without being an outlier"""
    losses = all_res[animal]['full_results'][arch]['test_losses']
    return int(np.argsort(losses)[len(losses) // 2])


def _get_fold_model_and_data(mouse, all_res, arch, fold_idx):
    """reconstruct model and normalised test data for one fold"""
    fold_results = all_res[mouse['animal']]['full_results'][arch]
    model = _reconstruct_model(fold_results['fold_models'][fold_idx])

    X, y, trial_ids = mouse['sessions_data'][fold_idx]
    norm_mean, norm_std = fold_results['fold_norm'][fold_idx]
    X_norm = X.copy()
    X_norm[:, CONTINUOUS_COLS] = (X_norm[:, CONTINUOUS_COLS] - norm_mean) / norm_std

    return model, X_norm, y, trial_ids


def _get_hidden_activations(model, X_norm):
    """run input through first layer + relu, return (n_bins, n_hidden)"""
    with torch.no_grad():
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        hidden = torch.relu(model.net[0](X_t)).numpy()
    return hidden


def _predict_from_hidden(model, hidden):
    """run hidden activations through the output layer, return P(lick)"""
    with torch.no_grad():
        h_t = torch.tensor(hidden, dtype=torch.float32)
        logits = model.net[2](h_t).squeeze(-1)
        return torch.sigmoid(logits).numpy()


#%% compute

def analyse_hidden_units(mouse, all_res, arch, fold_idx=None):
    """ablate each hidden unit and extract input weights from one fold

    uses the fold with median test loss unless fold_idx is specified

    returns dict with:
        fold_idx: which fold was used
        importance: (n_hidden,) loss increase when each unit is zeroed
        input_weights: (n_hidden, n_features) first-layer weights
        input_bias: (n_hidden,) first-layer biases
        output_weights: (n_hidden,) second-layer weights
        output_bias: scalar second-layer bias
    """
    animal = mouse['animal']
    if fold_idx is None:
        fold_idx = _pick_fold(all_res, animal, arch)

    model, X_norm, y, trial_ids = _get_fold_model_and_data(
        mouse, all_res, arch, fold_idx)

    # extract weights
    state = all_res[animal]['full_results'][arch]['fold_models'][fold_idx]
    input_weights = state['net.0.weight'].numpy()
    input_bias = state['net.0.bias'].numpy()
    output_weights = state['net.2.weight'][0].numpy()
    output_bias = state['net.2.bias'].item()

    n_hidden = input_weights.shape[0]
    pos_weight = compute_class_weight(y)

    # baseline loss
    baseline_loss = evaluate(model, X_norm, y, pos_weight)

    # ablate each unit
    importance = np.zeros(n_hidden)
    hidden = _get_hidden_activations(model, X_norm)

    for unit in range(n_hidden):
        hidden_abl = hidden.copy()
        hidden_abl[:, unit] = 0.0
        y_pred = _predict_from_hidden(model, hidden_abl)
        y_clip = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -(pos_weight * y * np.log(y_clip)
                 + (1 - y) * np.log(1 - y_clip)).mean()
        importance[unit] = loss - baseline_loss

    return dict(
        fold_idx=fold_idx,
        importance=importance,
        input_weights=input_weights,
        input_bias=input_bias,
        output_weights=output_weights,
        output_bias=output_bias,
    )


#%% plots

def plot_network_schematic(unit_result, animal, arch, save_path='default'):
    """hidden unit overview: stimulus filter, non-stimulus input weights (vertical
    bars aligned across units), and a circle sized by importance / coloured by
    output weight (red = drives licking, blue = suppresses)
    """
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, f'network_schematic_{animal}_{arch}.png')

    importance = unit_result['importance']
    input_w = unit_result['input_weights']
    output_w = unit_result['output_weights']
    n_hidden = len(importance)

    order = np.argsort(importance)[::-1]
    t_ax = np.arange(N_TF_HIST) * BIN_WIDTH - 2.0
    other_features = _non_stimulus_features()
    other_names = list(other_features.keys())
    n_other = len(other_names)

    # importance -> circle size, clipping negatives
    imp_clipped = np.clip(importance, 0, None)
    imp_max = imp_clipped.max() if imp_clipped.max() > 0 else 1.0
    imp_norm = imp_clipped / imp_max
    min_size, max_size = 80, 600

    # output weight -> diverging colourmap
    out_abs_max = max(abs(output_w.min()), abs(output_w.max()), 1e-6)
    cmap = plt.cm.RdBu_r

    # layout: 3 columns per row — [stimulus filter | vertical bars | circle]
    fig, axes = plt.subplots(
        n_hidden, 3, figsize=(12, 1.8 * n_hidden),
        gridspec_kw={'width_ratios': [3, n_other * 0.4, 0.8]})
    if n_hidden == 1:
        axes = axes[None, :]

    # shared y limits for stimulus filters and bar plots
    stim_max = np.abs(input_w[:, :N_TF_HIST]).max() * 1.1
    other_vals = np.array([[input_w[u][other_features[name]].mean()
                            for name in other_names] for u in range(n_hidden)])
    bar_max = np.abs(other_vals).max() * 1.1 if np.abs(other_vals).max() > 0 else 1.0

    for row, unit_idx in enumerate(order):
        w = input_w[unit_idx]
        is_bottom = (row == n_hidden - 1)

        # stimulus filter
        ax_stim = axes[row, 0]
        ax_stim.plot(t_ax, w[:N_TF_HIST], linewidth=0.8)
        ax_stim.axhline(0, color='k', linewidth=0.3)
        ax_stim.set_ylim(-stim_max, stim_max)
        ax_stim.set_ylabel(f'unit {unit_idx}', fontsize=12)
        ax_stim.tick_params(labelsize=9)
        if is_bottom:
            ax_stim.set_xlabel('Time before current bin (s)', fontsize=12)
        else:
            ax_stim.set_xticklabels([])
        sns.despine(ax=ax_stim)

        # non-stimulus weights as vertical bars
        ax_bar = axes[row, 1]
        vals = [w[other_features[name]].mean() for name in other_names]
        colours = ['tab:red' if v > 0 else 'tab:blue' for v in vals]
        ax_bar.bar(range(n_other), vals, color=colours, width=0.7)
        ax_bar.axhline(0, color='k', linewidth=0.3)
        ax_bar.set_ylim(-bar_max, bar_max)
        ax_bar.set_xlim(-0.5, n_other - 0.5)
        ax_bar.set_yticks([])
        if is_bottom:
            ax_bar.set_xticks(range(n_other))
            ax_bar.set_xticklabels(other_names, rotation=45, ha='right', fontsize=10)
        else:
            ax_bar.set_xticks([])
        sns.despine(ax=ax_bar, left=True)

        # circle: size = importance, colour = output weight
        ax_circ = axes[row, 2]
        size = min_size + (max_size - min_size) * imp_norm[unit_idx]
        colour = cmap(0.5 + output_w[unit_idx] / (2 * out_abs_max))
        ax_circ.scatter(0.5, 0.5, s=size, color=colour,
                        edgecolor='k', linewidth=0.5)
        ax_circ.set_xlim(0, 1)
        ax_circ.set_ylim(0, 1)
        ax_circ.set_xticks([])
        ax_circ.set_yticks([])
        ax_circ.axis('off')

    # colourbar for output weight
    sm = plt.cm.ScalarMappable(cmap=cmap,
        norm=plt.Normalize(-out_abs_max, out_abs_max))
    cbar = fig.colorbar(sm, ax=axes[:, 2].tolist(), fraction=0.3,
                        pad=0.1, shrink=0.5)
    cbar.set_label('Output weight', fontsize=12)

    fig.suptitle(f'{animal} — {arch} (fold {unit_result["fold_idx"]})', fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


def plot_unit_summary(unit_result, animal, arch, top_n=8, save_path='default'):
    """input weight profiles and output weight for the top-N most important units

    per unit row: stimulus filter (left), non-stimulus weights (right)
    """
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, f'unit_summary_{animal}_{arch}.png')

    importance = unit_result['importance']
    input_w = unit_result['input_weights']
    output_w = unit_result['output_weights']

    top_n = min(top_n, len(importance))
    order = np.argsort(importance)[::-1][:top_n]
    t_ax = np.arange(N_TF_HIST) * BIN_WIDTH - 2.0

    other_features = _non_stimulus_features()
    other_names = list(other_features.keys())

    fig, axes = plt.subplots(top_n, 2, figsize=(10, 2.5 * top_n),
                             gridspec_kw={'width_ratios': [2, 1]})
    if top_n == 1:
        axes = axes[None, :]

    for row, unit_idx in enumerate(order):
        w = input_w[unit_idx]

        # stimulus filter
        axes[row, 0].plot(t_ax, w[:N_TF_HIST])
        axes[row, 0].axhline(0, color='k', linewidth=0.5)
        sign = '+' if output_w[unit_idx] > 0 else ''
        axes[row, 0].set_ylabel(
            f'unit {unit_idx}\n'
            f'imp={importance[unit_idx]:.4f}\n'
            f'out_w={sign}{output_w[unit_idx]:.2f}',
            fontsize=7)
        if row == top_n - 1:
            axes[row, 0].set_xlabel('Time before current bin (s)')

        # non-stimulus weights
        bar_vals = [w[other_features[name]].mean() for name in other_names]
        colours = ['tab:red' if v > 0 else 'tab:blue' for v in bar_vals]
        axes[row, 1].barh(range(len(other_names)), bar_vals, color=colours)
        axes[row, 1].axvline(0, color='k', linewidth=0.5)
        axes[row, 1].set_yticks(range(len(other_names)))
        axes[row, 1].set_yticklabels(other_names, fontsize=7)

    fig.suptitle(f'{animal} — {arch} (fold {unit_result["fold_idx"]})', fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


def plot_unit_responses(mouse, all_res, arch, unit_result, sess_idx=None,
                        top_n=3, n_trials=5, save_path='default'):
    """hidden unit activations and full model prediction for example lick trials

    uses the same fold as the unit analysis. shows trials from that fold's
    held-out session with the highest peak activation of the top unit.
    """
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR,
            f'unit_responses_{mouse["animal"]}_{arch}.png')

    fold_idx = unit_result['fold_idx']
    if sess_idx is None:
        sess_idx = fold_idx

    model, X_norm, y, trial_ids = _get_fold_model_and_data(
        mouse, all_res, arch, sess_idx)

    importance = unit_result['importance']
    top_n = min(top_n, len(importance))
    top_units = np.argsort(importance)[::-1][:top_n]

    hidden = _get_hidden_activations(model, X_norm)
    y_pred = _predict_from_hidden(model, hidden)

    # pick lick trials with highest peak activation of the top unit
    unique_trials = np.unique(trial_ids)
    lick_trials = [tr for tr in unique_trials if y[trial_ids == tr].max() > 0]
    if not lick_trials:
        return None

    peak_act = np.array([hidden[trial_ids == tr, top_units[0]].max()
                         for tr in lick_trials])
    selected = [lick_trials[i] for i in np.argsort(peak_act)[::-1][:n_trials]]

    fig, axes = plt.subplots(len(selected), 1, figsize=(8, 2.5 * len(selected)),
                             sharex=False)
    if len(selected) == 1:
        axes = [axes]

    unit_colours = plt.cm.tab10(np.linspace(0, 1, top_n))

    for row, tr in enumerate(selected):
        mask = trial_ids == tr
        bins = np.arange(mask.sum()) * BIN_WIDTH

        ax = axes[row]
        ax.plot(bins, y[mask], color='grey', alpha=0.4, label='target')
        ax.plot(bins, y_pred[mask], color='k', linewidth=1.5, label='model')
        for i, unit_idx in enumerate(top_units):
            ax.plot(bins, hidden[mask, unit_idx], color=unit_colours[i],
                    label=f'unit {unit_idx}', linewidth=1.0, alpha=0.8)

        ax.set_ylabel(f'trial {int(tr)}', fontsize=8)
        if row == 0:
            ax.legend(fontsize=7, loc='upper right')
        if row == len(selected) - 1:
            ax.set_xlabel('Time in trial (s)')
        sns.despine(ax=ax)

    sess_name = mouse['session_names'][sess_idx]
    fig.suptitle(f'{mouse["animal"]} — {sess_name} — {arch} '
                 f'(fold {fold_idx})', fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=400)
    return fig


def plot_unit_ablation_trials(mouse, all_res, arch, unit_result,
                              sess_idx=None, lick_only=True,
                              save_path='default'):
    """per-trial model output with each hidden unit ablated

    one pdf, one page per trial. each page shows target, full model prediction,
    and prediction with each unit zeroed out.
    """
    if save_path == 'default':
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR,
            f'unit_ablation_trials_{mouse["animal"]}_{arch}.pdf')

    fold_idx = unit_result['fold_idx']
    if sess_idx is None:
        sess_idx = fold_idx

    model, X_norm, y, trial_ids = _get_fold_model_and_data(
        mouse, all_res, arch, sess_idx)
    outcomes = mouse['trial_outcomes'][sess_idx]

    hidden = _get_hidden_activations(model, X_norm)
    y_pred_full = _predict_from_hidden(model, hidden)

    importance = unit_result['importance']
    n_hidden = len(importance)
    order = np.argsort(importance)[::-1]

    # precompute ablated predictions for each unit
    ablated_preds = {}
    for unit_idx in range(n_hidden):
        hidden_abl = hidden.copy()
        hidden_abl[:, unit_idx] = 0.0
        ablated_preds[unit_idx] = _predict_from_hidden(model, hidden_abl)

    all_trial_ids = np.unique(trial_ids)
    if lick_only:
        all_trial_ids = [tr for tr in all_trial_ids
                         if y[trial_ids == tr].max() > 0]

    unit_colours = plt.cm.tab10(np.linspace(0, 1, min(n_hidden, 10)))

    with PdfPages(save_path) as pdf:
        for tr in all_trial_ids:
            mask = trial_ids == tr
            if mask.sum() < 2:
                continue
            bins = np.arange(mask.sum()) * BIN_WIDTH

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(bins, y[mask], color='grey', alpha=0.4, label='target')
            ax.plot(bins, y_pred_full[mask], color='tab:red', linewidth=2,
                    label='full model')

            for rank, unit_idx in enumerate(order):
                ax.plot(bins, ablated_preds[unit_idx][mask],
                        color=unit_colours[rank % len(unit_colours)],
                        linewidth=0.8, alpha=0.7,
                        label=f'no unit {unit_idx} (imp={importance[unit_idx]:.3f})')

            outcome = outcomes.get(tr, '?')
            ax.set_title(f'Trial {int(tr)} ({outcome})')
            ax.set_xlabel('Time in trial (s)')
            ax.set_ylabel('P(lick)')
            ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc='upper left',
                      borderaxespad=0)
            sns.despine(ax=ax)
            fig.tight_layout()

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f'Saved {len(all_trial_ids)} trial plots to {save_path}')


if __name__ == '__main__':
    #%% run
    all_res = load_results()
    mice = {a: load_mouse(a, all_res) for a in all_res}
    arch = 'h8'
    # arch = _best_network_arch(all_res[list(mice.keys())[0]]['full_results'])
    unit_results = {a: analyse_hidden_units(mice[a], all_res, arch) for a in mice}

    #%% unit importance + weights (per mouse)
    for animal in mice:
        plot_unit_summary(unit_results[animal], animal, arch)

    #%% unit response profiles
    animal = list(mice.keys())[0]
    plot_unit_responses(mice[animal], all_res, arch, unit_results[animal])

    #%% per-trial unit ablation
    for animal in mice:
        plot_unit_ablation_trials(mice[animal], all_res, arch, unit_results[animal])
