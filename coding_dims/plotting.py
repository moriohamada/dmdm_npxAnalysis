"""plotting for coding dimension analyses"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

sns.set_style("whitegrid")

from config import PATHS, PLOT_OPTIONS


EARLY_COL = PLOT_OPTIONS['colours']['block']['early']
LATE_COL = PLOT_OPTIONS['colours']['block']['late']
FAST_COL = PLOT_OPTIONS['colours']['tf_pref']['fast']
SLOW_COL = PLOT_OPTIONS['colours']['tf_pref']['slow']


#%% TF coding dimensions

def plot_tf_dimensions(npx_dir=PATHS['npx_dir_local'], save_dir=None):
    """
    plot between-block cosine similarity of TF coding directions (with null),
    and time-resolved projections of fast and slow TF responses onto TF dims.
    """
    with open(Path(npx_dir) / 'coding_dims' / 'tf_dimensions.pkl', 'rb') as f:
        results = pickle.load(f)

    animals = list(results.keys())
    if not animals:
        return

    sample = results[animals[0]]
    window_labels = list(sample['dimensions']['early'].keys())
    n_wins = len(window_labels)
    tf_t_ax = sample['tf_t_ax']

    fig, axes = plt.subplots(2, max(n_wins, 1), figsize=(7 * max(n_wins, 1), 10),
                             squeeze=False)

    # row 0: between-block cosine similarity + null distribution
    for wi, wl in enumerate(window_labels):
        ax = axes[0, wi]
        real_vals = []
        null_all = []
        for animal in animals:
            bc = results[animal]['between_block_cosine'].get(wl)
            if bc is None:
                continue
            real_vals.append(bc['real'])
            null_all.append(bc['null'])

        if not real_vals:
            continue

        null_flat = np.concatenate(null_all)
        ax.hist(null_flat[~np.isnan(null_flat)], bins=30,
                weights=np.ones(np.sum(~np.isnan(null_flat))) / np.sum(~np.isnan(null_flat)),
                color='grey', alpha=0.5, label='Null')

        for rv in real_vals:
            ax.axvline(rv, color='black', linewidth=1, alpha=0.4)
        mean_real = np.nanmean(real_vals)
        ax.axvline(mean_real, color='red', linewidth=2,
                   label=f'Mean = {mean_real:.3f}')

        p = np.mean([np.nanmean(n >= r) for n, r in zip(null_all, real_vals)])
        ax.set_title(f'TF dim {wl}\nbetween-block cosine (p={p:.3f})')
        ax.set_xlabel('Cosine similarity')
        ax.set_ylabel('Probability')
        ax.legend(fontsize=7)

    # row 1: time-resolved projections (same-block), fast and slow separate
    for wi, wl in enumerate(window_labels):
        ax = axes[1, wi]
        for block, block_col in [('early', EARLY_COL), ('late', LATE_COL)]:
            for polarity, pol_col, ls in [('fast', FAST_COL, '-'),
                                           ('slow', SLOW_COL, '--')]:
                traces = []
                for animal in animals:
                    proj = results[animal]['cross_projections'][block][block].get(wl)
                    if proj is None:
                        continue
                    trace = proj[polarity]
                    ax.plot(tf_t_ax, trace, color=pol_col, alpha=0.1, linewidth=0.5)
                    traces.append(trace)
                if traces:
                    mean_proj = np.nanmean(traces, axis=0)
                    ax.plot(tf_t_ax, mean_proj, color=pol_col, linewidth=2,
                            linestyle=ls, label=f'{block} {polarity}')

        ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_xlabel('Post-pulse time (s)')
        ax.set_ylabel('Projection (a.u.)')
        ax.set_title(f'TF resp onto TF dim {wl}\n(same-block)')
        ax.legend(fontsize=7)

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / 'tf_dimensions.png', dpi=300, bbox_inches='tight')

    return fig


#%% motor coding dimensions

def plot_motor_dimensions(npx_dir=PATHS['npx_dir_local'], save_dir=None):
    """
    plot between-block cosine similarity of motor coding directions (with null),
    and time-resolved projections of lick activity onto motor dims.
    """
    with open(Path(npx_dir) / 'coding_dims' / 'motor_dimensions.pkl', 'rb') as f:
        results = pickle.load(f)

    animals = list(results.keys())
    if not animals:
        return

    sample = results[animals[0]]
    window_labels = list(sample['dimensions']['early'].keys())
    n_wins = len(window_labels)
    lick_t_ax = sample['lick_t_ax']

    fig, axes = plt.subplots(2, max(n_wins, 1), figsize=(7 * max(n_wins, 1), 10),
                             squeeze=False)

    # row 0: between-block cosine similarity + null
    for wi, wl in enumerate(window_labels):
        ax = axes[0, wi]
        real_vals = []
        null_all = []
        for animal in animals:
            bc = results[animal]['between_block_cosine'].get(wl)
            if bc is None:
                continue
            real_vals.append(bc['real'])
            null_all.append(bc['null'])

        if not real_vals:
            continue

        null_flat = np.concatenate(null_all)
        ax.hist(null_flat[~np.isnan(null_flat)], bins=30,
                weights=np.ones(np.sum(~np.isnan(null_flat))) / np.sum(~np.isnan(null_flat)),
                color='grey', alpha=0.5, label='Null')

        for rv in real_vals:
            ax.axvline(rv, color='black', linewidth=1, alpha=0.4)
        mean_real = np.nanmean(real_vals)
        ax.axvline(mean_real, color='red', linewidth=2,
                   label=f'Mean = {mean_real:.3f}')

        p = np.mean([np.nanmean(n >= r) for n, r in zip(null_all, real_vals)])
        ax.set_title(f'Motor dim {wl}\nbetween-block cosine (p={p:.3f})')
        ax.set_xlabel('Cosine similarity')
        ax.set_ylabel('Probability')
        ax.legend(fontsize=7)

    # row 1: time-resolved lick projections (same-block)
    for wi, wl in enumerate(window_labels):
        ax = axes[1, wi]
        for block, colour in [('early', EARLY_COL), ('late', LATE_COL)]:
            traces = []
            for animal in animals:
                proj = results[animal]['cross_projections'][block][block].get(wl)
                if proj is not None:
                    ax.plot(lick_t_ax, proj, color=colour, alpha=0.2, linewidth=0.7)
                    traces.append(proj)
            if traces:
                mean_proj = np.nanmean(traces, axis=0)
                ax.plot(lick_t_ax, mean_proj, color=colour, linewidth=2,
                        label=f'{block} block')

        ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_xlabel('Time from lick (s)')
        ax.set_ylabel('Projection (a.u.)')
        ax.set_title(f'Lick activity onto motor dim {wl}\n(same-block)')
        ax.legend(fontsize=7)

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / 'motor_dimensions.png', dpi=300, bbox_inches='tight')

    return fig


#%% cross-type analysis

def plot_cross_type(npx_dir=PATHS['npx_dir_local'], save_dir=None):
    """
    plot cross-type cosine similarity: TF dims vs motor dims, per block.
    """
    with open(Path(npx_dir) / 'coding_dims' / 'cross_type_analysis.pkl', 'rb') as f:
        results = pickle.load(f)

    animals = list(results.keys())
    if not animals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for bi, (block, colour) in enumerate([('early', EARLY_COL), ('late', LATE_COL)]):
        ax = axes[bi]
        all_keys = set()
        for animal in animals:
            all_keys.update(results[animal]['cross_cosine'].get(block, {}).keys())
        all_keys = sorted(all_keys)

        if not all_keys:
            continue

        vals_per_key = {k: [] for k in all_keys}
        for animal in animals:
            cc = results[animal]['cross_cosine'].get(block, {})
            for k in all_keys:
                if k in cc:
                    vals_per_key[k].append(cc[k])

        means = [np.nanmean(vals_per_key[k]) for k in all_keys]
        x = np.arange(len(all_keys))

        for animal in animals:
            cc = results[animal]['cross_cosine'].get(block, {})
            y = [cc.get(k, np.nan) for k in all_keys]
            ax.scatter(x, y, color=colour, alpha=0.3, s=20)

        ax.bar(x, means, color=colour, alpha=0.4, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace('_x_', '\nvs\n') for k in all_keys],
                           fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Cosine similarity')
        ax.set_title(f'{block.capitalize()} block: TF vs motor dim alignment')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / 'cross_type_cosine.png', dpi=300, bbox_inches='tight')

    return fig
