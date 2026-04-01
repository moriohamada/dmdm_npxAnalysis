"""plotting for coding dimension analyses"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

sns.set_style('whitegrid')

from config import PATHS, PLOT_OPTIONS, CODING_DIM_OPS
from coding_dims.extract import _file_suffix
from utils.time import window_label



EARLY_COL = PLOT_OPTIONS['colours']['block']['early']
LATE_COL = PLOT_OPTIONS['colours']['block']['late']
FAST_COL = PLOT_OPTIONS['colours']['tf_pref']['fast']
SLOW_COL = PLOT_OPTIONS['colours']['tf_pref']['slow']


def _load_results(npx_dir, dim_type, area, unit_filter):
    suffix = _file_suffix(area, unit_filter)
    path = Path(npx_dir) / 'coding_dims' / f'{dim_type}_dimensions_{suffix}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f), suffix


#%% between-block cosine: per-animal figures

def _plot_between_block_per_animal(results, dim_type, suffix, save_dir=None):
    """one subplot per animal per window - cosine sim (with null dist)"""
    animals = sorted(results.keys())
    if not animals:
        return

    sample = results[animals[0]]
    window_labels = sorted(sample['between_block_cosine'].keys())
    n_wins = len(window_labels)

    fig, axes = plt.subplots(len(animals), max(n_wins, 1),
                             figsize=(5 * max(n_wins, 1), 3 * len(animals)),
                             squeeze=False)

    for ai, animal in enumerate(animals):
        for wi, wl in enumerate(window_labels):
            ax = axes[ai, wi]
            bc = results[animal]['between_block_cosine'].get(wl)
            if bc is None:
                ax.set_visible(False)
                continue

            null = bc['null']
            valid_null = null[~np.isnan(null)]
            real = bc['real']
            p = np.mean(valid_null <= real) if len(valid_null) > 0 else np.nan

            ax.hist(valid_null, bins=30, density=True,
                    color='grey', alpha=0.5)
            ax.axvline(real, color='black', linewidth=2)
            ax.set_title(f'{animal}  (cos={real:.3f}, p={p:.3f})', fontsize=8)
            if ai == len(animals) - 1:
                ax.set_xlabel('Cosine similarity')
            if wi == 0:
                ax.set_ylabel('Density')
            if ai == 0:
                ax.set_title(f'{wl}\n{animal}  (cos={real:.3f}, p={p:.3f})', fontsize=8)

    fig.suptitle(f'{dim_type} coding dim between-block consistency [{suffix}]', fontsize=11)
    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f'{dim_type}_between_block_per_animal_{suffix}.png',
                    dpi=300, bbox_inches='tight')
    return fig


def _plot_between_block_summary(results, dim_type, suffix, save_dir=None):
    """
    summary: per-animal real cosines + grand mean against resampled grand-average null
    distribution
    """
    from coding_dims.analysis import pooled_null_test

    animals = sorted(results.keys())
    if not animals:
        return

    sample = results[animals[0]]
    window_labels = sorted(sample['between_block_cosine'].keys())
    n_wins = len(window_labels)

    pooled = pooled_null_test(results, n_perm=10000)

    fig, axes = plt.subplots(1, max(n_wins, 1),
                             figsize=(6 * max(n_wins, 1), 4),
                             squeeze=False)

    for wi, wl in enumerate(window_labels):
        ax = axes[0, wi]
        po = pooled.get(wl)
        if po is None:
            ax.set_visible(False)
            continue

        # null distribution of population means
        valid_null = po['null_means'][~np.isnan(po['null_means'])]
        ax.hist(valid_null, bins=40, density=True,
                color='grey', alpha=0.4, label='Null (resampled)')

        # individual animal cosines
        real_vals = []
        for animal in animals:
            bc = results[animal]['between_block_cosine'].get(wl)
            if bc is not None:
                real_vals.append(bc['real'])
                ax.axvline(bc['real'], color='black', linewidth=0.8, alpha=0.4)

        # grand mean
        grand_mean = np.nanmean(real_vals)
        ax.axvline(grand_mean, color='red', linewidth=2.5,
                   label=f'Grand mean = {grand_mean:.3f}')

        ax.set_title(f'{wl}  (p={po["p_value"]:.4f}, n={po["n_animals"]})')
        ax.set_xlabel('Cosine similarity')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    fig.suptitle(f'{dim_type} coding dim between-block consistency [{suffix}]', fontsize=11)
    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f'{dim_type}_between_block_summary_{suffix}.png',
                    dpi=300, bbox_inches='tight')
    return fig


#%% TF and motor dimension plots

def plot_tf_dimensions(npx_dir=PATHS['npx_dir_local'], save_dir=PATHS['plots_dir'],
                       area=None, unit_filter=None):
    """
    between-block consistency of TF coding directions + projected tf resps
    """
    results, suffix = _load_results(npx_dir, 'tf', area, unit_filter)
    save_dir = Path(save_dir) / 'coding_dims'
    save_dir.mkdir(parents=True, exist_ok=True)

    _plot_between_block_per_animal(results, 'tf', suffix, save_dir)
    _plot_between_block_summary(results, 'tf', suffix, save_dir)

    # time-resolved projections (same-block)
    animals = sorted(results.keys())
    if not animals:
        return

    sample = results[animals[0]]
    window_labels = sorted(sample['dimensions']['early'].keys())
    n_wins = len(window_labels)
    tf_t_ax = sample['tf_t_ax']

    fig, axes = plt.subplots(1, max(n_wins, 1),
                             figsize=(6 * max(n_wins, 1), 4),
                             squeeze=False)

    for wi, wl in enumerate(window_labels):
        ax = axes[0, wi]
        for block, block_col in [('early', EARLY_COL), ('late', LATE_COL)]:
            for polarity, pol_col, ls in [('fast', FAST_COL, '-'),
                                           ('slow', SLOW_COL, '--')]:
                traces = []
                for animal in animals:
                    proj = results[animal]['cross_projections'][block][block].get(wl)
                    if proj is None:
                        continue
                    trace = proj[polarity]
                    ax.plot(tf_t_ax, trace, color=block_col, alpha=0.15,
                            linewidth=0.5, linestyle=ls)
                    traces.append(trace)
                if traces:
                    mean_proj = np.nanmean(traces, axis=0)
                    ax.plot(tf_t_ax, mean_proj, color=block_col, linewidth=2,
                            linestyle=ls, label=f'{block} {polarity}')

        ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_xlabel('Time from TF pulse (s)')
        ax.set_ylabel('Projection (a.u.)')
        ax.set_title(f'TF onto TF dim {wl} (same-block)')
        ax.legend(fontsize=7)

    fig.suptitle(f'TF projections [{suffix}]', fontsize=11)
    plt.tight_layout()
    fig.savefig(save_dir / f'tf_projections_{suffix}.png',
                dpi=300, bbox_inches='tight')
    return fig


def plot_motor_dimensions(npx_dir=PATHS['npx_dir_local'], save_dir=PATHS['plots_dir'],
                          area=None, unit_filter=None):
    """between-block consistency of motor coding directions + projected lick resps"""
    results, suffix = _load_results(npx_dir, 'motor', area, unit_filter)
    save_dir = Path(save_dir) / 'coding_dims'
    save_dir.mkdir(parents=True, exist_ok=True)

    _plot_between_block_per_animal(results, 'motor', suffix, save_dir)
    _plot_between_block_summary(results, 'motor', suffix, save_dir)

    # projections (same-block)
    animals = sorted(results.keys())
    if not animals:
        return

    sample = results[animals[0]]
    window_labels = sorted(sample['dimensions']['early'].keys())
    n_wins = len(window_labels)
    lick_t_ax = sample['lick_t_ax']

    fig, axes = plt.subplots(1, max(n_wins, 1),
                             figsize=(6 * max(n_wins, 1), 4),
                             squeeze=False)

    for wi, wl in enumerate(window_labels):
        ax = axes[0, wi]
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
        ax.set_title(f'Lick onto motor dim {wl} (same-block)')
        ax.legend(fontsize=7)

    fig.suptitle(f'Motor projections [{suffix}]', fontsize=11)
    plt.tight_layout()
    fig.savefig(save_dir / f'motor_projections_{suffix}.png',
                dpi=300, bbox_inches='tight')
    return fig


#%% alignment plots

def plot_alignment(npx_dir=PATHS['npx_dir_local'], save_dir=PATHS['plots_dir'],
                   area=None, unit_filter=None):
    """early vs late alignment scatter + TF onto motor dim projections"""
    suffix = _file_suffix(area, unit_filter)
    save_dir = Path(save_dir) / 'coding_dims'
    save_dir.mkdir(parents=True, exist_ok=True)

    path = Path(npx_dir) / 'coding_dims' / f'alignment_{suffix}.pkl'
    with open(path, 'rb') as f:
        results = pickle.load(f)

    animals = sorted(results.keys())
    if not animals:
        return

    # dimensions needed for null cloud (shuffle neuron identity)
    tf_results = _load_results(npx_dir, 'tf', area, unit_filter)[0]
    motor_results = _load_results(npx_dir, 'motor', area, unit_filter)[0]

    # one figure per TF x motor window combo
    sample = results[animals[0]]
    combo_keys = sorted(sample['alignment'].get('early', {}).keys())
    tf_t_ax = sample.get('tf_t_ax')

    figs = []
    for combo_key in combo_keys:
        # parse window labels from key like 'tf_0.10_0.30_x_motor_-1.00_-0.60'
        tf_wl = combo_key.split('_x_motor_')[0].replace('tf_', '')
        motor_wl = combo_key.split('_x_motor_')[1]

        # collect early/late alignment per animal
        early_vals, late_vals = [], []
        for animal in animals:
            aln = results[animal]['alignment']
            e = aln.get('early', {}).get(combo_key)
            l = aln.get('late', {}).get(combo_key)
            if e is not None and l is not None:
                early_vals.append(e)
                late_vals.append(l)

        early_vals = np.array(early_vals)
        late_vals = np.array(late_vals)

        # null: shuffle neuron identity in TF dim, recompute alignment with motor dim
        from coding_dims.extract import cosine_similarity
        rng = np.random.default_rng(0)
        n_null = 500
        null_early_pts, null_late_pts = [], []
        for animal in animals:
            tf_dims = tf_results.get(animal, {}).get('dimensions', {})
            motor_dims_a = motor_results.get(animal, {}).get('dimensions', {})
            for block, null_list in [('early', null_early_pts),
                                      ('late', null_late_pts)]:
                tf_w = tf_dims.get(block, {}).get(tf_wl)
                motor_w = motor_dims_a.get(block, {}).get(motor_wl)
                if tf_w is None or motor_w is None:
                    continue
                for _ in range(n_null):
                    shuf = tf_w[rng.permutation(len(tf_w))]
                    null_list.append(cosine_similarity(shuf, motor_w))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # early vs late alignment scatter
        ax = axes[0]

        # null cloud (random direction vs motor dim)
        n_null_pts = min(len(null_early_pts), len(null_late_pts))
        if n_null_pts > 0:
            ax.scatter(null_early_pts[:n_null_pts], null_late_pts[:n_null_pts],
                       color='grey', alpha=0.05, s=10, zorder=1, rasterized=True)

        # data points
        ax.scatter(early_vals, late_vals, color='black', s=40, zorder=5,
                   label='Animals')

        # grand mean + 95% CIs
        mean_e = np.nanmean(early_vals)
        mean_l = np.nanmean(late_vals)
        ci_e = 1.96 * np.nanstd(early_vals) / np.sqrt(len(early_vals))
        ci_l = 1.96 * np.nanstd(late_vals) / np.sqrt(len(late_vals))
        ax.errorbar(mean_e, mean_l, xerr=ci_e, yerr=ci_l,
                    color='red', markersize=10, marker='o', linewidth=2,
                    zorder=10, label='Grand mean')

        # diagonal CI
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.3)

        # off-diagonal deviation: CI of (early - late)
        diffs = early_vals - late_vals
        mean_diff = np.nanmean(diffs)
        ci_diff = 1.96 * np.nanstd(diffs) / np.sqrt(len(diffs))
        ax.set_title(f'TF-motor alignment\n'
                     f'diff = {mean_diff:.3f} [{mean_diff-ci_diff:.3f}, {mean_diff+ci_diff:.3f}]',
                     fontsize=9)

        ax.set_xlabel(f'Early block alignment')
        ax.set_ylabel(f'Late block alignment')
        ax.legend(fontsize=7)
        ax.set_aspect('equal')

        #  tf resp onto motor dim, per block
        for pi, (block, block_col) in enumerate([('early', EARLY_COL),
                                                   ('late', LATE_COL)]):
            ax = axes[1 + pi]

            for cond_suffix, colour, ls, label in [
                ('_pos', FAST_COL, '-', 'fast'),
                ('_neg', SLOW_COL, '--', 'slow'),
            ]:
                # find matching TF conditions for this block
                block_prefix = 'earlyBlock' if block == 'early' else 'lateBlock'
                cond_key = f'tf/{block_prefix}_early{cond_suffix}'

                traces = []
                for animal in animals:
                    tom = results[animal].get('tf_onto_motor', {})
                    block_tom = tom.get(block, {})
                    motor_tom = block_tom.get(motor_wl, {})
                    trace = motor_tom.get(cond_key)
                    if trace is not None and tf_t_ax is not None:
                        ax.plot(tf_t_ax, trace, color=colour, alpha=0.15,
                                linewidth=0.5, linestyle=ls)
                        traces.append(trace)

                if traces and tf_t_ax is not None:
                    mean_trace = np.nanmean(traces, axis=0)
                    ax.plot(tf_t_ax, mean_trace, color=colour, linewidth=2,
                            linestyle=ls, label=label)

            ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
            ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
            ax.set_xlabel('Time from TF pulse (s)')
            ax.set_ylabel('Projection onto motor dim (a.u.)')
            ax.set_title(f'{block.capitalize()} block TF onto {block} motor dim {motor_wl}')
            ax.legend(fontsize=7)

        fig.suptitle(f'TF {tf_wl} x motor {motor_wl} [{suffix}]', fontsize=11)
        plt.tight_layout()

        fig.savefig(save_dir / f'alignment_tf{tf_wl}_motor{motor_wl}_{suffix}.png',
                    dpi=300, bbox_inches='tight')

        figs.append(fig)

    return figs
