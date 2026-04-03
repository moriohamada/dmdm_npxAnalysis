"""plotting for coding dimension analyses"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
sns.set_style('whitegrid')

from config import PATHS, PLOT_OPTIONS, CODING_DIM_OPS, ANALYSIS_OPTIONS
from coding_dims.extract import _file_suffix
from utils.time import window_label
from utils.norm import baseline_subtract



EARLY_COL = PLOT_OPTIONS['colours']['block']['early']
LATE_COL = PLOT_OPTIONS['colours']['block']['late']
FAST_COL = PLOT_OPTIONS['colours']['tf_pref']['fast']
SLOW_COL = PLOT_OPTIONS['colours']['tf_pref']['slow']

# pre-event baseline windows for projection plots
TF_BL_WINDOW = ANALYSIS_OPTIONS['tf_context']
LICK_BL_WINDOW = ANALYSIS_OPTIONS['lick_bl']
BLON_BL_WINDOW = (-1.0, 0.0)


def _load_results(npx_dir, dim_type, area, unit_filter):
    suffix = _file_suffix(area, unit_filter)
    path = Path(npx_dir) / 'coding_dims' / f'{dim_type}_dimensions_{suffix}.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f), suffix


#%% between-block cosine: per-animal figures

def _plot_between_block_per_animal(results, dim_type, suffix, save_dir=None):
    """one subplot per animal per window - cosine sim with null dist"""
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
    summary: per-animal real cosines + grand mean against resampled grand-average null distribution
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

    fig, axes = plt.subplots(3, max(n_wins, 1),
                             figsize=(6 * max(n_wins, 1), 11),
                             squeeze=False)

    for wi, wl in enumerate(window_labels):
        # collect traces per animal
        animal_traces = {}
        for animal in animals:
            animal_traces[animal] = {}
            for block in ('early', 'late'):
                proj = results[animal]['cross_projections'][block][block].get(wl)
                if proj is not None:
                    animal_traces[animal][block] = {
                        'fast': proj['fast'], 'slow': proj['slow']}

        # raw projections
        ax = axes[0, wi]
        for block, block_col in [('early', EARLY_COL), ('late', LATE_COL)]:
            for polarity, ls in [('fast', '-'), ('slow', '--')]:
                traces = []
                for animal in animals:
                    trace = animal_traces[animal].get(block, {}).get(polarity)
                    if trace is not None:
                        ax.plot(tf_t_ax, trace, color=block_col, alpha=0.15,
                                linewidth=0.5, linestyle=ls)
                        traces.append(trace)
                if traces:
                    ax.plot(tf_t_ax, np.nanmean(traces, axis=0), color=block_col,
                            linewidth=2, linestyle=ls, label=f'{block} {polarity}')
        ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_ylabel('Projection (a.u.)')
        ax.set_title(f'TF onto TF dim {wl} (same-block)')
        ax.legend(fontsize=7)

        # baseline-subtracted
        ax = axes[1, wi]
        for block, block_col in [('early', EARLY_COL), ('late', LATE_COL)]:
            for polarity, ls in [('fast', '-'), ('slow', '--')]:
                traces = []
                for animal in animals:
                    trace = animal_traces[animal].get(block, {}).get(polarity)
                    if trace is not None:
                        sub = baseline_subtract(trace, tf_t_ax, TF_BL_WINDOW)
                        ax.plot(tf_t_ax, sub, color=block_col, alpha=0.15,
                                linewidth=0.5, linestyle=ls)
                        traces.append(sub)
                if traces:
                    ax.plot(tf_t_ax, np.nanmean(traces, axis=0), color=block_col,
                            linewidth=2, linestyle=ls, label=f'{block} {polarity}')
        ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_ylabel('Projection (bl-sub)')
        ax.set_title('Baseline-subtracted')
        ax.legend(fontsize=7)

        # (fast-slow)_early - (fast-slow)_late, baseline-subtracted
        ax = axes[2, wi]
        diffs = []
        for animal in animals:
            e = animal_traces[animal].get('early', {})
            l = animal_traces[animal].get('late', {})
            if all(k in e for k in ('fast', 'slow')) and \
               all(k in l for k in ('fast', 'slow')):
                e_fast = baseline_subtract(e['fast'], tf_t_ax, TF_BL_WINDOW)
                e_slow = baseline_subtract(e['slow'], tf_t_ax, TF_BL_WINDOW)
                l_fast = baseline_subtract(l['fast'], tf_t_ax, TF_BL_WINDOW)
                l_slow = baseline_subtract(l['slow'], tf_t_ax, TF_BL_WINDOW)
                diff = (e_fast - e_slow) - (l_fast - l_slow)
                ax.plot(tf_t_ax, diff, color='grey', alpha=0.3, linewidth=0.7)
                diffs.append(diff)
        if diffs:
            ax.plot(tf_t_ax, np.nanmean(diffs, axis=0), color='black',
                    linewidth=2, label=f'Mean (n={len(diffs)})')
        ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_xlabel('Time from TF pulse (s)')
        ax.set_ylabel('Δ projection')
        ax.set_title('(fast−slow)early − (fast−slow)late')
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

    fig, axes = plt.subplots(3, max(n_wins, 1),
                             figsize=(6 * max(n_wins, 1), 11),
                             squeeze=False)

    for wi, wl in enumerate(window_labels):
        # collect traces per animal
        animal_traces = {}
        for animal in animals:
            animal_traces[animal] = {}
            for block in ('early', 'late'):
                proj = results[animal]['cross_projections'][block][block].get(wl)
                if proj is not None:
                    animal_traces[animal][block] = proj

        # raw projections
        ax = axes[0, wi]
        for block, colour in [('early', EARLY_COL), ('late', LATE_COL)]:
            traces = []
            for animal in animals:
                trace = animal_traces[animal].get(block)
                if trace is not None:
                    ax.plot(lick_t_ax, trace, color=colour, alpha=0.2, linewidth=0.7)
                    traces.append(trace)
            if traces:
                ax.plot(lick_t_ax, np.nanmean(traces, axis=0), color=colour,
                        linewidth=2, label=f'{block} block')
        ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_ylabel('Projection (a.u.)')
        ax.set_title(f'Lick onto motor dim {wl} (same-block)')
        ax.legend(fontsize=7)

        # baseline-subtracted
        ax = axes[1, wi]
        for block, colour in [('early', EARLY_COL), ('late', LATE_COL)]:
            traces = []
            for animal in animals:
                trace = animal_traces[animal].get(block)
                if trace is not None:
                    sub = baseline_subtract(trace, lick_t_ax, LICK_BL_WINDOW)
                    ax.plot(lick_t_ax, sub, color=colour, alpha=0.2, linewidth=0.7)
                    traces.append(sub)
            if traces:
                ax.plot(lick_t_ax, np.nanmean(traces, axis=0), color=colour,
                        linewidth=2, label=f'{block} block')
        ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_ylabel('Projection (bl-sub)')
        ax.set_title('Baseline-subtracted')
        ax.legend(fontsize=7)

        # early - late per animal
        ax = axes[2, wi]
        diffs = []
        for animal in animals:
            e = animal_traces[animal].get('early')
            l = animal_traces[animal].get('late')
            if e is not None and l is not None:
                e_sub = baseline_subtract(e, lick_t_ax, LICK_BL_WINDOW)
                l_sub = baseline_subtract(l, lick_t_ax, LICK_BL_WINDOW)
                diff = e_sub - l_sub
                ax.plot(lick_t_ax, diff, color='grey', alpha=0.3, linewidth=0.7)
                diffs.append(diff)
        if diffs:
            ax.plot(lick_t_ax, np.nanmean(diffs, axis=0), color='black',
                    linewidth=2, label=f'Mean (n={len(diffs)})')
        ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_xlabel('Time from lick (s)')
        ax.set_ylabel('Δ projection')
        ax.set_title('Early − late')
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
        from utils.stats import cosine_similarity
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

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes[1, 0].set_visible(False)
        axes[2, 0].set_visible(False)
        axes[2, 2].set_visible(False)

        # early vs late alignment scatter
        ax = axes[0, 0]

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

        # collect TF-onto-motor traces per animal for all rows
        block_animal_traces = {}
        for animal in animals:
            block_animal_traces[animal] = {}
            for block in ('early', 'late'):
                block_animal_traces[animal][block] = {}
                block_prefix = 'earlyBlock' if block == 'early' else 'lateBlock'
                for cond_suffix, polarity in [('_pos', 'fast'), ('_neg', 'slow')]:
                    cond_key = f'tf/{block_prefix}_early{cond_suffix}'
                    trace = results[animal].get('tf_onto_motor', {}).get(
                        block, {}).get(motor_wl, {}).get(cond_key)
                    if trace is not None:
                        block_animal_traces[animal][block][polarity] = trace

        # raw projections
        for pi, (block, block_col) in enumerate([('early', EARLY_COL),
                                                   ('late', LATE_COL)]):
            ax = axes[0, 1 + pi]
            for polarity, colour, ls in [('fast', FAST_COL, '-'),
                                          ('slow', SLOW_COL, '--')]:
                traces = []
                for animal in animals:
                    trace = block_animal_traces[animal].get(block, {}).get(polarity)
                    if trace is not None and tf_t_ax is not None:
                        ax.plot(tf_t_ax, trace, color=colour, alpha=0.15,
                                linewidth=0.5, linestyle=ls)
                        traces.append(trace)
                if traces and tf_t_ax is not None:
                    ax.plot(tf_t_ax, np.nanmean(traces, axis=0), color=colour,
                            linewidth=2, linestyle=ls, label=polarity)
            ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
            ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
            ax.set_ylabel('Projection onto motor dim (a.u.)')
            ax.set_title(f'{block.capitalize()} block TF onto {block} motor dim {motor_wl}')
            ax.legend(fontsize=7)

        # baseline-subtracted projections
        for pi, (block, block_col) in enumerate([('early', EARLY_COL),
                                                   ('late', LATE_COL)]):
            ax = axes[1, 1 + pi]
            for polarity, colour, ls in [('fast', FAST_COL, '-'),
                                          ('slow', SLOW_COL, '--')]:
                traces = []
                for animal in animals:
                    trace = block_animal_traces[animal].get(block, {}).get(polarity)
                    if trace is not None and tf_t_ax is not None:
                        sub = baseline_subtract(trace, tf_t_ax, TF_BL_WINDOW)
                        ax.plot(tf_t_ax, sub, color=colour, alpha=0.15,
                                linewidth=0.5, linestyle=ls)
                        traces.append(sub)
                if traces and tf_t_ax is not None:
                    ax.plot(tf_t_ax, np.nanmean(traces, axis=0), color=colour,
                            linewidth=2, linestyle=ls, label=polarity)
            ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
            ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
            ax.set_ylabel('Projection (bl-sub)')
            ax.set_title(f'{block.capitalize()} block (bl-sub)')
            ax.legend(fontsize=7)

        # (fast−slow)_early − (fast−slow)_late
        ax = axes[2, 1]
        gain_diffs = []
        for animal in animals:
            e = block_animal_traces[animal].get('early', {})
            l = block_animal_traces[animal].get('late', {})
            if all(k in e for k in ('fast', 'slow')) and \
               all(k in l for k in ('fast', 'slow')) and tf_t_ax is not None:
                ef = baseline_subtract(e['fast'], tf_t_ax, TF_BL_WINDOW)
                es = baseline_subtract(e['slow'], tf_t_ax, TF_BL_WINDOW)
                lf = baseline_subtract(l['fast'], tf_t_ax, TF_BL_WINDOW)
                ls_sub = baseline_subtract(l['slow'], tf_t_ax, TF_BL_WINDOW)
                gain_diffs.append((ef - es) - (lf - ls_sub))
        for d in gain_diffs:
            ax.plot(tf_t_ax, d, color='grey', alpha=0.3, linewidth=0.7)
        if gain_diffs:
            ax.plot(tf_t_ax, np.nanmean(gain_diffs, axis=0), color='black',
                    linewidth=2, label=f'Mean (n={len(gain_diffs)})')
        ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax.set_xlabel('Time from TF pulse (s)')
        ax.set_ylabel('Δ projection')
        ax.set_title('(fast−slow)early − (fast−slow)late')
        ax.legend(fontsize=7)

        fig.suptitle(f'TF {tf_wl} x motor {motor_wl} [{suffix}]', fontsize=11)
        plt.tight_layout()

        fig.savefig(save_dir / f'alignment_tf{tf_wl}_motor{motor_wl}_{suffix}.png',
                    dpi=300, bbox_inches='tight')

        figs.append(fig)

    return figs


#%% block dimension significance plots

def plot_block_significance(npx_dir=PATHS['npx_dir_local'], save_dir=PATHS['plots_dir'],
                            area=None, unit_filter=None):
    """per-animal AUC null distributions + summary for block coding dimensions"""
    results, suffix = _load_results(npx_dir, 'block', area, unit_filter)
    save_dir = Path(save_dir) / 'coding_dims'
    save_dir.mkdir(parents=True, exist_ok=True)

    animals = sorted(results.keys())
    if not animals:
        return

    sample = results[animals[0]]
    window_labels = sorted(sample['real_aucs'].keys())
    n_wins = len(window_labels)

    # per-animal: one subplot per animal per window
    fig, axes = plt.subplots(len(animals), max(n_wins, 1),
                             figsize=(5 * max(n_wins, 1), 3 * len(animals)),
                             squeeze=False)

    for ai, animal in enumerate(animals):
        for wi, wl in enumerate(window_labels):
            ax = axes[ai, wi]
            real_auc = results[animal]['real_aucs'].get(wl)
            null = results[animal]['null_aucs'].get(wl)
            if real_auc is None or null is None:
                ax.set_visible(False)
                continue

            valid_null = null[~np.isnan(null)]
            p = np.mean(valid_null >= real_auc) if len(valid_null) > 0 else np.nan

            ax.hist(valid_null, bins=30, density=True, color='grey', alpha=0.5)
            ax.axvline(real_auc, color='black', linewidth=2)
            ax.set_title(f'{animal}  (AUC={real_auc:.3f}, p={p:.3f})', fontsize=8)
            if ai == len(animals) - 1:
                ax.set_xlabel('AUC')
            if wi == 0:
                ax.set_ylabel('Density')
            if ai == 0:
                ax.set_title(f'{wl}\n{animal}  (AUC={real_auc:.3f}, p={p:.3f})', fontsize=8)

    fig.suptitle(f'Block coding dim held-out AUC [{suffix}]', fontsize=11)
    plt.tight_layout()
    fig.savefig(save_dir / f'block_auc_per_animal_{suffix}.png',
                dpi=300, bbox_inches='tight')

    # summary: individual animal AUCs + across-animals null
    from coding_dims.analysis import analyse_block_dimensions
    block_stats = analyse_block_dimensions(npx_dir=npx_dir, area=area, unit_filter=unit_filter)

    fig_s, axes_s = plt.subplots(1, max(n_wins, 1),
                                 figsize=(6 * max(n_wins, 1), 4),
                                 squeeze=False)

    for wi, wl in enumerate(window_labels):
        ax = axes_s[0, wi]
        aa = block_stats['across_animals'].get(wl)
        po = block_stats['pooled'].get(wl)
        if aa is None:
            ax.set_visible(False)
            continue

        # across-animals null distribution
        valid_null = aa['null_means'][~np.isnan(aa['null_means'])]
        ax.hist(valid_null, bins=40, density=True, color='grey', alpha=0.4,
                label='Null (resampled)')

        # individual animal AUCs
        pa = block_stats['per_animal'].get(wl, {})
        animal_aucs = pa.get('aucs', [])
        for auc_val in animal_aucs:
            ax.axvline(auc_val, color='black', linewidth=0.8, alpha=0.4)

        # grand mean
        grand_mean = aa['observed_mean']
        ax.axvline(grand_mean, color='red', linewidth=2.5,
                   label=f'Grand mean = {grand_mean:.3f}')

        title = f'{wl}  (across p={aa["p_value"]:.4f}'
        if po is not None:
            title += f', pooled p={po["p_value"]:.4f}'
        title += f', n={aa["n_animals"]})'
        ax.set_title(title)
        ax.set_xlabel('AUC')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)

    fig_s.suptitle(f'Block coding dim summary [{suffix}]', fontsize=11)
    plt.tight_layout()
    fig_s.savefig(save_dir / f'block_auc_summary_{suffix}.png',
                  dpi=300, bbox_inches='tight')

    return fig, fig_s


#%% projections of all event types onto all dimensions

def plot_cross_projections(npx_dir=PATHS['npx_dir_local'], save_dir=PATHS['plots_dir'],
                           area=None, unit_filter=None):
    """
    project baseline, TF, and lick responses onto every coding dimension.
    one figure per dimension class (block, tf, motor), columns = windows,
    rows = event types. individual animals as thin lines, mean as thick
    """
    suffix = _file_suffix(area, unit_filter)
    save_dir = Path(save_dir) / 'coding_dims'
    save_dir.mkdir(parents=True, exist_ok=True)

    proj_path = Path(npx_dir) / 'coding_dims' / f'cross_dimension_projections_{suffix}.pkl'
    with open(proj_path, 'rb') as f:
        proj_results = pickle.load(f)

    animals = sorted(proj_results.keys())
    if not animals:
        return

    sample = proj_results[animals[0]]
    t_axes = sample['t_axes']

    # group dimensions by class
    dim_classes = {'block': [], 'tf': [], 'motor': []}
    for dn in sample['dim_names']:
        if dn.startswith('block_'):
            dim_classes['block'].append(dn)
        elif dn.startswith('tf_'):
            dim_classes['tf'].append(dn)
        elif dn.startswith('motor_'):
            dim_classes['motor'].append(dn)

    # each row: (label, event_type, baseline_window, conditions)
    event_rows = [
        ('baseline', 'blOn', BLON_BL_WINDOW, [
            ('early block', 'blOn/early', EARLY_COL, '-'),
            ('late block', 'blOn/late', LATE_COL, '-'),
        ]),
        ('TF pulse', 'tf', TF_BL_WINDOW, [
            ('early fast', 'tf/earlyBlock_early_pos', EARLY_COL, '-'),
            ('early slow', 'tf/earlyBlock_early_neg', EARLY_COL, '--'),
            ('late fast', 'tf/lateBlock_early_pos', LATE_COL, '-'),
            ('late slow', 'tf/lateBlock_early_neg', LATE_COL, '--'),
        ]),
        ('lick (FA)', 'lick', LICK_BL_WINDOW, [
            ('early block', 'lick/earlyBlock_early_fa', EARLY_COL, '-'),
            ('late block', 'lick/lateBlock_early_fa', LATE_COL, '-'),
        ]),
    ]

    figs = []
    for dim_class, dim_names in dim_classes.items():
        if not dim_names:
            continue

        n_cols = len(dim_names)
        n_rows = len(event_rows) * 3
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 3 * n_rows),
                                 squeeze=False)

        for ci, dim_name in enumerate(sorted(dim_names)):
            for ri, (row_label, event_type, bl_window, conditions) in enumerate(event_rows):
                base_row = ri * 3
                t_ax = t_axes.get(event_type)

                if t_ax is None:
                    for offset in range(3):
                        axes[base_row + offset, ci].set_visible(False)
                    continue

                # collect per-animal traces for this event type + dimension
                animal_traces = {}
                for animal in animals:
                    proj = proj_results[animal]['projections'].get(dim_name, {})
                    animal_traces[animal] = {
                        rk: proj[rk] for _, rk, _, _ in conditions if rk in proj}

                # raw projections
                ax = axes[base_row, ci]
                for label, rk, colour, ls in conditions:
                    traces = [animal_traces[a][rk] for a in animals
                              if rk in animal_traces[a]]
                    for t in traces:
                        ax.plot(t_ax, t, color=colour, alpha=0.15,
                                linewidth=0.5, linestyle=ls)
                    if traces:
                        ax.plot(t_ax, np.nanmean(traces, axis=0), color=colour,
                                linewidth=2, linestyle=ls, label=label)
                ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
                ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
                if ci == 0:
                    ax.set_ylabel(f'{row_label}\nraw')
                if base_row == 0:
                    ax.set_title(dim_name, fontsize=9)
                ax.legend(fontsize=6, loc='upper right')

                # baseline-subtracted
                ax = axes[base_row + 1, ci]
                for label, rk, colour, ls in conditions:
                    traces = [baseline_subtract(animal_traces[a][rk], t_ax, bl_window)
                              for a in animals if rk in animal_traces[a]]
                    for t in traces:
                        ax.plot(t_ax, t, color=colour, alpha=0.15,
                                linewidth=0.5, linestyle=ls)
                    if traces:
                        ax.plot(t_ax, np.nanmean(traces, axis=0), color=colour,
                                linewidth=2, linestyle=ls, label=label)
                ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
                ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
                if ci == 0:
                    ax.set_ylabel(f'{row_label}\nbl-sub')
                ax.legend(fontsize=6, loc='upper right')

                # early−late difference
                ax = axes[base_row + 2, ci]
                diffs = []
                if event_type == 'tf':
                    # (fast−slow)_early − (fast−slow)_late
                    keys = [rk for _, rk, _, _ in conditions]
                    for a in animals:
                        at = animal_traces[a]
                        if len(at) == 4:
                            bs = [baseline_subtract(at[k], t_ax, bl_window)
                                  for k in keys]
                            diffs.append((bs[0] - bs[1]) - (bs[2] - bs[3]))
                else:
                    keys = [rk for _, rk, _, _ in conditions]
                    for a in animals:
                        at = animal_traces[a]
                        if len(at) == 2:
                            diffs.append(
                                baseline_subtract(at[keys[0]], t_ax, bl_window) -
                                baseline_subtract(at[keys[1]], t_ax, bl_window))
                for d in diffs:
                    ax.plot(t_ax, d, color='grey', alpha=0.3, linewidth=0.7)
                if diffs:
                    ax.plot(t_ax, np.nanmean(diffs, axis=0), color='black',
                            linewidth=2, label=f'Mean (n={len(diffs)})')
                ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
                ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
                if ci == 0:
                    diff_label = 'Δ(fast−slow)' if event_type == 'tf' else 'early−late'
                    ax.set_ylabel(f'{row_label}\n{diff_label}')
                ax.set_xlabel('Time (s)')
                ax.legend(fontsize=6, loc='upper right')

        fig.suptitle(f'{dim_class} dimensions — all projections [{suffix}]', fontsize=11)
        plt.tight_layout()
        fig.savefig(save_dir / f'projections_{dim_class}_{suffix}.png',
                    dpi=300, bbox_inches='tight')
        figs.append(fig)

    return figs


#%% cross-class alignment scatters

def plot_cross_class_alignment(npx_dir=PATHS['npx_dir_local'], save_dir=PATHS['plots_dir'],
                               bm_ops=CODING_DIM_OPS,
                               area=None, unit_filter=None):
    """
    scatter plots of cross-class dimension alignment: early-block cosine vs
    late-block cosine for each pair of dimensions from different classes.
    one figure per dimension pair, with null cloud from neuron-shuffled cosines
    """
    from utils.stats import cosine_similarity

    suffix = _file_suffix(area, unit_filter)
    save_dir = Path(save_dir) / 'coding_dims' / 'alignment'
    save_dir.mkdir(parents=True, exist_ok=True)

    cos_path = Path(npx_dir) / 'coding_dims' / f'cross_dimension_cosines_{suffix}.pkl'
    with open(cos_path, 'rb') as f:
        cos_results = pickle.load(f)

    # load cross-dimension projections for projection panels
    proj_path = Path(npx_dir) / 'coding_dims' / f'cross_dimension_projections_{suffix}.pkl'
    with open(proj_path, 'rb') as f:
        proj_results = pickle.load(f)

    # load raw dimensions for null cloud
    block_results = _load_results(npx_dir, 'block', area, unit_filter)[0]
    tf_results = _load_results(npx_dir, 'tf', area, unit_filter)[0]
    motor_results = _load_results(npx_dir, 'motor', area, unit_filter)[0]

    animals = sorted(cos_results.keys())
    if not animals:
        return

    sample = cos_results[animals[0]]
    dim_names = sample['dim_names']

    # classify dimensions
    block_dims = sorted(d for d in dim_names if d.startswith('block_'))
    tf_windows = sorted(set(d.replace('tf_early_', '').replace('tf_late_', '')
                            for d in dim_names if d.startswith('tf_')))
    motor_windows = sorted(set(d.replace('motor_early_', '').replace('motor_late_', '')
                               for d in dim_names if d.startswith('motor_')))
    block_windows = sorted(set(d.replace('block_', '') for d in block_dims))

    rng = np.random.default_rng(0)
    n_null = 500
    figs = []

    # tf x block: scatter + TF pulse projections onto block dimension
    proj_animals = sorted(proj_results.keys())
    sample_proj = proj_results[proj_animals[0]] if proj_animals else None
    t_axes = sample_proj['t_axes'] if sample_proj else {}

    for tf_wl in tf_windows:
        for b_wl in block_windows:
            tf_e = f'tf_early_{tf_wl}'
            tf_l = f'tf_late_{tf_wl}'
            b_d = f'block_{b_wl}'
            if tf_e not in dim_names or tf_l not in dim_names or b_d not in dim_names:
                continue

            early_vals, late_vals = [], []
            null_early, null_late = [], []

            for animal in animals:
                dn = cos_results[animal]['dim_names']
                cm = cos_results[animal]['cosine_matrix']
                if tf_e not in dn or tf_l not in dn or b_d not in dn:
                    continue
                i_te, i_tl, i_b = dn.index(tf_e), dn.index(tf_l), dn.index(b_d)
                early_vals.append(cm[i_te, i_b])
                late_vals.append(cm[i_tl, i_b])

                tf_r = tf_results.get(animal, {})
                block_r = block_results.get(animal, {})
                tf_e_vec = tf_r.get('dimensions', {}).get('early', {}).get(tf_wl)
                tf_l_vec = tf_r.get('dimensions', {}).get('late', {}).get(tf_wl)
                block_vec = block_r.get('dimensions', {}).get(b_wl)
                if tf_e_vec is not None and tf_l_vec is not None and block_vec is not None:
                    n = min(len(tf_e_vec), len(block_vec))
                    for _ in range(n_null):
                        shuf = rng.permutation(n)
                        null_early.append(cosine_similarity(tf_e_vec[:n][shuf], block_vec[:n]))
                        null_late.append(cosine_similarity(tf_l_vec[:n][shuf], block_vec[:n]))

            if len(early_vals) < 2:
                continue

            fig, axes = plt.subplots(3, 3, figsize=(18, 14))
            axes[1, 0].set_visible(False)
            axes[2, 0].set_visible(False)
            axes[2, 2].set_visible(False)

            _alignment_scatter_ax(axes[0, 0],
                np.array(early_vals), np.array(late_vals),
                null_early, null_late,
                f'Early block: cos(TF {tf_wl}, block {b_wl})',
                f'Late block: cos(TF {tf_wl}, block {b_wl})')

            tf_t_ax = t_axes.get('tf')

            # collect traces per animal
            block_animal_traces = {}
            for animal in proj_animals:
                block_animal_traces[animal] = {}
                for block in ('early', 'late'):
                    block_animal_traces[animal][block] = {}
                    block_prefix = 'earlyBlock' if block == 'early' else 'lateBlock'
                    for pol_key, pol_name in [('pos', 'fast'), ('neg', 'slow')]:
                        resp_key = f'tf/{block_prefix}_early_{pol_key}'
                        trace = proj_results[animal]['projections'].get(
                            b_d, {}).get(resp_key)
                        if trace is not None:
                            block_animal_traces[animal][block][pol_name] = trace

            # raw projections
            for pi, (block, block_col) in enumerate([('early', EARLY_COL),
                                                      ('late', LATE_COL)]):
                ax = axes[0, 1 + pi]
                for polarity, ls, label in [('fast', '-', 'fast'), ('slow', '--', 'slow')]:
                    traces = []
                    for animal in proj_animals:
                        trace = block_animal_traces[animal].get(
                            block, {}).get(polarity)
                        if trace is not None and tf_t_ax is not None:
                            ax.plot(tf_t_ax, trace, color=block_col, alpha=0.15,
                                    linewidth=0.5, linestyle=ls)
                            traces.append(trace)
                    if traces and tf_t_ax is not None:
                        ax.plot(tf_t_ax, np.nanmean(traces, axis=0), color=block_col,
                                linewidth=2, linestyle=ls, label=label)
                ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
                ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
                ax.set_ylabel('Projection onto block dim (a.u.)')
                ax.set_title(f'{block} block TF onto block {b_wl}')
                ax.legend(fontsize=7)

            # baseline-subtracted
            for pi, (block, block_col) in enumerate([('early', EARLY_COL),
                                                      ('late', LATE_COL)]):
                ax = axes[1, 1 + pi]
                for polarity, ls, label in [('fast', '-', 'fast'), ('slow', '--', 'slow')]:
                    traces = []
                    for animal in proj_animals:
                        trace = block_animal_traces[animal].get(
                            block, {}).get(polarity)
                        if trace is not None and tf_t_ax is not None:
                            sub = baseline_subtract(trace, tf_t_ax, TF_BL_WINDOW)
                            ax.plot(tf_t_ax, sub, color=block_col, alpha=0.15,
                                    linewidth=0.5, linestyle=ls)
                            traces.append(sub)
                    if traces and tf_t_ax is not None:
                        ax.plot(tf_t_ax, np.nanmean(traces, axis=0), color=block_col,
                                linewidth=2, linestyle=ls, label=label)
                ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
                ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
                ax.set_ylabel('Projection (bl-sub)')
                ax.set_title(f'{block} block (bl-sub)')
                ax.legend(fontsize=7)

            # (fast−slow)_early − (fast−slow)_late
            ax = axes[2, 1]
            gain_diffs = []
            for animal in proj_animals:
                e = block_animal_traces[animal].get('early', {})
                l = block_animal_traces[animal].get('late', {})
                if all(k in e for k in ('fast', 'slow')) and \
                   all(k in l for k in ('fast', 'slow')) and tf_t_ax is not None:
                    ef = baseline_subtract(e['fast'], tf_t_ax, TF_BL_WINDOW)
                    es = baseline_subtract(e['slow'], tf_t_ax, TF_BL_WINDOW)
                    lf = baseline_subtract(l['fast'], tf_t_ax, TF_BL_WINDOW)
                    ls_sub = baseline_subtract(l['slow'], tf_t_ax, TF_BL_WINDOW)
                    gain_diffs.append((ef - es) - (lf - ls_sub))
            for d in gain_diffs:
                ax.plot(tf_t_ax, d, color='grey', alpha=0.3, linewidth=0.7)
            if gain_diffs:
                ax.plot(tf_t_ax, np.nanmean(gain_diffs, axis=0), color='black',
                        linewidth=2, label=f'Mean (n={len(gain_diffs)})')
            ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
            ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
            ax.set_xlabel('Time from TF pulse (s)')
            ax.set_ylabel('Δ projection')
            ax.set_title('(fast−slow)early − (fast−slow)late')
            ax.legend(fontsize=7)

            fig.suptitle(f'TF {tf_wl} x block {b_wl} [{suffix}]', fontsize=11)
            plt.tight_layout()
            fig.savefig(save_dir / f'align_tf{tf_wl}_block{b_wl}_{suffix}.png',
                        dpi=300, bbox_inches='tight')
            figs.append(fig)

    # motor x block: scatter + lick projections onto block dimension
    for m_wl in motor_windows:
        for b_wl in block_windows:
            m_e = f'motor_early_{m_wl}'
            m_l = f'motor_late_{m_wl}'
            b_d = f'block_{b_wl}'
            if m_e not in dim_names or m_l not in dim_names or b_d not in dim_names:
                continue

            early_vals, late_vals = [], []
            null_early, null_late = [], []

            for animal in animals:
                dn = cos_results[animal]['dim_names']
                cm = cos_results[animal]['cosine_matrix']
                if m_e not in dn or m_l not in dn or b_d not in dn:
                    continue
                i_me, i_ml, i_b = dn.index(m_e), dn.index(m_l), dn.index(b_d)
                early_vals.append(cm[i_me, i_b])
                late_vals.append(cm[i_ml, i_b])

                motor_r = motor_results.get(animal, {})
                block_r = block_results.get(animal, {})
                m_e_vec = motor_r.get('dimensions', {}).get('early', {}).get(m_wl)
                m_l_vec = motor_r.get('dimensions', {}).get('late', {}).get(m_wl)
                block_vec = block_r.get('dimensions', {}).get(b_wl)
                if m_e_vec is not None and m_l_vec is not None and block_vec is not None:
                    n = min(len(m_e_vec), len(block_vec))
                    for _ in range(n_null):
                        shuf = rng.permutation(n)
                        null_early.append(cosine_similarity(m_e_vec[:n][shuf], block_vec[:n]))
                        null_late.append(cosine_similarity(m_l_vec[:n][shuf], block_vec[:n]))

            if len(early_vals) < 2:
                continue

            fig, axes = plt.subplots(3, 3, figsize=(18, 14))
            axes[1, 0].set_visible(False)
            axes[2, 0].set_visible(False)
            axes[2, 2].set_visible(False)

            _alignment_scatter_ax(axes[0, 0],
                np.array(early_vals), np.array(late_vals),
                null_early, null_late,
                f'Early block: cos(motor {m_wl}, block {b_wl})',
                f'Late block: cos(motor {m_wl}, block {b_wl})')

            lick_t_ax = t_axes.get('lick')

            # collect FA lick traces per animal per block
            block_traces = {a: {} for a in proj_animals}
            for animal in proj_animals:
                for block in ('early', 'late'):
                    block_prefix = 'earlyBlock' if block == 'early' else 'lateBlock'
                    resp_key = f'lick/{block_prefix}_early_fa'
                    trace = proj_results[animal]['projections'].get(
                        b_d, {}).get(resp_key)
                    if trace is not None:
                        block_traces[animal][block] = trace

            # raw projections
            for pi, (block, block_col) in enumerate([('early', EARLY_COL),
                                                      ('late', LATE_COL)]):
                ax = axes[0, 1 + pi]
                traces = []
                for animal in proj_animals:
                    trace = block_traces[animal].get(block)
                    if trace is not None and lick_t_ax is not None:
                        ax.plot(lick_t_ax, trace, color=block_col, alpha=0.15,
                                linewidth=0.5)
                        traces.append(trace)
                if traces and lick_t_ax is not None:
                    ax.plot(lick_t_ax, np.nanmean(traces, axis=0), color=block_col,
                            linewidth=2, label=f'{block} block')
                ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
                ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
                ax.set_ylabel('Projection onto block dim (a.u.)')
                ax.set_title(f'{block} block FA licks onto block {b_wl}')
                ax.legend(fontsize=7)

            # baseline-subtracted
            for pi, (block, block_col) in enumerate([('early', EARLY_COL),
                                                      ('late', LATE_COL)]):
                ax = axes[1, 1 + pi]
                traces = []
                for animal in proj_animals:
                    trace = block_traces[animal].get(block)
                    if trace is not None and lick_t_ax is not None:
                        sub = baseline_subtract(trace, lick_t_ax, LICK_BL_WINDOW)
                        ax.plot(lick_t_ax, sub, color=block_col, alpha=0.15,
                                linewidth=0.5)
                        traces.append(sub)
                if traces and lick_t_ax is not None:
                    ax.plot(lick_t_ax, np.nanmean(traces, axis=0), color=block_col,
                            linewidth=2, label=f'{block} block')
                ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
                ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
                ax.set_ylabel('Projection (bl-sub)')
                ax.set_title(f'{block} block (bl-sub)')
                ax.legend(fontsize=7)

            # early − late
            ax = axes[2, 1]
            diffs = []
            for animal in proj_animals:
                e = block_traces[animal].get('early')
                l = block_traces[animal].get('late')
                if e is not None and l is not None and lick_t_ax is not None:
                    diffs.append(
                        baseline_subtract(e, lick_t_ax, LICK_BL_WINDOW) -
                        baseline_subtract(l, lick_t_ax, LICK_BL_WINDOW))
            for d in diffs:
                ax.plot(lick_t_ax, d, color='grey', alpha=0.3, linewidth=0.7)
            if diffs:
                ax.plot(lick_t_ax, np.nanmean(diffs, axis=0), color='black',
                        linewidth=2, label=f'Mean (n={len(diffs)})')
            ax.axvline(0, color='grey', linewidth=0.5, linestyle=':')
            ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
            ax.set_xlabel('Time from lick (s)')
            ax.set_ylabel('Δ projection')
            ax.set_title('Early − late (FA)')
            ax.legend(fontsize=7)

            fig.suptitle(f'Motor {m_wl} x block {b_wl} [{suffix}]', fontsize=11)
            plt.tight_layout()
            fig.savefig(save_dir / f'align_motor{m_wl}_block{b_wl}_{suffix}.png',
                        dpi=300, bbox_inches='tight')
            figs.append(fig)

    # tf x motor: both have early/late block versions
    for tf_wl in tf_windows:
        for m_wl in motor_windows:
            early_vals, late_vals = [], []
            null_early, null_late = [], []

            for animal in animals:
                dn = cos_results[animal]['dim_names']
                cm = cos_results[animal]['cosine_matrix']
                for block, vals_list in [('early', early_vals), ('late', late_vals)]:
                    tf_d = f'tf_{block}_{tf_wl}'
                    m_d = f'motor_{block}_{m_wl}'
                    if tf_d not in dn or m_d not in dn:
                        continue
                    vals_list.append(cm[dn.index(tf_d), dn.index(m_d)])

                tf_r = tf_results.get(animal, {})
                motor_r = motor_results.get(animal, {})
                for block, null_list in [('early', null_early), ('late', null_late)]:
                    tf_vec = tf_r.get('dimensions', {}).get(block, {}).get(tf_wl)
                    m_vec = motor_r.get('dimensions', {}).get(block, {}).get(m_wl)
                    if tf_vec is not None and m_vec is not None:
                        n = min(len(tf_vec), len(m_vec))
                        for _ in range(n_null):
                            shuf = rng.permutation(n)
                            null_list.append(cosine_similarity(tf_vec[:n][shuf], m_vec[:n]))

            if len(early_vals) < 2:
                continue

            fig, ax = plt.subplots(figsize=(6, 6))
            _alignment_scatter_ax(ax,
                np.array(early_vals), np.array(late_vals),
                null_early, null_late,
                f'Early block: cos(TF {tf_wl}, motor {m_wl})',
                f'Late block: cos(TF {tf_wl}, motor {m_wl})')
            fig.suptitle(f'TF {tf_wl} x motor {m_wl} [{suffix}]', fontsize=11)
            plt.tight_layout()
            fig.savefig(save_dir / f'align_tf{tf_wl}_motor{m_wl}_{suffix}.png',
                        dpi=300, bbox_inches='tight')
            figs.append(fig)

    return figs


def _alignment_scatter_ax(ax, early_vals, late_vals, null_early, null_late,
                          xlabel, ylabel):
    """draw alignment scatter onto an existing axes"""
    # null cloud
    if null_early and null_late:
        n_pts = min(len(null_early), len(null_late))
        ax.scatter(null_early[:n_pts], null_late[:n_pts], color='grey',
                   alpha=0.03, s=10, zorder=1, rasterized=True)

    # data
    ax.scatter(early_vals, late_vals, color='black', s=40, zorder=5)

    # grand mean + CI
    mean_e = np.nanmean(early_vals)
    mean_l = np.nanmean(late_vals)
    ci_e = 1.96 * np.nanstd(early_vals) / np.sqrt(len(early_vals))
    ci_l = 1.96 * np.nanstd(late_vals) / np.sqrt(len(late_vals))
    ax.errorbar(mean_e, mean_l, xerr=ci_e, yerr=ci_l,
                color='red', marker='o', markersize=10, linewidth=2, zorder=10)

    # diagonal
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
