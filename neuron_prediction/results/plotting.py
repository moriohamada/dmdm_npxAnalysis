"""
per-neuron kernel plots and population summaries, shared across glm variants
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
from multiprocessing import Pool

sns.set_style("whitegrid")

from config import PATHS, GLM_OPTIONS, PLOT_OPTIONS
from neuron_prediction.results import FIT_TYPES
from neuron_prediction.results.classify import extract_kernels
from data.session import Session
from utils.rois import PLOT_CLASS_AREAS


def _title_tag(fit_type):
    """display tag for a fit_type, e.g. glm_ridge -> 'GLM (ridge)'"""
    if fit_type == 'glm':
        return 'GLM'
    suffix = fit_type.replace('glm_', '')
    return f'GLM ({suffix})'


def plot_glm_kernels(weights, col_map, neuron_idx=0, region=None,
                     mean_r=None, classifications=None,
                     save_dir=None):
    """plot all GLM kernels for one neuron"""
    kernels = extract_kernels(weights, col_map)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()

    # panel 0: baseline TF
    ax = axes[0]
    if 'tf' in kernels:
        t, w = kernels['tf']
        ax.plot(t, w, 'k', lw=1.5)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Baseline TF')
    ax.set_xlabel('Time (s)')

    # panel 1: change onset (overlaid by magnitude)
    ax = axes[1]
    ch_keys = sorted([k for k in kernels if k.startswith('change_tf')])
    ch_cmap = plt.colormaps[PLOT_OPTIONS['colours']['ch_tf_cmap']]
    if ch_keys:
        colours = ch_cmap(np.linspace(0.15, 0.85, len(ch_keys)))
        colours[0] = (0.6, 0.6, 0.6, 1.0)
        for key, colour in zip(ch_keys, colours):
            t, w = kernels[key]
            label = key.replace('change_tf', '')
            ax.plot(t, w, color=colour, lw=1.2, label=label)
        ax.legend(fontsize=6, title='TF', title_fontsize=6)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Change onset')
    ax.set_xlabel('Time (s)')

    # panel 2: lick prep + exec
    ax = axes[2]
    if 'lick_prep' in kernels:
        t, w = kernels['lick_prep']
        ax.plot(t, w, color='steelblue', lw=1.5, label='prep')
    if 'lick_exec' in kernels:
        t, w = kernels['lick_exec']
        ax.plot(t, w, color='firebrick', lw=1.5, label='exec')
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5, ls='--')
    ax.set_title('Lick')
    ax.set_xlabel('Time from lick (s)')
    ax.legend(fontsize=7)

    # panel 3: trial start
    ax = axes[3]
    if 'trial_start' in kernels:
        t, w = kernels['trial_start']
        ax.plot(t, w, 'k', lw=1.5)
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Trial start')
    ax.set_xlabel('Time (s)')

    # title
    title = f'Unit {neuron_idx}'
    if region:
        title += f' ({region})'
    if mean_r is not None:
        title += f' - r={mean_r:.2f}'
    if classifications:
        flags = [k.replace('_sig', '') for k, v in classifications.items()
                 if k.endswith('_sig') and v]
        if flags:
            title += f' [{", ".join(flags)}]'
    fig.suptitle(title, fontsize=11)

    for a in axes:
        a.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / f'glm_unit_{neuron_idx:04d}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def _plot_one_neuron(args):
    """worker for parallel kernel plotting"""
    i, results_dir, col_map, region, class_row, save_dir = args

    matplotlib.use('Agg')

    res_path = results_dir / f'neuron_{i}.npz'
    if not res_path.exists():
        return

    res = np.load(res_path, allow_pickle=True)
    mean_r = np.nanmean(res['full_r'])
    classifications = None
    if class_row is not None:
        classifications = {c: class_row[c] for c in class_row.index
                           if c.endswith('_sig')}

    plot_glm_kernels(res['weights'], col_map,
                     neuron_idx=i, region=region,
                     mean_r=mean_r, classifications=classifications,
                     save_dir=str(save_dir))


def plot_all_glm_kernels(sess_dir, fit_type='glm_ridge', plots_dir=None, n_workers=1):
    """plot GLM kernels for all neurons in one session"""
    sess_dir = Path(sess_dir)
    sess = Session.load(str(sess_dir / 'session.pkl'))
    results_dir = sess_dir / f'{fit_type}_results'

    with open(sess_dir / 'glm_spec.pkl', 'rb') as f:
        col_map = pickle.load(f)

    class_path = sess_dir / f'{fit_type}_classifications.csv'
    class_df = None
    if class_path.exists():
        class_df = pd.read_csv(class_path)

    if plots_dir is None:
        plots_dir = sess_dir / f'{fit_type}_kernels'
    save_dir = Path(plots_dir) / sess.animal / sess.name / f'{fit_type}_kernels'

    regions = sess.unit_info['brain_region_comb'].values
    n_neurons = len(sess.fr_stats)

    args = []
    for i in range(n_neurons):
        region = regions[i] if i < len(regions) else None
        class_row = class_df.iloc[i] if class_df is not None and i < len(class_df) else None
        args.append((i, results_dir, col_map, region, class_row, save_dir))

    if n_workers > 1:
        with Pool(n_workers) as pool:
            pool.map(_plot_one_neuron, args)
    else:
        for a in args:
            _plot_one_neuron(a)

    print(f'Saved kernel plots to {save_dir}')


#%% population summaries

def load_all_classifications(fit_type, npx_dir=None):
    """load classifications for fit_type across all sessions, with metadata"""
    if npx_dir is None:
        npx_dir = PATHS['npx_dir_local']
    rows = []
    for subj in sorted(os.listdir(npx_dir)):
        subj_dir = os.path.join(npx_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for sess_name in sorted(os.listdir(subj_dir)):
            sd = os.path.join(subj_dir, sess_name)
            class_path = os.path.join(sd, f'{fit_type}_classifications.csv')
            if not os.path.exists(class_path):
                continue
            cdf = pd.read_csv(class_path)
            sess = Session.load(os.path.join(sd, 'session.pkl'))
            cdf['region'] = sess.unit_info['brain_region_comb'].values[:len(cdf)]
            cdf['animal'] = sess.animal
            cdf['session'] = sess.name
            rows.append(cdf)

    all_units = pd.concat(rows, ignore_index=True)
    print(f'{len(all_units)} units across {all_units["session"].nunique()} sessions')
    return all_units


def plot_fraction_significant(fit_type='glm_ridge', npx_dir=None,
                              min_units=10, save_dir=None):
    """fraction of significant units per lesion group, by brain region"""
    if npx_dir is None:
        npx_dir = PATHS['npx_dir_local']
    all_units = load_all_classifications(fit_type, npx_dir)
    group_names = list(GLM_OPTIONS['lesion_groups'].keys())

    region_to_class = {}
    for cls, regions in PLOT_CLASS_AREAS.items():
        for r in regions:
            region_to_class[r] = cls

    all_units['area_class'] = all_units['region'].map(region_to_class)

    class_names = list(PLOT_CLASS_AREAS.keys())
    cmap = plt.colormaps['tab10'].resampled(len(class_names))
    class_colours = {cls: cmap(i) for i, cls in enumerate(class_names)}

    fig, axes = plt.subplots(1, len(group_names),
                             figsize=(4 * len(group_names), 6), sharey=True)

    row_order = None
    for ai, (ax, g) in enumerate(zip(axes, group_names)):
        sig_col = f'{g}_sig'
        if sig_col not in all_units.columns:
            continue

        sub = all_units.dropna(subset=[sig_col, 'area_class'])
        region_n = sub.groupby('region').size()
        keep = region_n[region_n >= min_units].index
        sub = sub[sub['region'].isin(keep)]

        summary = sub.groupby('region').agg(
            n_total=(sig_col, 'count'),
            n_sig=(sig_col, 'sum'),
            area_class=('area_class', 'first'),
        )
        summary['frac'] = summary['n_sig'] / summary['n_total']

        summary['class_order'] = summary['area_class'].map(
            {c: i for i, c in enumerate(class_names)})

        if row_order is None:
            # sort by first feature only
            summary = summary.sort_values(['class_order', 'frac'],
                                          ascending=[True, True])
            row_order = summary.index.tolist()
        else:
            summary = summary.reindex(row_order).dropna(subset=['n_total'])

        colours = [class_colours[c] for c in summary['area_class']]
        ax.barh(range(len(summary)), summary['frac'], color=colours)
        ax.set_yticks(range(len(summary)))
        ax.set_yticklabels(summary.index, fontsize=7)
        for tick, c in zip(ax.get_yticklabels(), summary['area_class']):
            tick.set_color(class_colours[c])
        ax.set_xlabel('Fraction significant')
        ax.set_title(g)
        ax.set_xlim(0, 1)

        if ai == 0:
            handles = [Patch(color=class_colours[c], label=c.replace('_', ' '))
                       for c in class_names]
            ax.legend(handles=handles, fontsize=6, loc='lower right')

    fig.suptitle(f'{_title_tag(fit_type)} lesion significance by region',
                 fontsize=12)
    fig.tight_layout()

    if save_dir:
        save_dir = Path(save_dir) / fit_type
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f'{fit_type}_fraction_significant.png',
                    dpi=300, bbox_inches='tight')

    return fig
