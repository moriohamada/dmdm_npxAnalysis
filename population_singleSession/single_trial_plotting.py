"""
plotting for single-trial projections (per session/area/event/dim).
"""

from config import PATHS, PLOT_OPTIONS
from utils.smoothing import centred_boxcar

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


SMOOTH_S = {'tf': 0.100, 'lick': 0.250, 'bl': 0.250, 'ch': 0.250, 'bl_traj': 0.250}
N_VAL_BINS = 30
DIMS_TO_PLOT = [('mv', 'pot', 0), ('mv', 'pot', 1),
                ('mv', 'null', 0), ('mv', 'null', 1),
                ('tf', 'pot', 0),
                ('tf', 'movenull', 0)]
EVENTS_TO_PLOT = ['bl', 'tf', 'ch', 'lick', 'bl_traj']


def _smooth(traces, t_ax, win_s):
    """symmetric NaN-aware boxcar smoothing along time. traces: (n_trials, nT)"""
    if traces.size == 0:
        return traces
    bin_size = float(np.median(np.diff(t_ax)))
    n_bins = max(1, int(round(win_s / bin_size)))
    if n_bins <= 1:
        return traces
    return centred_boxcar(traces, n_bins, axis=-1)


def _gather_traces(proj, area, block, axis_key, dim_idx, event, cond_keys):
    """concat single-trial traces across cond_keys, returning (n_trials, nT) for one dim"""
    block_dict = proj['data'].get(area, {}).get(block, {})
    ev_dict = block_dict.get(axis_key, {}).get(event, {})
    parts = []
    for cond in cond_keys:
        arr = ev_dict.get(cond)
        if arr is None or arr.shape[0] == 0:
            continue
        parts.append(arr[:, dim_idx, :])
    if not parts:
        return np.empty((0, 0))
    return np.concatenate(parts, axis=0)


def _occupancy(traces, val_edges):
    """trials × time → val_bins × time histogram"""
    if traces.size == 0:
        return None
    nT = traces.shape[1]
    hist = np.zeros((len(val_edges) - 1, nT))
    for t_idx in range(nT):
        col = traces[:, t_idx]
        col = col[~np.isnan(col)]
        if col.size:
            hist[:, t_idx] = np.histogram(col, bins=val_edges)[0]
    return hist


def _val_edges(traces_list, n_bins=N_VAL_BINS):
    """common val edges across trace pools, robust to outliers"""
    vals = np.concatenate([trace.ravel() for trace in traces_list if trace.size])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.linspace(-1, 1, n_bins + 1)
    lo, hi = np.percentile(vals, [1, 99])
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    return np.linspace(lo, hi, n_bins + 1)


def _plot_lines_mean(ax, t_ax, traces, color, label=None, alpha_trial=0.05):
    if traces.size == 0:
        return
    for trace in traces:
        ax.plot(t_ax, trace, color=color, alpha=alpha_trial, lw=0.5)
    ax.plot(t_ax, np.nanmean(traces, axis=0), color=color, lw=2, label=label)


def _plot_heatmap(ax, t_ax, hist, val_edges, vmax=None, cmap='magma'):
    if hist is None:
        ax.set_facecolor('lightgrey')
        return None
    if vmax is None:
        vmax = np.nanpercentile(hist, 99) or 1
    extent = [t_ax[0], t_ax[-1], val_edges[0], val_edges[-1]]
    vmin = -vmax if cmap == 'RdBu_r' else 0
    return ax.imshow(hist, origin='lower', aspect='auto', extent=extent,
                     cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')


def _add_cbar(fig, ax, im, label):
    if im is None:
        return
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def _event_conds(event):
    """psth condition keys to pool, per block per cond_label"""
    if event == 'tf':
        return {
            'early': {'pos': ['earlyBlock_early_pos'], 'neg': ['earlyBlock_early_neg']},
            'late':  {'pos': ['lateBlock_early_pos'],  'neg': ['lateBlock_early_neg']},
        }
    if event == 'lick':
        return {
            'early': {'hit': ['earlyBlock_early_hit'], 'fa': ['earlyBlock_early_fa']},
            'late':  {'hit': ['lateBlock_late_hit'],   'fa': ['lateBlock_late_fa']},
        }
    if event == 'bl':
        return {'early': {'all': ['early']}, 'late': {'all': ['late']}}
    if event == 'bl_traj':
        return {'early': {'all': ['hit', 'fa', 'miss', 'abort']},
                'late':  {'all': ['hit', 'fa', 'miss', 'abort']}}
    return None


def _ch_conds_for_session(proj, area, axis_key):
    ev_dict = proj['data'][area]['all'][axis_key].get('ch', {})
    out = {'early': [], 'late': []}
    for cond in ev_dict.keys():
        if cond.startswith('early_hit_'):
            out['early'].append(cond)
        elif cond.startswith('late_hit_'):
            out['late'].append(cond)
    return out


def _fig_tf(proj, area, dim_label, axis_key, dim_idx, t_ax):
    block_col = PLOT_OPTIONS['colours']['block']
    tf_col    = PLOT_OPTIONS['colours']['tf']
    win_s     = SMOOTH_S['tf']
    spec      = _event_conds('tf')

    traces = {(block, cond): _smooth(_gather_traces(proj, area, 'all', axis_key,
                                                    dim_idx, 'tf', cond_keys),
                                     t_ax, win_s)
              for block, conds in spec.items() for cond, cond_keys in conds.items()}
    edges = _val_edges(list(traces.values()))

    fig, axs = plt.subplots(2, 2, figsize=(7, 4.5), sharex=True)
    for col, block in enumerate(['early', 'late']):
        ax_lines = axs[0, col]
        for cond, color in [('pos', tf_col['pos']), ('neg', tf_col['neg'])]:
            block_traces = traces[(block, cond)]
            _plot_lines_mean(ax_lines, t_ax, block_traces, color,
                             label=f'{cond} (n={block_traces.shape[0]})')
        ax_lines.axvline(0, c='k', alpha=.3, lw=.7)
        ax_lines.set_title(f'{block} block', fontsize=9, color=block_col[block])
        ax_lines.legend(fontsize=6)
        ax_lines.set_ylim(edges[0], edges[-1])

        ax_heat = axs[1, col]
        hist_pos = _occupancy(traces[(block, 'pos')], edges)
        hist_neg = _occupancy(traces[(block, 'neg')], edges)
        im = None
        if hist_pos is not None and hist_neg is not None:
            contrast = hist_pos - hist_neg
            vmax = np.nanmax(np.abs(contrast)) or 1
            im = _plot_heatmap(ax_heat, t_ax, contrast, edges, vmax=vmax, cmap='RdBu_r')
        ax_heat.axvline(0, c='k', alpha=.3, lw=.7)
        ax_heat.set_xlabel('t (s)')
        _add_cbar(fig, ax_heat, im, 'fast - slow')

    axs[0, 0].set_ylabel(dim_label)
    sns.despine(fig=fig)
    return fig


def _fig_lick(proj, area, dim_label, axis_key, dim_idx, t_ax):
    block_col = PLOT_OPTIONS['colours']['block']
    win_s     = SMOOTH_S['lick']
    spec      = _event_conds('lick')

    pooled = {cond: _smooth(_gather_traces(proj, area, 'all', axis_key, dim_idx, 'lick',
                                           [spec[block][cond][0]
                                            for block in ('early', 'late')]),
                            t_ax, win_s)
              for cond in ('hit', 'fa')}
    per_bc = {(block, cond): _smooth(_gather_traces(proj, area, 'all', axis_key,
                                                    dim_idx, 'lick', spec[block][cond]),
                                     t_ax, win_s)
              for block in ('early', 'late') for cond in ('hit', 'fa')}
    edges = _val_edges(list(pooled.values()) + list(per_bc.values()))

    fig, axs = plt.subplots(2, 4, figsize=(11, 4.5), sharex=True)
    for col, cond in enumerate(['hit', 'fa']):
        ax = axs[0, col]
        cond_traces = pooled[cond]
        _plot_lines_mean(ax, t_ax, cond_traces, 'k',
                         label=f'{cond} (n={cond_traces.shape[0]})')
        ax.axvline(0, c='k', alpha=.3, lw=.7)
        ax.set_title(f'{cond} (pooled)', fontsize=9)
        ax.legend(fontsize=6)
        ax.set_ylim(edges[0], edges[-1])
    axs[0, 2].axis('off'); axs[0, 3].axis('off')
    axs[0, 0].set_ylabel(dim_label)

    last_im = None
    for col, (cond, block) in enumerate([('hit', 'early'), ('hit', 'late'),
                                         ('fa', 'early'),  ('fa', 'late')]):
        ax = axs[1, col]
        hist = _occupancy(per_bc[(block, cond)], edges)
        last_im = _plot_heatmap(ax, t_ax, hist, edges)
        ax.axvline(0, c='k', alpha=.3, lw=.7)
        ax.set_xlabel('t (s)')
        n_trials = per_bc[(block, cond)].shape[0]
        ax.set_title(f'{cond} - {block} (n={n_trials})',
                     fontsize=8, color=block_col[block])
    axs[1, 0].set_ylabel('occupancy')
    _add_cbar(fig, axs[1, 3], last_im, 'count')
    sns.despine(fig=fig)
    return fig


def _fig_pooled(proj, area, dim_label, axis_key, dim_idx, event, t_ax):
    """bl, ch, bl_traj: pooled lines + per-block occupancy heatmaps. fa endpoints on bl_traj"""
    block_col = PLOT_OPTIONS['colours']['block']
    win_s     = SMOOTH_S[event]

    if event == 'ch':
        ch_conds    = _ch_conds_for_session(proj, area, axis_key)
        pooled_keys = ch_conds['early'] + ch_conds['late']
        per_block   = ch_conds
    else:
        spec        = _event_conds(event)
        pooled_keys = spec['early']['all'] + spec['late']['all']
        per_block   = {block: spec[block]['all'] for block in ('early', 'late')}

    pooled = _smooth(_gather_traces(proj, area, 'all', axis_key, dim_idx,
                                    event, pooled_keys), t_ax, win_s)
    per_bc = {block: _smooth(_gather_traces(proj, area, block, axis_key, dim_idx,
                                            event, per_block[block]), t_ax, win_s)
              for block in ('early', 'late')}
    edges = _val_edges([pooled] + list(per_bc.values()))

    fig, axs = plt.subplots(2, 2, figsize=(7, 4.5), sharex=True)
    ax = axs[0, 0]
    _plot_lines_mean(ax, t_ax, pooled, 'k', label=f'all (n={pooled.shape[0]})')
    ax.axvline(0, c='k', alpha=.3, lw=.7)
    ax.set_title('all trials (pooled)', fontsize=9)
    ax.legend(fontsize=6)
    ax.set_ylim(edges[0], edges[-1])
    ax.set_ylabel(dim_label)
    axs[0, 1].axis('off')

    if event == 'bl_traj':
        fa_info = proj.get('info', {}).get(area, {}).get('all', {}).get('fa')
        if fa_info is not None and 'end_t' in fa_info.columns:
            fa_traces = _smooth(_gather_traces(proj, area, 'all', axis_key,
                                               dim_idx, 'bl_traj', ['fa']),
                                t_ax, win_s)
            for trial_idx, end_t in enumerate(fa_info['end_t'].values):
                if np.isnan(end_t) or trial_idx >= fa_traces.shape[0]:
                    continue
                t_idx = int(np.argmin(np.abs(t_ax - end_t)))
                if 0 <= t_idx < fa_traces.shape[1]:
                    ax.plot(end_t, fa_traces[trial_idx, t_idx], 'o',
                            color='red', ms=2, alpha=0.6)

    last_im = None
    for col, block in enumerate(['early', 'late']):
        ax_heat = axs[1, col]
        hist = _occupancy(per_bc[block], edges)
        last_im = _plot_heatmap(ax_heat, t_ax, hist, edges)
        ax_heat.axvline(0, c='k', alpha=.3, lw=.7)
        ax_heat.set_xlabel('t (s)')
        n_trials = per_bc[block].shape[0]
        ax_heat.set_title(f'{block} (n={n_trials})',
                          fontsize=8, color=block_col[block])
    axs[1, 0].set_ylabel('occupancy')
    _add_cbar(fig, axs[1, 1], last_im, 'count')
    sns.despine(fig=fig)
    return fig


def _figure_for_event(proj, area, dim_label, axis_key, dim_idx, event, t_ax):
    if event == 'tf':
        return _fig_tf(proj, area, dim_label, axis_key, dim_idx, t_ax)
    if event == 'lick':
        return _fig_lick(proj, area, dim_label, axis_key, dim_idx, t_ax)
    return _fig_pooled(proj, area, dim_label, axis_key, dim_idx, event, t_ax)


def visualize_responses(proj, animal, session_name):
    """save one png per (area, event, dim) under
    <plots_dir>/single_trial_traj/<animal>/<session>/<area>/"""
    base = Path(PATHS['plots_dir']) / 'single_trial_traj' / animal / session_name
    events_present = [event for event in EVENTS_TO_PLOT if event in proj['t_ax']]

    for area in proj['data'].keys():
        block_data = proj['data'][area].get('all', {})
        save_dir = base / area
        save_dir.mkdir(parents=True, exist_ok=True)

        for source, axis_name, dim_idx in DIMS_TO_PLOT:
            axis_key = f'{source}_{axis_name}'
            if axis_key not in block_data:
                continue
            example_arr = next((arr for ev_dict in block_data[axis_key].values()
                                for arr in (ev_dict.values()
                                            if isinstance(ev_dict, dict) else [])
                                if hasattr(arr, 'shape') and arr.size > 0), None)
            if example_arr is None or example_arr.shape[1] <= dim_idx:
                continue
            n_dim = example_arr.shape[1]
            dim_label = f'{axis_key}_d{dim_idx}' if n_dim > 1 else axis_key

            for event in events_present:
                t_ax = proj['t_ax'][event]
                fig = _figure_for_event(proj, area, dim_label, axis_key,
                                        dim_idx, event, t_ax)
                fig.suptitle(f'{animal}/{session_name} – {area} – {event} – {dim_label}',
                             fontsize=10)
                fig.tight_layout()
                fig.savefig(save_dir / f'{event}_{dim_label}.png', dpi=150,
                            bbox_inches='tight')
                plt.close(fig)
