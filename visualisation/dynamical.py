"""
Visualisation for projected trajectories in PC space.

Per-event-type figures. Rows = flow field conditions, Cols = trajectory conditions.
Each subplot shows one flow field with overlaid trajectories (pos/neg, fa/hit, etc).
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from pathlib import Path
sns.set_style("whitegrid")

from config import ANALYSIS_OPTIONS, PATHS, PLOT_OPTIONS
from data.session import Session
from utils.rois import in_any_area, in_group
from analyses.dynamical import CONDITIONS

FLOW_CONDS = list(CONDITIONS.keys())
_FLOW_LABELS = {
    'earlyBlock_early': 'flow: early block, early',
    'lateBlock_early':  'flow: late block, early',
    'lateBlock_late':   'flow: late block, late',
}

# trajectory columns per event type: (psth_condition_prefix, title)
_TRAJ_COLS = {
    'tf':   [('earlyBlock_early', 'early block, early'),
             ('lateBlock_early',  'late block, early'),
             ('lateBlock_late',   'late block, late')],
    'blOn': [('early', 'early block'),
             ('late',  'late block')],
    'lick': [('earlyBlock_early', 'early block, early'),
             ('lateBlock_early',  'late block, early'),
             ('lateBlock_late',   'late block, late')],
}


def _get_area_mask(areas, pca_key):
    """Return boolean mask for the neurons used by a given pca_key."""
    group_name = pca_key.split('_', 1)[1]
    if group_name == 'all':
        return in_any_area(areas)
    return in_group(areas, group_name)


def _get_sigma(ops, event_type):
    """Smoothing sigma in bins for trajectory plotting.
    Config values are in seconds, converted to bins here."""
    short_events = ('tf', 'blOn', 'bl')
    key = 'smooth_sigma_short' if event_type in short_events else 'smooth_sigma_long'
    return PLOT_OPTIONS[key] / ops['sp_bin_width']


def plot_trajectory(traj_2d, t_ax, ax,
                    color='k', label=None, lw=1.5,
                    smooth_sigma=0):
    """
    Plot a 2D trajectory with start marker (circle) and t=0 marker (square).
    smooth_sigma in bins (0 = no smoothing, causal).
    """
    if smooth_sigma > 0:
        from utils.smoothing import causal_gaussian
        traj_2d = causal_gaussian(traj_2d, sigma_bins=smooth_sigma)
    ax.plot(traj_2d[0], traj_2d[1], color=color, lw=lw, label=label)
    ax.plot(traj_2d[0, 0], traj_2d[1, 0], 'o', color=color, ms=6)
    t0_idx = np.argmin(np.abs(t_ax))
    ax.plot(traj_2d[0, t0_idx], traj_2d[1, t0_idx], 's', color=color, ms=5)


def _load_plot_data(sess_dir, pca_key):
    """Load session, PCA weights, and all mean PSTHs (area-masked, baseline-subtracted)."""
    sess_data = Session.load(str(sess_dir / 'session.pkl'))
    areas = sess_data.unit_info['brain_region_comb'].values
    area_mask = _get_area_mask(areas, pca_key)

    with h5py.File(sess_dir / 'pca.h5', 'r') as f:
        weights = f[pca_key]['weights'][:]

    means = {}
    t_axes = {}
    with h5py.File(str(sess_dir / 'psths.h5'), 'r') as f:
        for key in f:
            if not key.endswith('_mean'):
                continue
            et = key.replace('_mean', '')
            if f't_ax/{et}' in f:
                t_axes[et] = f[f't_ax/{et}'][:]
            for cond in f[key]:
                data = f[key][cond][:]
                if area_mask is not None:
                    data = data[area_mask]
                t = t_axes.get(et)
                if t is not None:
                    bl = t < 0
                    if bl.any():
                        data = data - np.nanmean(data[:, bl], axis=1, keepdims=True)
                means[(et, cond)] = data

    return sess_data, weights, means, t_axes


def _get_traj(event_type, cond, means, t_axes, weights):
    """Project a mean PSTH into PC1/PC2, return (2, nT) trajectory and time axis."""
    key = (event_type, cond)
    if key not in means:
        return None, None
    z = weights.T @ means[key]
    return z[:2], t_axes[event_type]


def _plot_trajs_on_ax(ax, event_type, traj_cond, means, t_axes, weights, sigma):
    """Overlay trajectories for a given event_type + trajectory condition on an axis."""
    c_tf = PLOT_OPTIONS['colours']['tf']
    c_block = PLOT_OPTIONS['colours']['block']

    if event_type == 'tf':
        for pol, color in c_tf.items():
            traj, t = _get_traj(event_type, f'{traj_cond}_{pol}',
                                means, t_axes, weights)
            if traj is not None:
                plot_trajectory(traj, t, ax, color=color,
                                label=pol, smooth_sigma=sigma)

    elif event_type == 'blOn':
        color = c_block[traj_cond]
        traj, t = _get_traj(event_type, traj_cond, means, t_axes, weights)
        if traj is not None:
            plot_trajectory(traj, t, ax, color=color,
                            label=traj_cond, smooth_sigma=sigma)

    elif event_type == 'lick':
        block = 'early' if traj_cond.startswith('early') else 'late'
        color = c_block[block]
        for outcome, ls in [('fa', '-'), ('hit', '--')]:
            traj, t = _get_traj(event_type, f'{traj_cond}_{outcome}',
                                means, t_axes, weights)
            if traj is not None:
                plot_trajectory(traj, t, ax, color=color, lw=1.5,
                                label=outcome, smooth_sigma=sigma)


def _draw_empirical_flow(ax, flow, bin_centers):
    """Draw an empirical flow field (quiver) on an axis. NaN bins are skipped."""
    C0, C1 = np.meshgrid(bin_centers[0], bin_centers[1], indexing='ij')
    U = flow[:, :, 0]
    V = flow[:, :, 1]
    valid = ~np.isnan(U)
    if valid.any():
        ax.quiver(C0[valid], C1[valid], U[valid], V[valid],
                  color='k', alpha=0.4, scale=None)


def _draw_lds_flow(ax, A, grid_range=(-3, 3), n_grid=15):
    """Draw LDS-derived flow field on an axis using (A - I) dynamics."""
    A_2d = (A - np.eye(A.shape[0]))[:2, :2]
    g = np.linspace(grid_range[0], grid_range[1], n_grid)
    P1, P2 = np.meshgrid(g, g)
    U = A_2d[0, 0] * P1 + A_2d[0, 1] * P2
    V = A_2d[1, 0] * P1 + A_2d[1, 1] * P2
    ax.quiver(P1, P2, U, V, color='k', alpha=0.4, scale=None)


def _format_and_save(fig, axes, title, save_path, fixed_range=None):
    """Share axis limits across all subplots, label, save or show.

    fixed_range: if provided, (x_range, y_range) tuple to force axis limits
                 (e.g. from flow field grid extents).
    """
    if fixed_range is not None:
        shared_x, shared_y = fixed_range
    else:
        xlims = [ax.get_xlim() for ax in axes.flat]
        ylims = [ax.get_ylim() for ax in axes.flat]
        shared_x = (min(lo for lo, _ in xlims), max(hi for _, hi in xlims))
        shared_y = (min(lo for lo, _ in ylims), max(hi for _, hi in ylims))

    for ax in axes.flat:
        ax.set_xlim(shared_x)
        ax.set_ylim(shared_y)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_aspect('equal')

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.close()
    return fig


def plot_session_dynamics(sess_dir, pca_key='event_all',
                          event_type='tf', ops=ANALYSIS_OPTIONS,
                          save_path=None):
    """
    Per-event-type figure with LDS flow fields.
    Rows = flow field conditions (3), Cols = trajectory conditions.
    """
    sess_data, weights, means, t_axes = _load_plot_data(sess_dir, pca_key)
    sigma = _get_sigma(ops, event_type)

    # load LDS A matrices per condition
    lds_path = sess_dir / f'lds_{pca_key}.h5'
    A_mats = {}
    if lds_path.exists():
        with h5py.File(lds_path, 'r') as f:
            for cond in FLOW_CONDS:
                if cond in f:
                    A_mats[cond] = f[cond]['A'][:]

    traj_cols = _TRAJ_COLS[event_type]
    n_rows = len(FLOW_CONDS)
    n_cols = len(traj_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, flow_cond in enumerate(FLOW_CONDS):
        for col, (traj_cond, traj_title) in enumerate(traj_cols):
            ax = axes[row, col]
            _plot_trajs_on_ax(ax, event_type, traj_cond,
                              means, t_axes, weights, sigma)
            if row == 0:
                ax.set_title(traj_title)
            if col == 0:
                ax.set_ylabel(f'{_FLOW_LABELS[flow_cond]}\nPC2')
            ax.legend(fontsize=7)

    # share axes first, then draw LDS flow fields so they cover the full range
    title = (f'{sess_data.animal}_{sess_data.name} — {event_type} '
             f'LDS (pca={pca_key})')
    _format_and_save(fig, axes, title, save_path=None)

    shared_x, shared_y = axes[0, 0].get_xlim(), axes[0, 0].get_ylim()
    grid_range = (min(shared_x[0], shared_y[0]), max(shared_x[1], shared_y[1]))
    for row, flow_cond in enumerate(FLOW_CONDS):
        if flow_cond not in A_mats:
            continue
        for col in range(n_cols):
            _draw_lds_flow(axes[row, col], A_mats[flow_cond],
                           grid_range=grid_range)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)



def plot_empirical_flow(sess_dir, pca_key='event_all',
                        event_type='tf', ops=ANALYSIS_OPTIONS,
                        save_path=None):
    """
    Per-event-type figure with empirical flow fields.
    Rows = flow field conditions (3), Cols = trajectory conditions.
    """
    sess_data, weights, means, t_axes = _load_plot_data(sess_dir, pca_key)
    sigma = _get_sigma(ops, event_type)

    flow_path = sess_dir / f'flow_{pca_key}.h5'
    if not flow_path.exists():
        print(f'  No flow file: {flow_path}')
        return None

    # load empirical flow data
    with h5py.File(flow_path, 'r') as f:
        bin_centers = [f[f'bin_centers/{d}'][:] for d in f['bin_centers']]
        grid_edges = [f[f'grid_edges/{d}'][:] for d in f['grid_edges']]
        flows = {}
        for cond in FLOW_CONDS:
            if cond in f:
                flows[cond] = f[cond]['mean_flow'][:]

    if not flows:
        return None

    # axis limits from grid extent
    flow_range = ((grid_edges[0][0], grid_edges[0][-1]),
                  (grid_edges[1][0], grid_edges[1][-1]))

    traj_cols = _TRAJ_COLS[event_type]
    n_rows = len(FLOW_CONDS)
    n_cols = len(traj_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, flow_cond in enumerate(FLOW_CONDS):
        for col, (traj_cond, traj_title) in enumerate(traj_cols):
            ax = axes[row, col]

            _plot_trajs_on_ax(ax, event_type, traj_cond,
                              means, t_axes, weights, sigma)

            if flow_cond in flows:
                _draw_empirical_flow(ax, flows[flow_cond], bin_centers)

            if row == 0:
                ax.set_title(traj_title)
            if col == 0:
                ax.set_ylabel(f'{_FLOW_LABELS[flow_cond]}\nPC2')
            ax.legend(fontsize=7)

    title = (f'{sess_data.animal}_{sess_data.name} — {event_type} '
             f'empirical flow (pca={pca_key})')
    return _format_and_save(fig, axes, title, save_path,
                                  fixed_range=flow_range)


def _time_graded_line(ax, traj_2d, t_ax, color=(0, 0, 0), lw=2):
    """
    Plot a 2D trajectory with luminance graded by time.
    Early = light (white-blended), late = full saturated color.
    Circle at t=0, square at end.
    """
    color = np.asarray(color)
    points = np.column_stack([traj_2d[0], traj_2d[1]])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    n_seg = len(segments)
    t_frac = np.linspace(0, 1, n_seg)
    seg_colors = (1 - t_frac[:, None]) + color * t_frac[:, None]
    lc = LineCollection(segments, colors=seg_colors, linewidths=lw)
    ax.add_collection(lc)
    ax.plot(traj_2d[0, 0], traj_2d[1, 0], 'o',
            color=(color + 1) / 2, ms=5, zorder=5)
    ax.plot(traj_2d[0, -1], traj_2d[1, -1], 's',
            color=color, ms=4, zorder=5)


def plot_pulse_trajectories(sess_dir, pca_key='event_all',
                            ops=ANALYSIS_OPTIONS, post_window=0.7,
                            n_state_bins=3, min_events=5, save_path=None):
    """
    TF pulse responses on flow fields, grouped by initial neural state.

    Single-trial TF responses are projected to PC1-PC2, binned by state at
    pulse onset (t=0), averaged within bins, smoothed, and plotted with
    luminance-graded color (white→saturated) from t=0 to t=post_window.

    Layout: rows = conditions (one subplot each).
    Pos (red) and neg (blue) TF pulses overlaid with their own colors.
    Flow field from matching condition drawn underneath.
    """
    from utils.smoothing import causal_gaussian

    sess_dir = Path(sess_dir)
    psth_path = sess_dir / 'psths.h5'
    flow_path = sess_dir / f'flow_{pca_key}.h5'
    lds_path = sess_dir / f'lds_{pca_key}.h5'
    pca_path = sess_dir / 'pca.h5'

    if not psth_path.exists() or not pca_path.exists():
        print(f'  Missing psths.h5 or pca.h5 in {sess_dir}')
        return None

    sess_data = Session.load(str(sess_dir / 'session.pkl'))
    areas = sess_data.unit_info['brain_region_comb'].values
    area_mask = _get_area_mask(areas, pca_key)

    with h5py.File(pca_path, 'r') as f:
        if pca_key not in f:
            return None
        weights = f[pca_key]['weights'][:]

    flows, bin_centers = {}, None
    A_mats = {}
    if flow_path.exists():
        with h5py.File(flow_path, 'r') as f:
            bin_centers = [f[f'bin_centers/{d}'][:] for d in f['bin_centers']]
            for cond in FLOW_CONDS:
                if cond in f:
                    flows[cond] = f[cond]['mean_flow'][:]
    if lds_path.exists():
        with h5py.File(lds_path, 'r') as f:
            for cond in FLOW_CONDS:
                if cond in f:
                    A_mats[cond] = f[cond]['A'][:]

    if not flows and not A_mats:
        print(f'  No flow or LDS files for {pca_key} in {sess_dir}')

    # load single-trial TF PSTHs and project to PC space
    cond_labels = ['earlyBlock_early', 'lateBlock_early', 'lateBlock_late']
    polarities = ['pos', 'neg']
    pol_colors = PLOT_OPTIONS['colours']['tf']
    sigma_bins = _get_sigma(ops, 'tf')

    projected = {}  # (cond, pol) → (n_events, 2, n_time)
    with h5py.File(psth_path, 'r') as f:
        t_ax_tf = f['t_ax/tf'][:]
        for cond in cond_labels:
            for pol in polarities:
                key = f'{cond}_{pol}'
                if f'tf/{key}' not in f:
                    continue
                data = f[f'tf/{key}'][:]  # (n_events, n_neurons, n_time)
                if data.shape[0] == 0:
                    continue
                data = data[:, area_mask, :]
                if data.shape[1] != weights.shape[0]:
                    continue
                # project each trial: (n_events, 2, n_time)
                z = np.einsum('pn,ent->ept', weights[:, :2].T, data)
                projected[(cond, pol)] = z

    if not projected:
        return None

    # time indices
    t0_idx = np.argmin(np.abs(t_ax_tf))
    t_end_idx = np.argmin(np.abs(t_ax_tf - post_window))
    t_win = t_ax_tf[t0_idx:t_end_idx + 1]

    # pre-event context window for binning
    ctx_start, ctx_end = ops['tf_context']
    ctx_s = np.argmin(np.abs(t_ax_tf - ctx_start))
    ctx_e = np.argmin(np.abs(t_ax_tf - ctx_end))

    # shared state-space bins across polarities (pool pre-event mean state)
    cond_edges = {}
    for cond in cond_labels:
        z0_all = []
        for pol in polarities:
            if (cond, pol) in projected:
                z0_all.append(projected[(cond, pol)][:, :, ctx_s:ctx_e].mean(axis=2))
        if z0_all:
            z0_pooled = np.concatenate(z0_all, axis=0)
            cond_edges[cond] = (
                np.linspace(np.percentile(z0_pooled[:, 0], 5),
                            np.percentile(z0_pooled[:, 0], 95),
                            n_state_bins + 1),
                np.linspace(np.percentile(z0_pooled[:, 1], 5),
                            np.percentile(z0_pooled[:, 1], 95),
                            n_state_bins + 1),
            )

    n_rows = len(cond_labels)
    fig = plt.figure(figsize=(13, 5.5 * n_rows))
    axes = fig.subplots(n_rows, 2, squeeze=False)
    delta_axes = axes[:, 1:]

    for row, cond in enumerate(cond_labels):
        ax = axes[row, 0]

        # draw flow field (check empirical flow has valid bins)
        has_empirical = (cond in flows and bin_centers is not None
                         and np.any(~np.isnan(flows[cond][:, :, 0])))
        if has_empirical:
            _draw_empirical_flow(ax, flows[cond], bin_centers)
        elif cond in A_mats and cond in cond_edges:
            pc1_e, pc2_e = cond_edges[cond]
            lo = min(pc1_e[0], pc2_e[0])
            hi = max(pc1_e[-1], pc2_e[-1])
            _draw_lds_flow(ax, A_mats[cond], grid_range=(lo, hi), n_grid=10)
        if row == 0:
            ax.set_title('pos & neg')

        if cond not in cond_edges:
            ax.set_title(f'{_FLOW_LABELS[cond]} — no TF data')
            continue
        pc1_edges, pc2_edges = cond_edges[cond]
        for e in pc1_edges[1:-1]:
            ax.axvline(e, color='grey', lw=0.5, ls='--', alpha=0.5)
        for e in pc2_edges[1:-1]:
            ax.axhline(e, color='grey', lw=0.5, ls='--', alpha=0.5)

        # compute per-bin smoothed trajectories for each polarity
        bin_trajs = {}  # (pol, i, j) → (2, n_time_win)
        bin_starts = {}  # (i, j) → (2,) mean starting position
        for pol in polarities:
            key = (cond, pol)
            if key not in projected:
                continue
            z_all = projected[key]
            z_ctx = z_all[:, :, ctx_s:ctx_e].mean(axis=2)
            for i in range(n_state_bins):
                for j in range(n_state_bins):
                    in_bin = ((z_ctx[:, 0] >= pc1_edges[i]) &
                              (z_ctx[:, 0] < pc1_edges[i + 1]) &
                              (z_ctx[:, 1] >= pc2_edges[j]) &
                              (z_ctx[:, 1] < pc2_edges[j + 1]))
                    if in_bin.sum() < min_events:
                        continue
                    mean_full = z_all[in_bin].mean(axis=0)
                    mean_full = causal_gaussian(mean_full, sigma_bins)
                    bin_trajs[(pol, i, j)] = mean_full[:, t0_idx:t_end_idx + 1]
                    if (i, j) not in bin_starts:
                        bin_starts[(i, j)] = z_ctx[in_bin].mean(axis=0)

        # plot pos and neg on main axis
        for pol in polarities:
            for i in range(n_state_bins):
                for j in range(n_state_bins):
                    if (pol, i, j) in bin_trajs:
                        _time_graded_line(ax, bin_trajs[(pol, i, j)], t_win,
                                         color=pol_colors[pol], lw=2.5)

        if not bin_trajs:
            ax.text(0.5, 0.5, 'too few events per bin',
                    transform=ax.transAxes, ha='center', va='center')

        ax.autoscale_view()
        pad1 = (pc1_edges[-1] - pc1_edges[0]) * 0.15
        pad2 = (pc2_edges[-1] - pc2_edges[0]) * 0.15
        ax.set_xlim(pc1_edges[0] - pad1, pc1_edges[-1] + pad1)
        ax.set_ylim(pc2_edges[0] - pad2, pc2_edges[-1] + pad2)
        ax.set_ylabel(f'{_FLOW_LABELS[cond]}\nPC2')
        ax.set_xlabel('PC1')

        # plot pos-minus-neg on delta axis, offset by bin starting position
        ax_d = delta_axes[row, 0]
        if cond in A_mats:
            _draw_lds_flow(ax_d, A_mats[cond],
                           grid_range=(min(pc1_edges[0], pc2_edges[0]),
                                       max(pc1_edges[-1], pc2_edges[-1])),
                           n_grid=10)
        for e in pc1_edges[1:-1]:
            ax_d.axvline(e, color='grey', lw=0.5, ls='--', alpha=0.5)
        for e in pc2_edges[1:-1]:
            ax_d.axhline(e, color='grey', lw=0.5, ls='--', alpha=0.5)

        diff_color = (0.5, 0.2, 0.7)
        for i in range(n_state_bins):
            for j in range(n_state_bins):
                if ('pos', i, j) in bin_trajs and ('neg', i, j) in bin_trajs:
                    delta = bin_trajs[('pos', i, j)] - bin_trajs[('neg', i, j)]
                    start = bin_starts.get((i, j), np.zeros(2))
                    delta_offset = delta + start[:, None]
                    _time_graded_line(ax_d, delta_offset, t_win,
                                     color=diff_color, lw=2)

        ax_d.set_xlim(ax.get_xlim())
        ax_d.set_ylim(ax.get_ylim())
        ax_d.set_ylabel(f'{_FLOW_LABELS[cond]}\nPC2')
        ax_d.set_xlabel('PC1')
        ax_d.set_title('pos - neg' if row == 0 else '')

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=pol_colors[p], lw=2.5, label=f'{p} TF')
               for p in polarities]
    handles.append(Line2D([0], [0], color=(0.5, 0.2, 0.7), lw=2, label='pos-neg'))
    axes[0, 0].legend(handles=handles, fontsize=8)

    title = (f'{sess_data.animal}_{sess_data.name} — TF pulse trajectories '
             f'by initial state (pca={pca_key})\n'
             f'light=t0, saturated=t+{post_window:.1f}s  |  '
             f'o=start, ■=end')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return fig
