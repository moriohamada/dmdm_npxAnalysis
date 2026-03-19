"""
Projected trajectories in PC space, binned by pre-event neural state.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from pathlib import Path
sns.set_style("whitegrid")

from config import ANALYSIS_OPTIONS, PLOT_OPTIONS
from data.session import Session
from utils.rois import in_any_area, in_group
from utils.smoothing import causal_boxcar
from population.dynamical import CONDITIONS

COND_LABELS = ['earlyBlock_early', 'lateBlock_early']
COND_NAMES  = {
    'earlyBlock_early': 'early block, early',
    'lateBlock_early':  'late block, early',
}


#%%

def _load_session(sess_dir, pca_key):
    """Load session, PCA weights (nN x nPC), area mask, LDS A and B matrices."""
    pca_path = sess_dir / 'pca.h5'
    if not pca_path.exists():
        return None, None, None, None, None

    sess_data = Session.load(str(sess_dir / 'session.pkl'))
    areas = sess_data.unit_info['brain_region_comb'].values
    group = pca_key.split('_', 1)[1]
    area_mask = in_any_area(areas) if group == 'all' else in_group(areas, group)

    with h5py.File(pca_path, 'r') as f:
        if pca_key not in f:
            return None, None, None, None, None
        weights = f[pca_key]['weights'][:]

    A_mats, B_mats = {}, {}
    lds_path = sess_dir / f'lds_{pca_key}.h5'
    if lds_path.exists():
        with h5py.File(lds_path, 'r') as f:
            for c in COND_LABELS:
                if c in f:
                    A_mats[c] = f[c]['A'][:]
                    B_mats[c] = f[c]['B'][:]

    return sess_data, area_mask, weights, A_mats, B_mats


def _load_projected(psth_path, event_type, keys, area_mask, weights, pcs=(0, 1)):
    """
    Load single-trial PSTHs, project to selected PCs.
    Returns dict {key: (nEv, 2, nT)} and t_ax array.
    """
    projected = {}
    t_ax = None
    with h5py.File(psth_path, 'r') as f:
        if f't_ax/{event_type}' not in f:
            return projected, t_ax
        t_ax = f[f't_ax/{event_type}'][:]
        W = weights[:, list(pcs)].T  # (2, nN)
        for key in keys:
            if f'{event_type}/{key}' not in f:
                continue
            data = f[f'{event_type}/{key}'][:]  # (nEv, nN, nT)
            if data.shape[0] == 0:
                continue
            data = data[:, area_mask, :]
            if data.shape[1] != weights.shape[0]:
                continue
            projected[key] = np.einsum('pn,ent->ept', W, data)
    return projected, t_ax


#%%

def _ctx_idx(t_ax, win):
    return (np.argmin(np.abs(t_ax - win[0])),
            np.argmin(np.abs(t_ax - win[1])))


def _bin_edges(z0, n_bins):
    return tuple(
        np.linspace(np.percentile(z0[:, d], 5),
                    np.percentile(z0[:, d], 95), n_bins + 1)
        for d in range(2))


def _bin_and_smooth(z_all, z_ctx, edges, t_slice, smooth_w, n_bins, min_ev):
    """
    Bin trials by pre-event state, average, smooth, slice.
    Returns dicts keyed by (i,j): trajs (2,nT), counts (int), starts (2,).
    """
    pc1_e, pc2_e = edges
    trajs, counts, starts = {}, {}, {}
    for i in range(n_bins):
        for j in range(n_bins):
            mask = ((z_ctx[:, 0] >= pc1_e[i]) & (z_ctx[:, 0] < pc1_e[i+1]) &
                    (z_ctx[:, 1] >= pc2_e[j]) & (z_ctx[:, 1] < pc2_e[j+1]))
            n = mask.sum()
            if n < min_ev:
                continue
            mean = causal_boxcar(z_all[mask].mean(axis=0), smooth_w)
            trajs[(i, j)] = mean[:, t_slice]
            counts[(i, j)] = n
            starts[(i, j)] = z_ctx[mask].mean(axis=0)
    return trajs, counts, starts


def _marker_size(n, lo, hi, ms_range=(3, 10)):
    """ map event count n in [lo, hi] to marker size."""
    if hi == lo:
        return np.mean(ms_range)
    return ms_range[0] + (ms_range[1] - ms_range[0]) * (n - lo) / (hi - lo)


def _plot_binned(ax, trajs, counts, color, lw=2.5):
    """Plot all binned trajectories from a single _bin_and_smooth call."""
    if not counts:
        return
    lo, hi = min(counts.values()), max(counts.values())
    for ij, traj in trajs.items():
        _graded_line(ax, traj, color, lw=lw, start_ms=_marker_size(counts[ij], lo, hi))


def _plot_diff(ax, trajs_a, trajs_b, counts_a, counts_b, starts, color, lw=2):
    """Plot a-b difference trajectories, offset by starting position."""
    all_counts = list(counts_a.values()) + list(counts_b.values())
    if not all_counts:
        return
    lo, hi = min(all_counts), max(all_counts)
    for ij in trajs_a:
        if ij not in trajs_b:
            continue
        delta = trajs_a[ij] - trajs_b[ij]
        s = starts.get(ij, np.zeros(2))
        m = _marker_size(min(counts_a[ij], counts_b[ij]), lo, hi)
        _graded_line(ax, delta + s[:, None], color, lw=lw, start_ms=m)


#%%

def _draw_flow(ax, A, grid_range, pcs=(0, 1), n_grid=10):
    dA = A - np.eye(A.shape[0])
    A2 = dA[np.ix_(list(pcs), list(pcs))]
    g = np.linspace(grid_range[0], grid_range[1], n_grid)
    P1, P2 = np.meshgrid(g, g)
    ax.quiver(P1, P2,
              A2[0, 0]*P1 + A2[0, 1]*P2,
              A2[1, 0]*P1 + A2[1, 1]*P2,
              color='k', alpha=0.4, scale=None)


def _graded_line(ax, traj, color, lw=2, start_ms=5):
    c = np.asarray(color)
    pts = np.column_stack([traj[0], traj[1]])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    frac = np.linspace(0, 1, len(segs))
    lc = LineCollection(segs, colors=(1 - frac[:, None]) + c * frac[:, None],
                        linewidths=lw)
    ax.add_collection(lc)
    ax.plot(traj[0, 0], traj[1, 0], 'o', color=(c + 1) / 2,
            ms=start_ms, zorder=5)
    ax.plot(traj[0, -1], traj[1, -1], 's', color=c, ms=4, zorder=5)


def _setup_ax(ax, edges, cond=None, A_mats=None, pcs=(0, 1)):
    pc1_e, pc2_e = edges
    gr = (min(pc1_e[0], pc2_e[0]), max(pc1_e[-1], pc2_e[-1]))
    if A_mats and cond in A_mats:
        _draw_flow(ax, A_mats[cond], gr, pcs=pcs)
    for e in pc1_e[1:-1]:
        ax.axvline(e, color='grey', lw=0.5, ls='--', alpha=0.5)
    for e in pc2_e[1:-1]:
        ax.axhline(e, color='grey', lw=0.5, ls='--', alpha=0.5)
    pad1 = (pc1_e[-1] - pc1_e[0]) * 0.15
    pad2 = (pc2_e[-1] - pc2_e[0]) * 0.15
    ax.set_xlim(pc1_e[0] - pad1, pc1_e[-1] + pad1)
    ax.set_ylim(pc2_e[0] - pad2, pc2_e[-1] + pad2)
    ax.set_xlabel(f'PC{pcs[0]+1}')
    ax.set_ylabel(f'PC{pcs[1]+1}')


def _scatter(ax, z0, color, marker='o'):
    ax.scatter(z0[:, 0], z0[:, 1], c=[color], s=20, alpha=0.5,
               edgecolors='none', marker=marker)


def _save_or_show(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _smooth_bins(ops, event_type):
    short = ('tf', 'blOn')
    k = 'smooth_window_short' if event_type in short else 'smooth_window_long'
    return int(round(PLOT_OPTIONS[k] / ops['sp_bin_width']))


def _predict_trajectory(A, B, z0, u_val, n_steps):
    """Simulate z(t+1) = Az(t) + B*u for n_steps. Returns (n_dim, n_steps+1)."""
    z = np.zeros((A.shape[0], n_steps + 1))
    z[:, 0] = z0
    Bu = (B * u_val).ravel()
    for t in range(n_steps):
        z[:, t+1] = A @ z[:, t] + Bu
    return z


#%%

def plot_tf_trajectories(sess_dir, pca_key='event_all', ops=ANALYSIS_OPTIONS,
                         plot_win=(0, 0.5), ctx_win=None, pcs=(0, 1),
                         n_bins=3, min_ev=5, save_path=None):
    """
    2 rows (conditions) x 4 cols (pos&neg | pos-neg | scatter | mean+predicted).
    """
    sess_dir = Path(sess_dir)
    sess_data, area_mask, weights, A_mats, B_mats = _load_session(sess_dir, pca_key)
    if sess_data is None:
        return None
    if ctx_win is None:
        ctx_win = ops['tf_context']

    pols = ['pos', 'neg']
    pol_col = PLOT_OPTIONS['colours']['tf']
    smooth_w = _smooth_bins(ops, 'tf')

    keys = [f'{c}_{p}' for c in COND_LABELS for p in pols]
    projected, t_ax = _load_projected(
        sess_dir / 'psths.h5', 'tf', keys, area_mask, weights, pcs=pcs)
    if not projected:
        return None

    t0 = np.argmin(np.abs(t_ax - plot_win[0]))
    te = np.argmin(np.abs(t_ax - plot_win[1]))
    ctx_s, ctx_e = _ctx_idx(t_ax, ctx_win)
    n_steps = te - t0

    fig, axes = plt.subplots(2, 4, figsize=(25, 11), squeeze=False)
    diff_col = (0.5, 0.2, 0.7)

    for row, cond in enumerate(COND_LABELS):
        ax, ax_d, ax_s, ax_m = axes[row]

        # pre-event states per polarity
        z0 = {}
        for p in pols:
            k = f'{cond}_{p}'
            if k in projected:
                z0[p] = projected[k][:, :, ctx_s:ctx_e].mean(axis=2)
        if not z0:
            continue

        edges = _bin_edges(np.concatenate(list(z0.values())), n_bins)
        for a in [ax, ax_d, ax_m]:
            _setup_ax(a, edges, cond, A_mats, pcs=pcs)
        _setup_ax(ax_s, edges, pcs=pcs)

        # scatter
        for p in pols:
            if p in z0:
                _scatter(ax_s, z0[p], pol_col[p])

        # bin and plot per polarity
        binned = {}
        for p in pols:
            if p not in z0:
                continue
            trajs, counts, starts = _bin_and_smooth(
                projected[f'{cond}_{p}'], z0[p], edges,
                slice(t0, te+1), smooth_w, n_bins, min_ev)
            binned[p] = (trajs, counts, starts)
            _plot_binned(ax, trajs, counts, pol_col[p])

        # difference
        if 'pos' in binned and 'neg' in binned:
            starts = {ij: binned['pos'][2].get(ij, binned['neg'][2].get(ij))
                       for ij in set(binned['pos'][2]) | set(binned['neg'][2])}
            _plot_diff(ax_d, binned['pos'][0], binned['neg'][0],
                       binned['pos'][1], binned['neg'][1], starts, diff_col)

        # mean trajectory (all events, no binning) + LDS predicted
        for p in pols:
            k = f'{cond}_{p}'
            if k not in projected:
                continue
            mean_traj = causal_boxcar(projected[k].mean(axis=0), smooth_w)[:, t0:te+1]
            _graded_line(ax_m, mean_traj, pol_col[p], lw=2.5)

            if cond in A_mats and cond in B_mats:
                z0_full = np.zeros(A_mats[cond].shape[0])
                z0_full[list(pcs)] = mean_traj[:, 0]
                u = 1.0 if p == 'pos' else -1.0
                pred = _predict_trajectory(A_mats[cond], B_mats[cond],
                                            z0_full, u, n_steps)
                ax_m.plot(pred[pcs[0]], pred[pcs[1]], '--', color=pol_col[p],
                          lw=1.5, alpha=0.6)

        ax.set_ylabel(f'{COND_NAMES[cond]}\nPC2')
        if row == 0:
            ax.set_title('pos & neg')
            ax_d.set_title('pos - neg')
            ax_s.set_title('pre-event state')
            ax_m.set_title('mean + predicted')

    h = [Line2D([0], [0], color=pol_col[p], lw=2.5, label=f'{p} TF') for p in pols]
    h.append(Line2D([0], [0], color=diff_col, lw=2, label='pos-neg'))
    h.append(Line2D([0], [0], color='k', ls='--', lw=1.5, label='LDS predicted'))
    axes[0, 0].legend(handles=h, fontsize=8)

    fig.suptitle(f'{sess_data.animal}_{sess_data.name} — TF pulses ({pca_key})')
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


#%%

def plot_bl_trajectories(sess_dir, pca_key='event_all', ops=ANALYSIS_OPTIONS,
                         plot_win=(0, 3.0), ctx_win=(-1.0, 0.0), pcs=(0, 1),
                         n_bins=3, min_ev=5, save_path=None):
    """
    2 rows (flow conditions) x 4 cols (early&late | early-late | scatter | mean).
    """
    sess_dir = Path(sess_dir)
    sess_data, area_mask, weights, A_mats, B_mats = _load_session(sess_dir, pca_key)
    if sess_data is None:
        return None

    blocks = ['early', 'late']
    bcol = PLOT_OPTIONS['colours']['block']
    smooth_w = _smooth_bins(ops, 'bl')

    projected, t_ax = _load_projected(
        sess_dir / 'psths.h5', 'bl', blocks, area_mask, weights, pcs=pcs)
    if not projected:
        return None

    t0 = np.argmin(np.abs(t_ax - plot_win[0]))
    te = np.argmin(np.abs(t_ax - plot_win[1]))
    ctx_s, ctx_e = _ctx_idx(t_ax, ctx_win)

    z0 = {b: projected[b][:, :, ctx_s:ctx_e].mean(axis=2)
          for b in blocks if b in projected}
    if not z0:
        return None
    edges = _bin_edges(np.concatenate(list(z0.values())), n_bins)

    binned = {}
    for b in blocks:
        if b not in z0:
            continue
        trajs, counts, starts = _bin_and_smooth(
            projected[b], z0[b], edges,
            slice(t0, te+1), smooth_w, n_bins, min_ev)
        binned[b] = (trajs, counts, starts)

    fig, axes = plt.subplots(2, 4, figsize=(25, 11), squeeze=False)
    diff_col = (0.3, 0.3, 0.3)

    for row, cond in enumerate(COND_LABELS):
        ax, ax_d, ax_s, ax_m = axes[row]

        for a in [ax, ax_d, ax_m]:
            _setup_ax(a, edges, cond, A_mats, pcs=pcs)
        _setup_ax(ax_s, edges, pcs=pcs)

        for b in blocks:
            if b in z0:
                _scatter(ax_s, z0[b], bcol[b])
            if b in binned:
                _plot_binned(ax, binned[b][0], binned[b][1], bcol[b])

        if 'early' in binned and 'late' in binned:
            starts = {ij: binned['early'][2].get(ij, binned['late'][2].get(ij))
                       for ij in set(binned['early'][2]) | set(binned['late'][2])}
            _plot_diff(ax_d, binned['early'][0], binned['late'][0],
                       binned['early'][1], binned['late'][1], starts, diff_col)

        # mean trajectory
        for b in blocks:
            if b not in projected:
                continue
            mean_traj = causal_boxcar(projected[b].mean(axis=0), smooth_w)[:, t0:te+1]
            _graded_line(ax_m, mean_traj, bcol[b], lw=2.5)

        ax.set_ylabel(f'{COND_NAMES[cond]}\nPC2')
        if row == 0:
            ax.set_title('early & late block')
            ax_d.set_title('early - late')
            ax_s.set_title('pre-event state')
            ax_m.set_title('mean')

    h = [Line2D([0], [0], color=bcol[b], lw=2.5, label=f'{b} block') for b in blocks]
    h.append(Line2D([0], [0], color=diff_col, lw=2, label='early-late'))
    axes[0, 0].legend(handles=h, fontsize=8)

    fig.suptitle(f'{sess_data.animal}_{sess_data.name} — baseline ({pca_key})')
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


#%%

def plot_lick_trajectories(sess_dir, pca_key='event_all', ops=ANALYSIS_OPTIONS,
                           plot_win=(-1.0, 0.0), ctx_win=(-1.5, -1.0), pcs=(0, 1),
                           n_bins=3, min_ev=3, save_path=None):
    """
    4 rows (FA, Hit, scatter FA, scatter Hit) x 2 cols (flow conditions).
    Each panel overlays both block conditions. Extra col for mean trajectory.
    """
    sess_dir = Path(sess_dir)
    sess_data, area_mask, weights, A_mats, B_mats = _load_session(sess_dir, pca_key)
    if sess_data is None:
        return None

    outcomes = ['fa', 'hit']
    bcol = PLOT_OPTIONS['colours']['block']
    cond_col = {'earlyBlock_early': bcol['early'],
                'lateBlock_early':  bcol['late']}
    smooth_w = _smooth_bins(ops, 'lick')

    keys = [f'{c}_{o}' for c in COND_LABELS for o in outcomes]
    projected, t_ax = _load_projected(
        sess_dir / 'psths.h5', 'lick', keys, area_mask, weights, pcs=pcs)
    if not projected:
        return None

    ts = np.argmin(np.abs(t_ax - plot_win[0]))
    te = np.argmin(np.abs(t_ax - plot_win[1]))
    ctx_s, ctx_e = _ctx_idx(t_ax, ctx_win)

    z0 = {k: projected[k][:, :, ctx_s:ctx_e].mean(axis=2) for k in projected}
    edges = _bin_edges(np.concatenate(list(z0.values())), n_bins)

    binned = {}
    for k in projected:
        trajs, counts, _ = _bin_and_smooth(
            projected[k], z0[k], edges,
            slice(ts, te+1), smooth_w, n_bins, min_ev)
        binned[k] = (trajs, counts)

    fig, axes = plt.subplots(4, 3, figsize=(19, 22), squeeze=False)

    for col, cond in enumerate(COND_LABELS):
        # trajectory rows
        for row, outcome in enumerate(outcomes):
            ax = axes[row, col]
            _setup_ax(ax, edges, cond, A_mats, pcs=pcs)
            for c in COND_LABELS:
                k = f'{c}_{outcome}'
                if k in binned:
                    _plot_binned(ax, binned[k][0], binned[k][1], cond_col[c])
            if row == 0:
                ax.set_title(COND_NAMES[cond])
            if col == 0:
                ax.set_ylabel(f'{outcome.upper()}\nPC{pcs[1]+1}')

        # scatter rows (one per outcome)
        for row_off, outcome in enumerate(outcomes):
            ax_s = axes[2 + row_off, col]
            _setup_ax(ax_s, edges, pcs=pcs)
            for c in COND_LABELS:
                k = f'{c}_{outcome}'
                if k in z0:
                    _scatter(ax_s, z0[k], cond_col[c])
            if col == 0:
                ax_s.set_ylabel(f'{outcome.upper()} scatter\nPC{pcs[1]+1}')

    # mean column
    for row, outcome in enumerate(outcomes):
        ax_m = axes[row, 2]
        _setup_ax(ax_m, edges, pcs=pcs)
        for c in COND_LABELS:
            k = f'{c}_{outcome}'
            if k in projected:
                mean_traj = causal_boxcar(projected[k].mean(axis=0), smooth_w)[:, ts:te+1]
                _graded_line(ax_m, mean_traj, cond_col[c], lw=2.5)
        if row == 0:
            ax_m.set_title('mean')
        if col == 0:
            ax_m.set_ylabel(f'{outcome.upper()}\nPC2')

    # hide unused bottom-right cells
    for r in [2, 3]:
        axes[r, 2].set_visible(False)

    h = [Line2D([0], [0], color=cond_col[c], lw=2.5,
                label=COND_NAMES[c]) for c in COND_LABELS]
    axes[0, 0].legend(handles=h, fontsize=8)

    fig.suptitle(f'{sess_data.animal}_{sess_data.name} — lick ({pca_key})')
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


#%%

def plot_session_dynamics(sess_dir, pca_key='event_all',
                          ops=ANALYSIS_OPTIONS, pcs=(0, 1), save_dir=None):
    sess_dir = Path(sess_dir)
    pc_tag = f'pcs_{pcs[0]+1}{pcs[1]+1}'
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        paths = {k: str(save_dir / f'{k}_{pc_tag}.png')
                 for k in ('tf_pulses', 'bl_trajectories', 'lick_trajectories')}
    else:
        paths = {k: None for k in ('tf_pulses', 'bl_trajectories',
                                    'lick_trajectories')}

    plot_tf_trajectories(sess_dir, pca_key, ops, pcs=pcs, save_path=paths['tf_pulses'])
    plot_bl_trajectories(sess_dir, pca_key, ops, pcs=pcs, save_path=paths['bl_trajectories'])
    plot_lick_trajectories(sess_dir, pca_key, ops, pcs=pcs, save_path=paths['lick_trajectories'])
