"""
Visualisation for projected trajectories in PC space.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set_style("whitegrid")

from config import ANALYSIS_OPTIONS, PATHS, PLOT_COLOURS
from data.session import Session
from utils.rois import in_any_area, in_group


def _get_area_mask(areas, pca_key):
    """Return boolean mask for the neurons used by a given pca_key."""
    group_name = pca_key.split('_', 1)[1]
    if group_name == 'all':
        return in_any_area(areas)
    return in_group(areas, group_name)


def _smooth_sigma(ops, multiplier):
    """Smoothing sigma in bins for trajectory plotting."""
    return multiplier * ops['sp_smooth_width'] / ops['sp_bin_width']


def plot_flow_field(A, ax, grid_range=(-3, 3), n_grid=15,
                    color='k', alpha=0.4):
    """
    Plot flow field for dynamics matrix A in PC1/PC2 space.
    Uses the top-left 2x2 block of (A - I).
    """
    A_2d = (A - np.eye(A.shape[0]))[:2, :2]

    g = np.linspace(grid_range[0], grid_range[1], n_grid)
    P1, P2 = np.meshgrid(g, g)
    U = A_2d[0, 0] * P1 + A_2d[0, 1] * P2
    V = A_2d[1, 0] * P1 + A_2d[1, 1] * P2

    ax.quiver(P1, P2, U, V, color=color, alpha=alpha, scale=None)
    ax.set_aspect('equal')


def project_to_pc12(mean_psth, weights, area_mask=None):
    """
    Project mean PSTH (nN x nT) into PC1/PC2 space.
    Returns (2, nT) or None.
    """
    if mean_psth is None:
        return None
    if area_mask is not None:
        mean_psth = mean_psth[area_mask]
    z = weights.T @ mean_psth  # (n_pcs, nT)
    return z[:2]               # (2, nT)


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


def plot_session_dynamics(sess_dir,
                          pca_key='event_all',
                          lds_cond='lateBlock_early',
                          ops=ANALYSIS_OPTIONS,
                          save_path=None):
    """
    Summary figure for one session: mean PSTH trajectories in PC1/PC2 space.

    Layout (2 rows x 4 cols):
        row 0: baseline onset / TF (x3 block/time conditions)
        row 1: lick FA / lick hit / change hit / change miss
    """
    sess_data = Session.load(str(sess_dir / 'session.pkl'))

    areas = sess_data.unit_info['brain_region_comb'].values
    area_mask = _get_area_mask(areas, pca_key)

    with h5py.File(sess_dir / 'pca.h5', 'r') as f:
        weights = f[pca_key]['weights'][:]
    lds_path = sess_dir / f'lds_{pca_key}.h5'
    A = None
    if lds_path.exists():
        with h5py.File(lds_path, 'r') as f:
            if lds_cond in f:
                A = f[lds_cond]['A'][:]

    # load all mean PSTHs in one file open
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
                # baseline subtract
                t = t_axes.get(et)
                if t is not None:
                    bl = t < 0
                    if bl.any():
                        data = data - np.nanmean(data[:, bl], axis=1, keepdims=True)
                means[(et, cond)] = data

    def get_traj(event_type, condition):
        key = (event_type, condition)
        if key not in means:
            return None, None
        z = weights.T @ means[key]
        return z[:2], t_axes[event_type]

    sigma_short = _smooth_sigma(ops, 5)
    sigma_long = _smooth_sigma(ops, 5)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # row 0: baseline onset + 3 TF conditions
    c_block = PLOT_COLOURS['block']
    for block, color in c_block.items():
        traj, t = get_traj('blOn', block)
        if traj is not None:
            plot_trajectory(traj, t, axes[0, 0], color=color,
                            label=f'{block} block', smooth_sigma=sigma_short)
    axes[0, 0].set_title('Baseline onset')
    axes[0, 0].legend(fontsize=7)

    c_tf = PLOT_COLOURS['tf']
    for col, bl in enumerate(['earlyBlock_early', 'lateBlock_early',
                               'lateBlock_late'], start=1):
        for pol, color in c_tf.items():
            traj, t = get_traj('tf', f'{bl}_{pol}')
            if traj is not None:
                plot_trajectory(traj, t, axes[0, col], color=color,
                                label=pol, smooth_sigma=sigma_short)
        axes[0, col].set_title(f'TF: {bl}')
        axes[0, col].legend(fontsize=7)

    # row 1: lick FA, lick hit, change hit, change miss
    for outcome_idx, outcome in enumerate(['fa', 'hit']):
        ax = axes[1, outcome_idx]
        for cond_key in ['earlyBlock_early', 'lateBlock_early', 'lateBlock_late']:
            block = 'early' if cond_key.startswith('early') else 'late'
            color = c_block[block]
            traj, t = get_traj('lick', f'{cond_key}_{outcome}')
            if traj is not None:
                plot_trajectory(traj, t, ax, color=color, label=cond_key,
                                lw=1.5, smooth_sigma=sigma_long)
        ax.set_title(f'Lick ({outcome})')
        ax.legend(fontsize=7)

    # change hit / change miss
    ch_conds = [c for et, c in means if et == 'ch']
    for ch_idx, outcome in enumerate(['hit', 'miss']):
        ax = axes[1, ch_idx + 2]
        prefix = f'late_{outcome}_tf'
        matching = sorted([c for c in ch_conds if c.startswith(prefix)])
        if not matching:
            continue

        tf_vals = sorted([float(c.split('_tf')[-1]) for c in matching])
        cmap = plt.cm.get_cmap(PLOT_COLOURS['ch_tf_cmap'])
        colors = cmap(np.linspace(0.15, 0.85, len(tf_vals)))

        for tf_val, color in zip(tf_vals, colors):
            traj, t = get_traj('ch', f'{prefix}{tf_val}')
            if traj is not None:
                plot_trajectory(traj, t, ax, color=color,
                                label=f'tf={tf_val}', smooth_sigma=sigma_long)
        ax.set_title(f'Change (late block, {outcome})')
        ax.legend(fontsize=6)

    # share axis limits within event groups, then add flow fields
    axis_groups = [axes[0, :], axes[1, :2], axes[1, 2:]]
    for group in axis_groups:
        xlims = [ax.get_xlim() for ax in group]
        ylims = [ax.get_ylim() for ax in group]
        shared_x = (min(lo for lo, _ in xlims), max(hi for _, hi in xlims))
        shared_y = (min(lo for lo, _ in ylims), max(hi for _, hi in ylims))
        grid_range = (min(shared_x[0], shared_y[0]),
                      max(shared_x[1], shared_y[1]))
        for ax in group:
            ax.set_xlim(shared_x)
            ax.set_ylim(shared_y)
            plot_flow_field(A, ax, grid_range=grid_range)

    for ax in axes.flat:
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    fig.suptitle(f'{sess_data.animal}_{sess_data.name} (pca={pca_key}, lds={lds_cond})',
                 fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return fig
